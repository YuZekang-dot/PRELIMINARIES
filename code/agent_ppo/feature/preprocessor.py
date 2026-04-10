#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery feature preprocessor.
智运无人机特征预处理器。
"""


import numpy as np
from agent_ppo.conf.conf import Config

ACTION_DELTAS = [
    (1, 0),   # 0: right / 右
    (1, -1),  # 1: up-right / 右上
    (0, -1),  # 2: up / 上
    (-1, -1), # 3: up-left / 左上
    (-1, 0),  # 4: left / 左
    (-1, 1),  # 5: down-left / 左下
    (0, 1),   # 6: down / 下
    (1, 1),   # 7: down-right / 右下
]

MODE_WAREHOUSE_REFILL = "warehouse_refill"
MODE_STATION = "station"
MODE_CHARGER = "charger"
MODE_WAREHOUSE_CHARGE = "warehouse_charge"


def norm(v, max_v, min_v=0):
    """Normalize v to [0, 1].

    将 v 归一化到 [0, 1]。
    """
    v = np.clip(v, min_v, max_v)
    return (v - min_v) / (max_v - min_v)


def _get_entity_feature(found, cur_pos, target_pos, extra_flag=0.0):
    """Compute 7D entity feature relative to current position.

    计算实体位置相对于当前位置的 7 维特征。
    """
    relative_pos = (target_pos[0] - cur_pos[0], target_pos[1] - cur_pos[1])
    euclid_dist = np.sqrt(relative_pos[0] ** 2 + relative_pos[1] ** 2)
    chebyshev_dist = max(abs(relative_pos[0]), abs(relative_pos[1]))
    abs_norm = norm(np.array(target_pos), Config.MAX_COORD, 0)
    return np.array(
        [
            float(found),
            norm(relative_pos[0] / max(euclid_dist, 1e-4), 1, -1),
            norm(relative_pos[1] / max(euclid_dist, 1e-4), 1, -1),
            abs_norm[0],
            abs_norm[1],
            norm(chebyshev_dist, Config.MAX_CHEBYSHEV_DIST),
            float(extra_flag),
        ]
    )


def _get_rect_bounds(center_pos, width, height):
    """Build rectangle bounds from center position and extent.

    基于中心点和宽高生成矩形边界。
    """
    half_w = max(float(width) - 1.0, 0.0) / 2.0
    half_h = max(float(height) - 1.0, 0.0) / 2.0
    return (
        center_pos[0] - half_w,
        center_pos[0] + half_w,
        center_pos[1] - half_h,
        center_pos[1] + half_h,
    )


def _clip_to_rect(pos, rect_bounds):
    """Project a position to the nearest point in the rectangle.

    把位置投影到矩形区域上的最近点。
    """
    if rect_bounds is None:
        return pos

    x_min, x_max, z_min, z_max = rect_bounds
    return (
        float(np.clip(pos[0], x_min, x_max)),
        float(np.clip(pos[1], z_min, z_max)),
    )


def _chebyshev_distance_to_rect(pos, rect_bounds):
    """Chebyshev distance from a point to an axis-aligned rectangle.

    点到轴对齐矩形区域的最小 Chebyshev 距离。
    """
    if pos is None or rect_bounds is None:
        return float("inf")

    x_min, x_max, z_min, z_max = rect_bounds
    dx = max(x_min - pos[0], 0.0, pos[0] - x_max)
    dz = max(z_min - pos[1], 0.0, pos[1] - z_max)
    return max(dx, dz)


class Preprocessor:
    """feature preprocessor for Drone Delivery.

    智运无人机预处理器，仅保留最少信息。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state.

        重置所有状态。
        """
        self.cur_pos = (0, 0)

        # Game state / 游戏状态
        self.battery = 100
        self.battery_max = 100
        self.packages = []
        self.delivered = 0
        self.last_delivered = 0
        self.step_no = 0

        # Entities / 实体
        self.stations = []
        self.chargers = []
        self.warehouses = []
        self.npcs = []
        self.map_info = None
        self.env_legal_act = [1] * Config.ACTION_NUM
        self.charger_count = 0
        self.warehouse_count = 0
        self.package_count = 0
        self.terminated = False
        self.truncated = False

        # Goal state / 锁定目标状态
        self.mode = None
        self.target_id = None
        self.target_pos = None
        self.target_bounds = None

        # Previous snapshot / 上一时刻快照
        self.prev_pos = None
        self.prev_delivered = None
        self.prev_package_count = None
        self.prev_charger_count = None
        self.prev_warehouse_count = None
        self.prev_mode = None
        self.prev_target_id = None
        self.prev_target_pos = None
        self.prev_target_bounds = None
        self.prev_nearest_npc_dist = None

    def _parse_obs(self, env_obs):
        """Parse essential fields from observation dict.

        从 observation 字典中解析必要字段。
        """
        obs = env_obs["observation"]
        frame_state = obs["frame_state"]

        hero = frame_state["heroes"]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        self.battery = hero.get("battery", self.battery_max)
        self.battery_max = hero.get("battery_max", 100)
        self.packages = hero.get("packages", [])
        self.package_count = len(self.packages)

        self.last_delivered = self.delivered
        self.delivered = hero.get("delivered", 0)
        self.step_no = obs.get("step_no", 0)
        self.terminated = bool(env_obs.get("terminated", False))
        self.truncated = bool(env_obs.get("truncated", False))

        self.stations = []
        self.chargers = []
        self.warehouses = []
        for organ in frame_state.get("organs", []):
            st = organ.get("sub_type", 0)
            if st == 3:
                self.stations.append(organ)
            elif st == 2:
                self.chargers.append(organ)
            elif st == 1:
                self.warehouses.append(organ)

        self.npcs = frame_state.get("npcs", [])

        env_info = obs.get("env_info", {})
        self.charger_count = int(env_info.get("charger_count", 0))
        self.warehouse_count = int(env_info.get("warehouse_count", 0))

        # MapInfo protocol uses obs["map_info"] as the local 21x21 traversable grid.
        # For robustness, also accept the nested form {"map_info": ...} if present.
        map_info = obs.get("map_info")
        if isinstance(map_info, dict):
            map_info = map_info.get("map_info")
        self.map_info = map_info

        self.env_legal_act = obs.get("legal_action", [1] * Config.ACTION_NUM)

    def feature_process(self, env_obs, last_action):
        """Core feature extraction. Returns (feature, legal_action, reward).

        核心特征提取方法，返回完整特征向量、合法动作掩码和奖励。
        """
        self._parse_obs(env_obs)
        self._update_goal_state()

        # 1. Hero state features (4D) / 英雄状态特征（4D）
        battery_ratio = norm(self.battery, self.battery_max)
        package_count_norm = norm(len(self.packages), 3)
        cur_pos_norm = norm(np.array(self.cur_pos, dtype=float), Config.MAX_COORD, 0)
        hero_feat = np.array(
            [
                battery_ratio,
                package_count_norm,
                cur_pos_norm[0],
                cur_pos_norm[1],
            ]
        )
        battery_low = 1.0 if (self.battery / max(self.battery_max, 1)) < 0.3 else 0.0

        # 2. Nearest 3 station features (21D) / 最近 3 个驿站特征（21D）
        # Target stations first, then by distance
        # 目标驿站优先，然后按距离排序
        target_ids = set(self.packages)

        def station_sort_key(s):
            is_tgt = s.get("config_id", 0) in target_ids
            dist = self._get_region_distance(self.cur_pos, self._get_station_bounds(s))
            return (0 if is_tgt else 1, dist)

        sorted_stations = sorted(self.stations, key=station_sort_key)
        station_feat_list = []
        target_visible = 0.0
        for i in range(Config.STATION_TOPK):
            if i < len(sorted_stations):
                s = sorted_stations[i]
                is_target = s.get("config_id", 0) in target_ids
                station_feat_list.append(
                    _get_entity_feature(
                        True,
                        self.cur_pos,
                        (s["pos"]["x"], s["pos"]["z"]),
                        extra_flag=float(is_target),
                    )
                )
                target_visible = max(target_visible, float(is_target))
            else:
                station_feat_list.append(_get_entity_feature(False, self.cur_pos, self.cur_pos, extra_flag=0.0))
        station_feat = np.concatenate(station_feat_list)

        # 3. Nearest 1 charger feature (7D) / 最近 1 个充电桩特征（7D）
        sorted_chargers = sorted(
            self.chargers,
            key=lambda c: self._get_region_distance(self.cur_pos, self._get_charger_bounds(c)),
        )
        if len(sorted_chargers) > 0:
            c = sorted_chargers[0]
            charger_feat = _get_entity_feature(
                True,
                self.cur_pos,
                (c["pos"]["x"], c["pos"]["z"]),
                extra_flag=battery_low,
            )
        else:
            charger_feat = _get_entity_feature(False, self.cur_pos, self.cur_pos, extra_flag=0.0)

        # 4. Nearest 1 NPC feature (7D) / 最近 1 个官方无人机特征（7D）
        sorted_npcs = sorted(
            self.npcs,
            key=lambda n: self._chebyshev_distance(self.cur_pos, (n["pos"]["x"], n["pos"]["z"])),
        )
        if len(sorted_npcs) > 0:
            n = sorted_npcs[0]
            npc_pos = (n["pos"]["x"], n["pos"]["z"])
            is_threat_close = float(abs(npc_pos[0] - self.cur_pos[0]) <= 10 and abs(npc_pos[1] - self.cur_pos[1]) <= 10)
            npc_feat = _get_entity_feature(
                True,
                self.cur_pos,
                npc_pos,
                extra_flag=is_threat_close,
            )
        else:
            npc_feat = _get_entity_feature(False, self.cur_pos, self.cur_pos, extra_flag=0.0)

        # 5. Legal action mask (8D) / 合法动作掩码（8D）
        legal_action = self._get_legal_action()

        # 6. Binary indicators (3D) / 二值指示器（3D）
        has_package = 1.0 if len(self.packages) > 0 else 0.0
        indicators = np.array([has_package, battery_low, target_visible])

        # 7. Warehouse feature (5D) / 仓库特征（5D）
        warehouse_feat = self._get_warehouse_feature()

        # 8. Mode feature (4D) / 模式 one-hot（4D）
        mode_feat = self._get_mode_feature()

        # 9. Target feature (5D) / 当前目标特征（5D）
        target_feat = self._get_target_feature()

        # 10. Local spatial map (21x21x3) / 局部空间图（21x21x3）
        local_map_feat = self._get_local_map_feature()

        # Concatenate features / 拼接标量特征 + 局部地图特征
        feature = np.concatenate(
            [
                hero_feat,
                station_feat,
                charger_feat,
                npc_feat,
                np.array(legal_action, dtype=float),
                indicators,
                warehouse_feat,
                mode_feat,
                target_feat,
                local_map_feat,
            ]
        )

        reward = self._reward_process()
        self._snapshot_state()

        return feature, legal_action, reward

    def _get_progress_coef(self, mode):
        """Get progress reward coefficient by mode.

        按模式获取势函数奖励系数。
        """
        if mode == MODE_STATION:
            return Config.PROGRESS_REWARD_STATION
        if mode == MODE_CHARGER:
            return Config.PROGRESS_REWARD_CHARGER
        if mode == MODE_WAREHOUSE_REFILL:
            return Config.PROGRESS_REWARD_WAREHOUSE_REFILL
        if mode == MODE_WAREHOUSE_CHARGE:
            return Config.PROGRESS_REWARD_WAREHOUSE_CHARGE
        return 0.0

    def _get_direction_feature(self, from_pos, to_pos):
        """Encode direction from one position to another as 2D normalized feature.

        把从起点指向终点的方向编码为二维归一化特征。
        """
        if from_pos is None or to_pos is None:
            return np.array([0.5, 0.5], dtype=float)

        relative_pos = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
        euclid_dist = np.sqrt(relative_pos[0] ** 2 + relative_pos[1] ** 2)
        return np.array(
            [
                norm(relative_pos[0] / max(euclid_dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(euclid_dist, 1e-4), 1, -1),
            ],
            dtype=float,
        )

    def _get_organ_pos(self, organ):
        """Extract organ position.

        获取物件位置。
        """
        # Assume organ["pos"] is already the semantic center point.
        # 假设 organ["pos"] 本身就是语义中心点，不再使用 w/h 做额外偏移。
        return (float(organ["pos"]["x"]), float(organ["pos"]["z"]))

    def _get_station_bounds(self, station):
        """Get station area bounds.

        获取驿站矩形区域边界。
        """
        if station is None:
            return None
        return _get_rect_bounds(self._get_organ_pos(station), width=3.0, height=3.0)

    def _get_charger_bounds(self, charger):
        """Get charger area bounds.

        获取充电桩作用区域边界。
        """
        if charger is None:
            return None

        center_pos = self._get_organ_pos(charger)
        charge_radius = max(float(charger.get("range", 1.0)), 1.0)
        return (
            center_pos[0] - charge_radius,
            center_pos[0] + charge_radius,
            center_pos[1] - charge_radius,
            center_pos[1] + charge_radius,
        )

    def _get_warehouse_bounds(self, warehouse):
        """Get warehouse area bounds.

        获取仓库区域边界。
        """
        if warehouse is None:
            return None

        center_pos = self._get_organ_pos(warehouse)
        width = max(float(warehouse.get("w", 1.0)), 1.0)
        height = max(float(warehouse.get("h", 1.0)), 1.0)
        return _get_rect_bounds(center_pos, width=width, height=height)

    def _get_region_distance(self, pos, rect_bounds):
        """Chebyshev distance from a position to a region.

        位置到区域的最小 Chebyshev 距离。
        """
        return _chebyshev_distance_to_rect(pos, rect_bounds)

    def _get_target_region_distance(self, pos, target_bounds=None, target_pos=None):
        """Distance from a position to the current target region.

        位置到当前目标区域的最小 Chebyshev 距离。
        """
        if target_bounds is not None:
            return self._get_region_distance(pos, target_bounds)
        if target_pos is not None:
            return self._chebyshev_distance(pos, target_pos)
        return float("inf")

    def _get_target_projection(self, target_bounds=None, target_pos=None):
        """Get the nearest point on target region from current position.

        获取当前点投影到目标区域上的最近点。
        """
        if target_bounds is not None:
            return _clip_to_rect(self.cur_pos, target_bounds)
        return target_pos if target_pos is not None else self.cur_pos

    def _get_warehouse_feature(self):
        """Build explicit warehouse feature.

        构造显式仓库特征。
        """
        warehouse, warehouse_pos, _ = self._get_nearest_warehouse()
        if warehouse is None or warehouse_pos is None:
            return np.zeros(Config.WAREHOUSE_DIM, dtype=float)

        warehouse_bounds = self._get_warehouse_bounds(warehouse)
        warehouse_pos_norm = norm(np.array(warehouse_pos, dtype=float), Config.MAX_COORD, 0)
        warehouse_h_norm = norm(float(warehouse.get("h", 1.0)), Config.MAP_SIZE, 0)
        warehouse_w_norm = norm(float(warehouse.get("w", 1.0)), Config.MAP_SIZE, 0)
        warehouse_dist = self._get_region_distance(self.cur_pos, warehouse_bounds)

        return np.array(
            [
                warehouse_pos_norm[0],
                warehouse_pos_norm[1],
                warehouse_h_norm,
                warehouse_w_norm,
                norm(warehouse_dist, Config.MAX_CHEBYSHEV_DIST),
            ],
            dtype=float,
        )

    def _get_mode_feature(self):
        """Build one-hot mode feature.

        构造模式 one-hot 特征。
        """
        feat = np.zeros(Config.MODE_DIM, dtype=float)
        mode_to_idx = {
            MODE_WAREHOUSE_REFILL: 0,
            MODE_STATION: 1,
            MODE_CHARGER: 2,
            MODE_WAREHOUSE_CHARGE: 3,
        }
        if self.mode in mode_to_idx:
            feat[mode_to_idx[self.mode]] = 1.0
        return feat

    def _get_target_feature(self):
        """Build explicit current target feature.

        构造显式当前目标特征。
        """
        target_pos = self.target_pos
        target_bounds = self.target_bounds
        if target_pos is None and target_bounds is None:
            return np.zeros(Config.TARGET_DIM, dtype=float)

        target_proj = self._get_target_projection(target_bounds=target_bounds, target_pos=target_pos)
        direction_feat = self._get_direction_feature(self.cur_pos, target_proj)
        if target_pos is None:
            target_pos = target_proj
        target_abs_norm = norm(np.array(target_pos, dtype=float), Config.MAX_COORD, 0)
        target_dist = self._get_target_region_distance(
            self.cur_pos,
            target_bounds=target_bounds,
            target_pos=target_pos,
        )

        return np.array(
            [
                direction_feat[0],
                direction_feat[1],
                target_abs_norm[0],
                target_abs_norm[1],
                norm(target_dist, Config.MAX_CHEBYSHEV_DIST),
            ],
            dtype=float,
        )

    def _get_local_grid(self):
        """Return local 21x21 traversability grid as float array.

        返回局部 21x21 可通行网格。
        """
        grid = self.map_info
        if not isinstance(grid, list) or len(grid) != Config.LOCAL_MAP_SIDE:
            return np.zeros((Config.LOCAL_MAP_SIDE, Config.LOCAL_MAP_SIDE), dtype=np.float32)

        grid_np = np.array(grid, dtype=np.float32)
        if grid_np.shape != (Config.LOCAL_MAP_SIDE, Config.LOCAL_MAP_SIDE):
            return np.zeros((Config.LOCAL_MAP_SIDE, Config.LOCAL_MAP_SIDE), dtype=np.float32)
        return np.clip(grid_np, 0.0, 1.0)

    def _npc_danger_value(self, dist):
        """Reward-inspired NPC danger kernel in [0, 1].

        基于奖励衰减核的 NPC 风险值。
        """
        if dist > Config.NPC_PENALTY_RADIUS:
            return 0.0
        return float(np.exp(-(max(dist, 1.0) - 1.0) / 1.5))

    def _get_local_npc_danger_map(self):
        """Build local NPC danger map.

        构造局部 NPC 风险图。
        """
        danger_map = np.zeros((Config.LOCAL_MAP_SIDE, Config.LOCAL_MAP_SIDE), dtype=np.float32)
        if len(self.npcs) == 0:
            return danger_map

        for row in range(Config.LOCAL_MAP_SIDE):
            global_z = self.cur_pos[1] + (row - Config.LOCAL_MAP_RADIUS)
            for col in range(Config.LOCAL_MAP_SIDE):
                global_x = self.cur_pos[0] + (col - Config.LOCAL_MAP_RADIUS)
                cell_pos = (global_x, global_z)
                cell_danger = 0.0
                for npc in self.npcs:
                    npc_pos = (npc["pos"]["x"], npc["pos"]["z"])
                    dist = self._chebyshev_distance(cell_pos, npc_pos)
                    cell_danger = max(cell_danger, self._npc_danger_value(dist))
                danger_map[row, col] = cell_danger

        return danger_map

    def _get_local_target_potential_map(self):
        """Build local target potential map in [0, 1].

        构造局部目标势场图。
        """
        target_bounds = self.target_bounds
        target_pos = self.target_pos
        if target_bounds is None and target_pos is None:
            return np.zeros((Config.LOCAL_MAP_SIDE, Config.LOCAL_MAP_SIDE), dtype=np.float32)

        current_dist = self._get_target_region_distance(
            self.cur_pos,
            target_bounds=target_bounds,
            target_pos=target_pos,
        )
        if not np.isfinite(current_dist):
            return np.zeros((Config.LOCAL_MAP_SIDE, Config.LOCAL_MAP_SIDE), dtype=np.float32)

        potential_map = np.zeros((Config.LOCAL_MAP_SIDE, Config.LOCAL_MAP_SIDE), dtype=np.float32)
        scale = float(max(Config.LOCAL_MAP_RADIUS, 1))
        for row in range(Config.LOCAL_MAP_SIDE):
            global_z = self.cur_pos[1] + (row - Config.LOCAL_MAP_RADIUS)
            for col in range(Config.LOCAL_MAP_SIDE):
                global_x = self.cur_pos[0] + (col - Config.LOCAL_MAP_RADIUS)
                cell_pos = (global_x, global_z)
                cell_dist = self._get_target_region_distance(
                    cell_pos,
                    target_bounds=target_bounds,
                    target_pos=target_pos,
                )
                improvement = np.clip((current_dist - cell_dist) / scale, -1.0, 1.0)
                potential_map[row, col] = 0.5 * (improvement + 1.0)

        return potential_map

    def _get_local_map_feature(self):
        """Build flattened 21x21x3 local map feature.

        构造展平后的 21x21x3 局部地图特征。
        """
        traversable_grid = self._get_local_grid()
        obstacle_map = 1.0 - traversable_grid
        npc_danger_map = self._get_local_npc_danger_map()
        target_potential_map = self._get_local_target_potential_map()

        local_map = np.stack(
            [obstacle_map, npc_danger_map, target_potential_map],
            axis=0,
        ).astype(np.float32)
        return local_map.reshape(-1)

    def _chebyshev_distance(self, pos_a, pos_b):
        """Chebyshev distance under 8-neighbor movement.

        8 邻接移动下的切比雪夫距离。
        """
        if pos_a is None or pos_b is None:
            return float("inf")
        return max(abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1]))

    def _can_reach_target(self, target_pos=None, target_bounds=None):
        """Check whether current battery can reach the target position or region.

        判断当前电量是否足以到达目标位置或区域。
        """
        dist = self._get_target_region_distance(
            self.cur_pos,
            target_bounds=target_bounds,
            target_pos=target_pos,
        )
        if not np.isfinite(dist):
            return False
        return self.battery >= dist

    def _get_nearest_charger(self, from_pos=None):
        """Get nearest charger from a position.

        获取距离某个位置最近的充电桩。
        """
        if from_pos is None:
            from_pos = self.cur_pos
        if len(self.chargers) == 0:
            return None, None, float("inf")

        charger = min(
            self.chargers,
            key=lambda c: self._get_region_distance(from_pos, self._get_charger_bounds(c)),
        )
        charger_pos = self._get_organ_pos(charger)
        charger_dist = self._get_region_distance(from_pos, self._get_charger_bounds(charger))
        return charger, charger_pos, charger_dist

    def _get_nearest_warehouse(self, from_pos=None):
        """Get nearest warehouse from a position.

        获取距离某个位置最近的仓库。
        """
        if from_pos is None:
            from_pos = self.cur_pos
        if len(self.warehouses) == 0:
            return None, None, float("inf")

        warehouse = min(
            self.warehouses,
            key=lambda w: self._get_region_distance(from_pos, self._get_warehouse_bounds(w)),
        )
        warehouse_pos = self._get_organ_pos(warehouse)
        warehouse_dist = self._get_region_distance(from_pos, self._get_warehouse_bounds(warehouse))
        return warehouse, warehouse_pos, warehouse_dist

    def _get_station_by_id(self, target_id):
        """Find station organ by config_id.

        通过 config_id 查找驿站。
        """
        for station in self.stations:
            if station.get("config_id", 0) == target_id:
                return station
        return None

    def _get_charger_by_id(self, target_id):
        """Find charger organ by config_id.

        通过 config_id 查找充电桩。
        """
        for charger in self.chargers:
            if charger.get("config_id", 0) == target_id:
                return charger
        return None

    def _get_nearest_target_station(self):
        """Select nearest target station from current carried packages.

        从当前携带包裹对应的驿站中选择最近目标。
        """
        target_ids = set(self.packages)
        target_stations = [station for station in self.stations if station.get("config_id", 0) in target_ids]
        if len(target_stations) == 0:
            return None

        return min(
            target_stations,
            key=lambda s: self._get_region_distance(self.cur_pos, self._get_station_bounds(s)),
        )

    def _get_nearest_charge_target(self, from_pos=None):
        """Choose nearest charging target between charger and warehouse.

        在最近充电桩和仓库之间选择最近补能点。
        """
        if from_pos is None:
            from_pos = self.cur_pos

        charger, charger_pos, charger_dist = self._get_nearest_charger(from_pos)
        warehouse, warehouse_pos, warehouse_dist = self._get_nearest_warehouse(from_pos)

        if charger_pos is None and warehouse_pos is None:
            return None, None, None, float("inf")
        if charger_dist <= warehouse_dist:
            return MODE_CHARGER, charger.get("config_id", 0), charger_pos, charger_dist
        return MODE_WAREHOUSE_CHARGE, "warehouse", warehouse_pos, warehouse_dist

    def _is_station_safe(self, station):
        """Check whether current battery can finish delivery and retreat safely.

        判断当前电量是否足以完成投递并安全撤离。
        """
        if station is None:
            return False

        station_pos = self._get_organ_pos(station)
        station_bounds = self._get_station_bounds(station)
        d_to_station = self._get_region_distance(self.cur_pos, station_bounds)
        _, _, _, d_retreat = self._get_nearest_charge_target(from_pos=station_pos)
        if not np.isfinite(d_retreat):
            return False

        need = d_to_station + d_retreat + Config.SAFETY_MARGIN
        return self.battery >= need

    def _is_current_target_valid(self):
        """Check whether the locked goal is still valid.

        判断当前锁定目标是否仍然有效。
        """
        if self.mode is None:
            return False

        if self.mode == MODE_WAREHOUSE_REFILL:
            warehouse, warehouse_pos, _ = self._get_nearest_warehouse()
            warehouse_bounds = self._get_warehouse_bounds(warehouse)
            return (
                self.package_count == 0
                and warehouse_pos is not None
                and self._can_reach_target(target_pos=warehouse_pos, target_bounds=warehouse_bounds)
            )

        if self.mode == MODE_STATION:
            return self.target_id in set(self.packages) and self._get_station_by_id(self.target_id) is not None

        if self.mode == MODE_CHARGER:
            return self._get_charger_by_id(self.target_id) is not None

        if self.mode == MODE_WAREHOUSE_CHARGE:
            return self._get_nearest_warehouse()[1] is not None

        return False

    def _refresh_target_pos(self):
        """Refresh current target position from locked mode and target id.

        根据当前锁定模式和目标编号刷新目标位置。
        """
        if self.mode == MODE_STATION:
            station = self._get_station_by_id(self.target_id)
            self.target_pos = self._get_organ_pos(station) if station is not None else None
            self.target_bounds = self._get_station_bounds(station)
        elif self.mode == MODE_CHARGER:
            charger = self._get_charger_by_id(self.target_id)
            self.target_pos = self._get_organ_pos(charger) if charger is not None else None
            self.target_bounds = self._get_charger_bounds(charger)
        elif self.mode in (MODE_WAREHOUSE_REFILL, MODE_WAREHOUSE_CHARGE):
            warehouse, warehouse_pos, _ = self._get_nearest_warehouse()
            self.target_pos = warehouse_pos
            self.target_bounds = self._get_warehouse_bounds(warehouse)
        else:
            self.target_pos = None
            self.target_bounds = None

    def _should_replan(self):
        """Determine whether the goal should be reselected.

        判断当前是否需要重新选择目标。
        """
        if self.prev_pos is None:
            return True

        charge_event = self.prev_charger_count is not None and self.charger_count > self.prev_charger_count
        warehouse_event = self.prev_warehouse_count is not None and self.warehouse_count > self.prev_warehouse_count

        if self.package_count != self.prev_package_count:
            return True
        if self.delivered != self.prev_delivered:
            return True
        if charge_event or warehouse_event:
            return True
        if not self._is_current_target_valid():
            return True
        if self.mode == MODE_STATION:
            station = self._get_station_by_id(self.target_id)
            if not self._is_station_safe(station):
                return True

        return False

    def _select_goal(self):
        """Select a new locked goal based on current task state.

        根据当前任务状态选择新的锁定目标。
        """
        if self.package_count == 0:
            warehouse, warehouse_pos, _ = self._get_nearest_warehouse()
            warehouse_bounds = self._get_warehouse_bounds(warehouse)
            if warehouse_pos is not None and self._can_reach_target(target_pos=warehouse_pos, target_bounds=warehouse_bounds):
                self.mode = MODE_WAREHOUSE_REFILL
                self.target_id = "warehouse"
                self.target_pos = warehouse_pos
                self.target_bounds = warehouse_bounds
                return

            charger, charger_pos, _ = self._get_nearest_charger()
            if charger_pos is not None:
                self.mode = MODE_CHARGER
                self.target_id = charger.get("config_id", 0)
                self.target_pos = charger_pos
                self.target_bounds = self._get_charger_bounds(charger)
                return

            self.mode = MODE_WAREHOUSE_REFILL
            self.target_id = "warehouse"
            self.target_pos = warehouse_pos
            self.target_bounds = warehouse_bounds
            return

        station = self._get_nearest_target_station()
        if station is not None and self._is_station_safe(station):
            self.mode = MODE_STATION
            self.target_id = station.get("config_id", 0)
            self.target_pos = self._get_organ_pos(station)
            self.target_bounds = self._get_station_bounds(station)
            return

        mode, target_id, target_pos, _ = self._get_nearest_charge_target()
        if target_pos is not None:
            self.mode = mode
            self.target_id = target_id
            self.target_pos = target_pos
            if mode == MODE_CHARGER:
                self.target_bounds = self._get_charger_bounds(self._get_charger_by_id(target_id))
            else:
                warehouse, _, _ = self._get_nearest_warehouse()
                self.target_bounds = self._get_warehouse_bounds(warehouse)
            return

        if station is not None:
            self.mode = MODE_STATION
            self.target_id = station.get("config_id", 0)
            self.target_pos = self._get_organ_pos(station)
            self.target_bounds = self._get_station_bounds(station)
            return

        self.mode = None
        self.target_id = None
        self.target_pos = None
        self.target_bounds = None

    def _update_goal_state(self):
        """Update locked goal state with event-triggered replanning.

        按事件触发规则更新锁定目标状态。
        """
        self._refresh_target_pos()
        if self._should_replan():
            self._select_goal()
        else:
            self._refresh_target_pos()

    def _get_nearest_npc_distance(self):
        """Get Chebyshev distance to the nearest NPC.

        获取到最近官方无人机的切比雪夫距离。
        """
        if len(self.npcs) == 0:
            return float("inf")
        return min(
            self._chebyshev_distance(self.cur_pos, (npc["pos"]["x"], npc["pos"]["z"]))
            for npc in self.npcs
        )

    def _snapshot_state(self):
        """Save current state as previous snapshot for next step.

        保存当前状态作为下一步的上一时刻快照。
        """
        self.prev_pos = self.cur_pos
        self.prev_delivered = self.delivered
        self.prev_package_count = self.package_count
        self.prev_charger_count = self.charger_count
        self.prev_warehouse_count = self.warehouse_count
        self.prev_mode = self.mode
        self.prev_target_id = self.target_id
        self.prev_target_pos = self.target_pos
        self.prev_target_bounds = self.target_bounds
        self.prev_nearest_npc_dist = self._get_nearest_npc_distance()

    def _get_legal_action(self):
        """Get legal action mask.

        获取合法动作掩码。
        """
        env_legal_action = [1] * Config.ACTION_NUM
        if hasattr(self, "env_legal_act") and self.env_legal_act:
            for i, val in enumerate(list(self.env_legal_act)[: Config.ACTION_NUM]):
                env_legal_action[i] = int(val)

        map_legal_action = self._get_map_legal_action()
        legal_action = [int(env_ok and map_ok) for env_ok, map_ok in zip(env_legal_action, map_legal_action)]

        if sum(legal_action) == 0:
            if sum(env_legal_action) > 0:
                return env_legal_action
            return [1] * Config.ACTION_NUM

        return legal_action

    def _get_map_legal_action(self):
        """Compute legal action mask from local MapInfo.

        基于局部 MapInfo 计算合法动作掩码。
        """
        grid = self.map_info
        if not isinstance(grid, list) or len(grid) == 0 or not isinstance(grid[0], list) or len(grid[0]) == 0:
            return [1] * Config.ACTION_NUM

        center_r = len(grid) // 2
        center_c = len(grid[0]) // 2

        def is_passable(row, col):
            if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[row]):
                return False
            return int(grid[row][col]) == 1

        legal_action = []
        for dx, dz in ACTION_DELTAS:
            target_r = center_r + dz
            target_c = center_c + dx
            target_passable = is_passable(target_r, target_c)

            if not target_passable:
                legal_action.append(0)
                continue

            # Diagonal movement uses anti-corner-cutting:
            # target must be passable, and at least one adjacent edge cell must be passable.
            # 斜向移动防穿角：
            # 目标格必须可通行，且与该对角相邻的水平/垂直格至少一个可通行。
            if dx != 0 and dz != 0:
                horizontal_passable = is_passable(center_r, center_c + dx)
                vertical_passable = is_passable(center_r + dz, center_c)
                legal_action.append(int(horizontal_passable or vertical_passable))
            else:
                legal_action.append(1)

        return legal_action

    def _reward_process(self):
        """Reward function.

        奖励函数。
        """
        if self.prev_pos is None:
            return [0.0]

        reward = 0.0

        charge_event = self.prev_charger_count is not None and self.charger_count > self.prev_charger_count
        warehouse_event = self.prev_warehouse_count is not None and self.warehouse_count > self.prev_warehouse_count
        newly_delivered = max(0, self.delivered - self.prev_delivered)

        # 1. Delivery reward / 投递主奖励
        reward += Config.DELIVERY_REWARD_SCALE * newly_delivered

        # 2. Progress shaping / 锁定目标的势函数奖励
        if self.mode == self.prev_mode and self.target_id == self.prev_target_id:
            target_bounds = self.target_bounds if self.target_bounds is not None else self.prev_target_bounds
            target_pos = self.target_pos if self.target_pos is not None else self.prev_target_pos
            prev_dist = self._get_target_region_distance(
                self.prev_pos,
                target_bounds=target_bounds,
                target_pos=target_pos,
            )
            cur_dist = self._get_target_region_distance(
                self.cur_pos,
                target_bounds=target_bounds,
                target_pos=target_pos,
            )
            if np.isfinite(prev_dist) and np.isfinite(cur_dist):
                reward += self._get_progress_coef(self.mode) * np.clip(prev_dist - cur_dist, -1.0, 1.0)

        # 3. Charge event / 充电事件奖励
        if charge_event and self.prev_mode == MODE_CHARGER:
            reward += Config.CHARGE_EVENT_REWARD

        # 4. Warehouse event / 仓库事件奖励
        if warehouse_event:
            if self.prev_mode == MODE_WAREHOUSE_REFILL and self.prev_package_count == 0 and self.package_count > 0:
                reward += Config.WAREHOUSE_REFILL_REWARD
            elif self.prev_mode == MODE_WAREHOUSE_CHARGE:
                reward += Config.WAREHOUSE_CHARGE_REWARD

        # 5. Smooth NPC risk penalty / 平滑 NPC 风险惩罚
        nearest_npc_dist = self._get_nearest_npc_distance()
        if nearest_npc_dist <= Config.NPC_PENALTY_RADIUS:
            reward -= Config.NPC_PENALTY_SCALE * np.exp(-(nearest_npc_dist - 1.0) / 1.5)

        # Small reward for escaping danger / 轻微远离危险奖励
        if (
            self.prev_nearest_npc_dist is not None
            and self.prev_nearest_npc_dist <= Config.NPC_ESCAPE_TRIGGER_DIST
            and np.isfinite(nearest_npc_dist)
        ):
            reward += Config.NPC_ESCAPE_REWARD_SCALE * np.clip(
                nearest_npc_dist - self.prev_nearest_npc_dist,
                -1.0,
                1.0,
            )

        # 6. Light block penalty / 轻量阻塞惩罚
        if self.cur_pos == self.prev_pos and not charge_event and not warehouse_event:
            reward -= Config.BLOCK_PENALTY

        # 7. Step penalty / 每步惩罚
        reward -= Config.STEP_PENALTY

        # 8. Terminal penalty / 终局惩罚
        if self.terminated:
            if self.battery <= 0:
                reward += Config.TERMINAL_ENERGY_DEPLETED_PENALTY
            else:
                reward += Config.TERMINAL_COLLISION_PENALTY

        return [reward]
