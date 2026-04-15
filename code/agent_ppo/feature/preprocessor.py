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


import heapq
from collections import deque

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
        self.waypoint_pos = None
        self.waypoint_goal_key = None
        self.waypoint_bfs_dist = None
        self.waypoint_no_improve_steps = 0
        self.waypoint_stuck_replans = 0
        self.waypoint_reached = False
        self.completed_waypoint_pos = None
        self.completed_waypoint_goal_key = None
        self.completed_waypoint_cur_dist = None
        self.local_bfs_distances = None
        self.local_waypoint_distances = None
        self.local_waypoint_distances_key = None
        self.global_map = np.full(
            (Config.MAP_SIZE, Config.MAP_SIZE),
            Config.GLOBAL_MAP_UNKNOWN,
            dtype=np.int8,
        )
        self.global_path = []
        self.global_path_goal_key = None
        self.global_path_cost = float("inf")
        self.visit_count = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int32)
        self.last_visit_stamp = None
        self.position_history = deque(maxlen=Config.OSCILLATION_WINDOW)
        self.last_position_stamp = None
        self.no_move_steps = 0
        self.waypoint_missing_steps = 0
        self.last_stuck_reasons = []

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
        self.prev_waypoint_pos = None
        self.prev_waypoint_goal_key = None
        self.prev_waypoint_bfs_dist = None

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
        self.local_bfs_distances = None
        self.local_waypoint_distances = None
        self.local_waypoint_distances_key = None
        self._update_entity_prior_free()
        self._update_global_map()
        self._update_visit_count()
        self._update_motion_stuck_state()

    def feature_process(self, env_obs, last_action):
        """Core feature extraction. Returns (feature, legal_action, reward).

        核心特征提取方法，返回完整特征向量、合法动作掩码和奖励。
        """
        self._parse_obs(env_obs)
        self._update_goal_state()
        self._update_waypoint_state()

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

    def _has_valid_local_grid(self):
        """Return whether the current local map has the expected shape."""
        grid = self.map_info
        if not isinstance(grid, list) or len(grid) != Config.LOCAL_MAP_SIDE:
            return False
        if any(not isinstance(row, list) or len(row) != Config.LOCAL_MAP_SIDE for row in grid):
            return False
        return True

    def _local_cell_to_global_pos(self, row, col):
        """Convert a local grid cell into global map coordinates."""
        center = Config.LOCAL_MAP_RADIUS
        return (
            int(round(self.cur_pos[0])) + (col - center),
            int(round(self.cur_pos[1])) + (row - center),
        )

    def _global_pos_to_local_cell(self, pos):
        """Convert a global map coordinate into the current local grid cell."""
        if pos is None:
            return None
        center = Config.LOCAL_MAP_RADIUS
        row = center + (int(round(pos[1])) - int(round(self.cur_pos[1])))
        col = center + (int(round(pos[0])) - int(round(self.cur_pos[0])))
        if row < 0 or row >= Config.LOCAL_MAP_SIDE or col < 0 or col >= Config.LOCAL_MAP_SIDE:
            return None
        return row, col

    def _to_grid_pos(self, pos):
        """Convert a position-like value to an integer global cell."""
        if pos is None:
            return None
        return (int(round(pos[0])), int(round(pos[1])))

    def _update_global_map(self):
        """Merge the current 21x21 local MapInfo into the per-episode global map."""
        if not self._has_valid_local_grid():
            return

        grid = self._get_local_grid()
        center = Config.LOCAL_MAP_RADIUS
        base_x = int(round(self.cur_pos[0]))
        base_z = int(round(self.cur_pos[1]))
        for row in range(Config.LOCAL_MAP_SIDE):
            global_z = base_z + (row - center)
            if global_z < 0 or global_z >= Config.MAP_SIZE:
                continue
            for col in range(Config.LOCAL_MAP_SIDE):
                global_x = base_x + (col - center)
                if global_x < 0 or global_x >= Config.MAP_SIZE:
                    continue
                self.global_map[global_z, global_x] = (
                    Config.GLOBAL_MAP_FREE if int(grid[row, col]) == 1 else Config.GLOBAL_MAP_BLOCKED
                )

        if 0 <= base_x < Config.MAP_SIZE and 0 <= base_z < Config.MAP_SIZE:
            self.global_map[base_z, base_x] = Config.GLOBAL_MAP_FREE

    def _iter_rect_cells(self, rect_bounds):
        """Yield integer global cells covered by a rectangle."""
        if rect_bounds is None:
            return

        x_min, x_max, z_min, z_max = rect_bounds
        x0 = max(0, int(np.floor(x_min)))
        x1 = min(Config.MAP_SIZE - 1, int(np.ceil(x_max)))
        z0 = max(0, int(np.floor(z_min)))
        z1 = min(Config.MAP_SIZE - 1, int(np.ceil(z_max)))
        for z in range(z0, z1 + 1):
            for x in range(x0, x1 + 1):
                yield x, z

    def _mark_prior_free_region(self, rect_bounds):
        """Mark an entity region as weakly traversable without overriding observations."""
        for x, z in self._iter_rect_cells(rect_bounds):
            if self.global_map[z, x] == Config.GLOBAL_MAP_UNKNOWN:
                self.global_map[z, x] = Config.GLOBAL_MAP_PRIOR_FREE

    def _update_entity_prior_free(self):
        """Seed known entity regions as weak free priors before map observations arrive."""
        for warehouse in self.warehouses:
            self._mark_prior_free_region(self._get_warehouse_bounds(warehouse))
        for station in self.stations:
            self._mark_prior_free_region(self._get_station_bounds(station))
        for charger in self.chargers:
            self._mark_prior_free_region(self._get_charger_bounds(charger))

    def _get_visit_cost(self, pos):
        """Normalized visit cost in [0, 1] for a global position."""
        if pos is None:
            return 0.0
        x = int(round(pos[0]))
        z = int(round(pos[1]))
        if x < 0 or x >= Config.MAP_SIZE or z < 0 or z >= Config.MAP_SIZE:
            return 0.0
        return min(float(self.visit_count[z, x]), float(Config.VISIT_COUNT_CAP)) / max(
            float(Config.VISIT_COUNT_CAP),
            1.0,
        )

    def _is_in_global_map(self, pos):
        """Return whether a global position is inside the fixed 128x128 map."""
        if pos is None:
            return False
        x = int(round(pos[0]))
        z = int(round(pos[1]))
        return 0 <= x < Config.MAP_SIZE and 0 <= z < Config.MAP_SIZE

    def _update_visit_count(self):
        """Update per-episode global visit count once per observed step."""
        x = int(round(self.cur_pos[0]))
        z = int(round(self.cur_pos[1]))
        if x < 0 or x >= Config.MAP_SIZE or z < 0 or z >= Config.MAP_SIZE:
            return

        stamp = (self.step_no, x, z)
        if stamp == self.last_visit_stamp:
            return

        self.visit_count[z, x] += 1
        self.last_visit_stamp = stamp

    def _has_progress_event(self):
        """Return whether this step has a delivery, charge, or warehouse event."""
        charge_event = self.prev_charger_count is not None and self.charger_count > self.prev_charger_count
        warehouse_event = self.prev_warehouse_count is not None and self.warehouse_count > self.prev_warehouse_count
        delivery_event = self.prev_delivered is not None and self.delivered > self.prev_delivered
        package_event = self.prev_package_count is not None and self.package_count != self.prev_package_count
        return charge_event or warehouse_event or delivery_event or package_event

    def _update_motion_stuck_state(self):
        """Track recent movement patterns for faster stuck detection."""
        cur_cell = self._to_grid_pos(self.cur_pos)
        prev_cell = self._to_grid_pos(self.prev_pos)
        progress_event = self._has_progress_event()

        if cur_cell is not None and prev_cell is not None and cur_cell == prev_cell and not progress_event:
            self.no_move_steps += 1
        else:
            self.no_move_steps = 0

        stamp = (self.step_no, cur_cell)
        if cur_cell is not None and stamp != self.last_position_stamp:
            self.position_history.append(cur_cell)
            self.last_position_stamp = stamp

    def _get_local_bfs_distances(self):
        """BFS distances from the current local-map center to all reachable cells."""
        if self.local_bfs_distances is not None:
            return self.local_bfs_distances

        center = Config.LOCAL_MAP_RADIUS
        self.local_bfs_distances = self._compute_local_bfs_distances((center, center))
        return self.local_bfs_distances

    def _is_local_passable(self, grid, row, col):
        """Return whether a local-grid cell is passable."""
        if row < 0 or row >= Config.LOCAL_MAP_SIDE or col < 0 or col >= Config.LOCAL_MAP_SIDE:
            return False
        return int(grid[row, col]) == 1

    def _compute_local_bfs_distances(self, start_cell):
        """BFS distances from a local-grid start cell to all reachable cells."""
        inf = float("inf")
        distances = np.full((Config.LOCAL_MAP_SIDE, Config.LOCAL_MAP_SIDE), inf, dtype=np.float32)
        if start_cell is None or not self._has_valid_local_grid():
            return distances

        grid = self._get_local_grid()
        start_row, start_col = start_cell
        if not self._is_local_passable(grid, start_row, start_col):
            return distances

        distances[start_row, start_col] = 0.0
        queue = deque([(start_row, start_col)])
        while queue:
            row, col = queue.popleft()
            next_dist = distances[row, col] + 1.0
            for dx, dz in ACTION_DELTAS:
                next_row = row + dz
                next_col = col + dx
                if not self._is_local_passable(grid, next_row, next_col):
                    continue
                if dx != 0 and dz != 0 and not (
                    self._is_local_passable(grid, row, col + dx) or self._is_local_passable(grid, row + dz, col)
                ):
                    continue
                if next_dist < distances[next_row, next_col]:
                    distances[next_row, next_col] = next_dist
                    queue.append((next_row, next_col))

        return distances

    def _get_waypoint_distance_map(self):
        """Reverse local BFS distances from the active waypoint to each local cell."""
        waypoint_cell = self._global_pos_to_local_cell(self.waypoint_pos)
        key = (self.waypoint_goal_key, self.waypoint_pos, waypoint_cell, self.step_no)
        if self.local_waypoint_distances is not None and self.local_waypoint_distances_key == key:
            return self.local_waypoint_distances

        self.local_waypoint_distances = self._compute_local_bfs_distances(waypoint_cell)
        self.local_waypoint_distances_key = key
        return self.local_waypoint_distances

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
        if self.waypoint_pos is None:
            return np.zeros((Config.LOCAL_MAP_SIDE, Config.LOCAL_MAP_SIDE), dtype=np.float32)

        waypoint_distances = self._get_waypoint_distance_map()
        finite_mask = np.isfinite(waypoint_distances)
        finite_scores = waypoint_distances[finite_mask]
        potential_map = np.zeros((Config.LOCAL_MAP_SIDE, Config.LOCAL_MAP_SIDE), dtype=np.float32)
        if finite_scores.size == 0:
            return potential_map

        max_score = float(np.max(finite_scores))
        if max_score <= 0.0:
            potential_map[finite_mask] = 1.0
            return potential_map

        potential_map[finite_mask] = 1.0 - (waypoint_distances[finite_mask] / max_score)
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

    def _is_global_passable(self, pos):
        """Return whether a global cell can be planned through."""
        cell = self._to_grid_pos(pos)
        if cell is None:
            return False
        x, z = cell
        if x < 0 or x >= Config.MAP_SIZE or z < 0 or z >= Config.MAP_SIZE:
            return False
        return int(self.global_map[z, x]) != Config.GLOBAL_MAP_BLOCKED

    def _get_global_cell_cost(self, pos):
        """Cost of entering a global cell in A*."""
        cell = self._to_grid_pos(pos)
        if cell is None:
            return float("inf")
        x, z = cell
        if x < 0 or x >= Config.MAP_SIZE or z < 0 or z >= Config.MAP_SIZE:
            return float("inf")
        cell_type = int(self.global_map[z, x])
        if cell_type == Config.GLOBAL_MAP_BLOCKED:
            return float("inf")
        if cell_type == Config.GLOBAL_MAP_UNKNOWN:
            base_cost = float(Config.GLOBAL_MAP_UNKNOWN_COST)
        elif cell_type == Config.GLOBAL_MAP_PRIOR_FREE:
            base_cost = float(Config.GLOBAL_MAP_PRIOR_FREE_COST)
        else:
            base_cost = float(Config.GLOBAL_MAP_FREE_COST)

        visit_cost = float(Config.VISIT_GLOBAL_COST_WEIGHT) * self._get_visit_cost((x, z))
        return base_cost + visit_cost

    def _can_traverse_global_step(self, from_cell, to_cell):
        """Return whether one global 8-neighbor step is passable."""
        if not self._is_global_passable(to_cell):
            return False

        dx = int(to_cell[0]) - int(from_cell[0])
        dz = int(to_cell[1]) - int(from_cell[1])
        if dx == 0 and dz == 0:
            return False
        if abs(dx) > 1 or abs(dz) > 1:
            return False
        if dx != 0 and dz != 0:
            horizontal_passable = self._is_global_passable((from_cell[0] + dx, from_cell[1]))
            vertical_passable = self._is_global_passable((from_cell[0], from_cell[1] + dz))
            return horizontal_passable or vertical_passable
        return True

    def _is_target_cell(self, cell):
        """Return whether a global cell is inside the current target region."""
        if cell is None:
            return False
        if self.target_bounds is not None:
            return self._get_region_distance(cell, self.target_bounds) <= 0.0
        if self.target_pos is not None:
            return self._chebyshev_distance(cell, self._to_grid_pos(self.target_pos)) <= 0.0
        return False

    def _target_heuristic(self, cell):
        """Admissible Chebyshev lower bound to the current target region."""
        return self._get_target_region_distance(
            cell,
            target_bounds=self.target_bounds,
            target_pos=self.target_pos,
        )

    def _clear_global_path(self):
        """Clear the cached global A* plan."""
        self.global_path = []
        self.global_path_goal_key = None
        self.global_path_cost = float("inf")

    def _reconstruct_global_path(self, came_from, goal_cell):
        """Reconstruct an A* path from the predecessor map."""
        path = []
        cell = goal_cell
        while cell is not None:
            path.append(cell)
            cell = came_from.get(cell)
        path.reverse()
        return path

    def _plan_global_path(self, goal_key):
        """Plan a global A* path from the current cell to the current target region."""
        if goal_key is None or (self.target_bounds is None and self.target_pos is None):
            self._clear_global_path()
            return False

        start = self._to_grid_pos(self.cur_pos)
        if start is None or not self._is_in_global_map(start):
            self._clear_global_path()
            return False

        start_x, start_z = start
        self.global_map[start_z, start_x] = Config.GLOBAL_MAP_FREE

        if self._is_target_cell(start):
            self.global_path = [start]
            self.global_path_goal_key = goal_key
            self.global_path_cost = 0.0
            return True

        start_h = self._target_heuristic(start)
        if not np.isfinite(start_h):
            self._clear_global_path()
            return False

        frontier = []
        counter = 0
        heapq.heappush(frontier, (start_h, 0.0, counter, start))
        came_from = {start: None}
        cost_so_far = {start: 0.0}

        while frontier:
            _, cur_cost, _, current = heapq.heappop(frontier)
            if cur_cost > cost_so_far.get(current, float("inf")) + 1e-6:
                continue

            if self._is_target_cell(current):
                self.global_path = self._reconstruct_global_path(came_from, current)
                self.global_path_goal_key = goal_key
                self.global_path_cost = cur_cost
                return True

            for dx, dz in ACTION_DELTAS:
                next_cell = (current[0] + dx, current[1] + dz)
                if not self._can_traverse_global_step(current, next_cell):
                    continue

                step_cost = self._get_global_cell_cost(next_cell)
                if not np.isfinite(step_cost):
                    continue

                new_cost = cost_so_far[current] + step_cost
                if new_cost >= cost_so_far.get(next_cell, float("inf")):
                    continue

                heuristic = self._target_heuristic(next_cell)
                if not np.isfinite(heuristic):
                    continue

                cost_so_far[next_cell] = new_cost
                came_from[next_cell] = current
                counter += 1
                heapq.heappush(frontier, (new_cost + heuristic, new_cost, counter, next_cell))

        self._clear_global_path()
        return False

    def _get_current_path_index(self):
        """Find the closest usable index on the cached global path."""
        if len(self.global_path) == 0:
            return 0

        cur_cell = self._to_grid_pos(self.cur_pos)
        if cur_cell is None:
            return 0

        best_idx = 0
        best_dist = float("inf")
        for idx, path_cell in enumerate(self.global_path):
            dist = self._chebyshev_distance(cur_cell, path_cell)
            if dist < best_dist or (dist == best_dist and idx > best_idx):
                best_dist = dist
                best_idx = idx
        return best_idx

    def _is_global_path_blocked(self):
        """Return whether newly observed obstacles invalidate the cached path."""
        if len(self.global_path) == 0:
            return True

        start_idx = self._get_current_path_index()
        prev_cell = self.global_path[start_idx]
        if not self._is_global_passable(prev_cell):
            return True

        for idx in range(start_idx + 1, len(self.global_path)):
            cell = self.global_path[idx]
            if not self._can_traverse_global_step(prev_cell, cell):
                return True
            prev_cell = cell
        return False

    def _get_path_deviation(self):
        """Distance from current cell to the closest cached global path cell."""
        if len(self.global_path) == 0:
            return 0.0

        cur_cell = self._to_grid_pos(self.cur_pos)
        if cur_cell is None:
            return 0.0

        return min(self._chebyshev_distance(cur_cell, path_cell) for path_cell in self.global_path)

    def _is_oscillation_stuck(self):
        """Detect short-cycle movement such as A-B-A-B."""
        if len(self.position_history) < Config.OSCILLATION_WINDOW:
            return False
        unique_count = len(set(self.position_history))
        return (
            unique_count <= Config.OSCILLATION_MIN_UNIQUE_POS
            and self.waypoint_no_improve_steps >= 1
            and not self._has_progress_event()
        )

    def _is_revisit_stuck(self):
        """Detect repeated visits that also fail to make waypoint progress."""
        return (
            self._get_visit_cost(self.cur_pos) >= Config.REVISIT_STUCK_COST
            and self.waypoint_no_improve_steps >= 1
            and not self._has_progress_event()
        )

    def _should_force_global_replan(self, old_waypoint_valid):
        """Decide whether local repair is too slow and A* should be forced now."""
        reasons = []

        if self.no_move_steps >= Config.NO_MOVE_STUCK_STEPS:
            reasons.append("no_move")
        if old_waypoint_valid and self.waypoint_no_improve_steps >= Config.WAYPOINT_REPLAN_STUCK_STEPS:
            reasons.append("no_progress")
        if self.waypoint_missing_steps >= Config.WAYPOINT_MISSING_REPLAN_STEPS:
            reasons.append("waypoint_missing")
        if self._is_oscillation_stuck():
            reasons.append("oscillation")
        if self._is_revisit_stuck():
            reasons.append("revisit")
        if len(self.global_path) > 0 and self._get_path_deviation() >= Config.PATH_DEVIATION_REPLAN_DIST:
            reasons.append("path_deviation")

        self.last_stuck_reasons = reasons
        return len(reasons) > 0

    def _ensure_global_path(self, goal_key, force=False):
        """Keep a usable global path for the current goal."""
        if (
            force
            or self.global_path_goal_key != goal_key
            or len(self.global_path) == 0
            or self._is_global_path_blocked()
        ):
            return self._plan_global_path(goal_key)
        return True

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

    def _lock_goal(self, mode, target_id, target_pos, target_bounds):
        """Lock current goal fields in one place."""
        old_goal_key = self._get_goal_key()
        new_goal_key = None if mode is None or target_id is None else (mode, target_id)
        target_changed = (
            old_goal_key != new_goal_key
            or self.target_pos != target_pos
            or self.target_bounds != target_bounds
        )
        self.mode = mode
        self.target_id = target_id
        self.target_pos = target_pos
        self.target_bounds = target_bounds
        if target_changed:
            self._clear_global_path()
            self._clear_waypoint()
            self.waypoint_no_improve_steps = 0
            self.waypoint_stuck_replans = 0

    def _clear_goal(self):
        """Clear locked goal fields."""
        self._lock_goal(None, None, None, None)

    def _get_goal_key(self, mode=None, target_id=None):
        """Build a stable key for tracking the current locked goal."""
        mode = self.mode if mode is None else mode
        target_id = self.target_id if target_id is None else target_id
        if mode is None or target_id is None:
            return None
        return (mode, target_id)

    def _clear_waypoint(self):
        """Clear active waypoint state."""
        self.waypoint_pos = None
        self.waypoint_goal_key = None
        self.waypoint_bfs_dist = None
        self.waypoint_missing_steps = 0
        self.local_waypoint_distances = None
        self.local_waypoint_distances_key = None

    def _get_waypoint_local_dist(self, waypoint_pos, local_distances):
        """Get current local BFS distance to a global waypoint position."""
        cell = self._global_pos_to_local_cell(waypoint_pos)
        if cell is None:
            return float("inf")
        row, col = cell
        return float(local_distances[row, col])

    def _set_waypoint(self, goal_key, waypoint_pos, waypoint_dist):
        """Set the active waypoint and reset waypoint-distance cache."""
        self.waypoint_pos = waypoint_pos
        self.waypoint_goal_key = goal_key
        self.waypoint_bfs_dist = float(waypoint_dist)
        self.local_waypoint_distances = None
        self.local_waypoint_distances_key = None

    def _get_local_target_waypoint(self, local_distances):
        """Return the locally reachable target center, if visible."""
        if self.target_pos is None:
            return None, float("inf")

        target_cell = self._global_pos_to_local_cell(self.target_pos)
        if target_cell is None:
            return None, float("inf")

        row, col = target_cell
        bfs_dist = float(local_distances[row, col])
        if not np.isfinite(bfs_dist):
            return None, float("inf")

        return self._to_grid_pos(self.target_pos), bfs_dist

    def _get_path_waypoint(self, local_distances):
        """Select the farthest locally reachable path node within the lookahead radius."""
        if len(self.global_path) == 0:
            return None, float("inf")

        best_pos = None
        best_dist = float("inf")
        start_idx = self._get_current_path_index()
        for idx in range(start_idx, len(self.global_path)):
            path_pos = self.global_path[idx]
            cell = self._global_pos_to_local_cell(path_pos)
            if cell is None:
                if best_pos is not None:
                    break
                continue

            row, col = cell
            bfs_dist = float(local_distances[row, col])
            if not np.isfinite(bfs_dist):
                continue
            if bfs_dist <= 0.0 or bfs_dist > Config.WAYPOINT_LOOKAHEAD_RADIUS:
                continue

            best_pos = path_pos
            best_dist = bfs_dist

        return best_pos, best_dist

    def _select_waypoint(self, goal_key, local_distances, force_global_replan=False):
        """Select a stable local waypoint from the current global plan."""
        if goal_key is None or (self.target_bounds is None and self.target_pos is None):
            self._clear_waypoint()
            return False

        target_waypoint, target_dist = self._get_local_target_waypoint(local_distances)
        if target_waypoint is not None:
            self._set_waypoint(goal_key, target_waypoint, target_dist)
            return True

        if not self._ensure_global_path(goal_key, force=force_global_replan):
            self._clear_waypoint()
            return False

        best_pos, best_dist = self._get_path_waypoint(local_distances)
        if best_pos is None and not force_global_replan and self._ensure_global_path(goal_key, force=True):
            best_pos, best_dist = self._get_path_waypoint(local_distances)

        if best_pos is None:
            self._clear_waypoint()
            return False

        self._set_waypoint(goal_key, best_pos, best_dist)
        return True

    def _update_waypoint_state(self):
        """Keep or reselect a local reachable waypoint for the current goal."""
        self.waypoint_reached = False
        self.completed_waypoint_pos = None
        self.completed_waypoint_goal_key = None
        self.completed_waypoint_cur_dist = None

        goal_key = self._get_goal_key()
        if goal_key is None or (self.target_bounds is None and self.target_pos is None):
            self._clear_waypoint()
            self.waypoint_no_improve_steps = 0
            self.waypoint_stuck_replans = 0
            return

        local_distances = self._get_local_bfs_distances()
        if not np.isfinite(local_distances).any():
            self._clear_waypoint()
            self.waypoint_no_improve_steps = 0
            self.waypoint_stuck_replans = 0
            return

        old_dist = self._get_waypoint_local_dist(self.waypoint_pos, local_distances)
        old_waypoint_valid = self.waypoint_goal_key == goal_key and np.isfinite(old_dist)
        waypoint_missing = (
            self.waypoint_pos is not None
            and self.waypoint_goal_key == goal_key
            and not old_waypoint_valid
        )
        if waypoint_missing:
            self.waypoint_missing_steps += 1
        else:
            self.waypoint_missing_steps = 0

        if old_waypoint_valid and old_dist <= 0.0:
            self.waypoint_reached = True
            self.completed_waypoint_pos = self.waypoint_pos
            self.completed_waypoint_goal_key = self.waypoint_goal_key
            self.completed_waypoint_cur_dist = old_dist
            self.waypoint_no_improve_steps = 0
            self.waypoint_stuck_replans = 0
            self.waypoint_missing_steps = 0
            self._select_waypoint(goal_key, local_distances)
            return

        force_global_replan = self._should_force_global_replan(old_waypoint_valid)

        if old_waypoint_valid and not force_global_replan and self.waypoint_no_improve_steps < Config.WAYPOINT_REPLAN_STUCK_STEPS:
            self.waypoint_bfs_dist = old_dist
            return

        self.waypoint_no_improve_steps = 0
        if force_global_replan:
            self.waypoint_stuck_replans = 0
            self.waypoint_missing_steps = 0
        elif old_waypoint_valid:
            self.waypoint_stuck_replans += 1
        self._select_waypoint(goal_key, local_distances, force_global_replan=force_global_replan)

    def _get_nearest_target_station(self, only_safe=False):
        """Select nearest target station from current carried packages.

        从当前携带包裹对应的驿站中选择最近目标。
        """
        target_ids = set(self.packages)
        target_stations = [station for station in self.stations if station.get("config_id", 0) in target_ids]
        if only_safe:
            target_stations = [station for station in target_stations if self._is_station_safe(station)]
        if len(target_stations) == 0:
            return None

        return min(
            target_stations,
            key=lambda s: self._get_region_distance(self.cur_pos, self._get_station_bounds(s)),
        )

    def _get_station_goal(self, only_safe=False):
        """Build a station goal tuple if any matching target station exists."""
        station = self._get_nearest_target_station(only_safe=only_safe)
        if station is None:
            return None

        return (
            MODE_STATION,
            station.get("config_id", 0),
            self._get_organ_pos(station),
            self._get_station_bounds(station),
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
        if charger is not None and charger_pos is not None and charger_dist <= warehouse_dist:
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

    def _get_charge_exit_goal(self):
        """Return the next task goal once charging is no longer necessary."""
        if self.package_count == 0:
            warehouse, warehouse_pos, _ = self._get_nearest_warehouse()
            warehouse_bounds = self._get_warehouse_bounds(warehouse)
            if warehouse_pos is None:
                return None
            if not self._can_reach_target(target_pos=warehouse_pos, target_bounds=warehouse_bounds):
                return None
            return (MODE_WAREHOUSE_REFILL, "warehouse", warehouse_pos, warehouse_bounds)

        return self._get_station_goal(only_safe=True)

    def _should_exit_charge_mode(self):
        """Exit charging as soon as battery is enough for the next task stage."""
        if self.mode not in (MODE_CHARGER, MODE_WAREHOUSE_CHARGE):
            return False
        return self._get_charge_exit_goal() is not None

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
            return self._get_charger_by_id(self.target_id) is not None and not self._should_exit_charge_mode()

        if self.mode == MODE_WAREHOUSE_CHARGE:
            return self._get_nearest_warehouse()[1] is not None and not self._should_exit_charge_mode()

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
        charge_exit_goal = self._get_charge_exit_goal() if self.mode in (MODE_CHARGER, MODE_WAREHOUSE_CHARGE) else None
        if charge_exit_goal is not None:
            self._lock_goal(*charge_exit_goal)
            return

        if self.package_count == 0:
            warehouse, warehouse_pos, _ = self._get_nearest_warehouse()
            warehouse_bounds = self._get_warehouse_bounds(warehouse)
            if warehouse_pos is not None and self._can_reach_target(target_pos=warehouse_pos, target_bounds=warehouse_bounds):
                self._lock_goal(MODE_WAREHOUSE_REFILL, "warehouse", warehouse_pos, warehouse_bounds)
                return

            charger, charger_pos, _ = self._get_nearest_charger()
            if charger is not None and charger_pos is not None:
                self._lock_goal(
                    MODE_CHARGER,
                    charger.get("config_id", 0),
                    charger_pos,
                    self._get_charger_bounds(charger),
                )
                return

            self._lock_goal(MODE_WAREHOUSE_REFILL, "warehouse", warehouse_pos, warehouse_bounds)
            return

        safe_station_goal = self._get_station_goal(only_safe=True)
        if safe_station_goal is not None:
            self._lock_goal(*safe_station_goal)
            return

        mode, target_id, target_pos, _ = self._get_nearest_charge_target()
        if target_pos is not None:
            if mode == MODE_CHARGER:
                target_bounds = self._get_charger_bounds(self._get_charger_by_id(target_id))
            else:
                warehouse, _, _ = self._get_nearest_warehouse()
                target_bounds = self._get_warehouse_bounds(warehouse)
            self._lock_goal(mode, target_id, target_pos, target_bounds)
            return

        station_goal = self._get_station_goal(only_safe=False)
        if station_goal is not None:
            self._lock_goal(*station_goal)
            return

        self._clear_goal()

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
        self.prev_waypoint_pos = self.waypoint_pos
        self.prev_waypoint_goal_key = self.waypoint_goal_key
        self.prev_waypoint_bfs_dist = self.waypoint_bfs_dist

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
        prev_delivered = self.prev_delivered if self.prev_delivered is not None else self.delivered
        newly_delivered = max(0, self.delivered - prev_delivered)

        # 1. Delivery reward / 投递主奖励
        reward += Config.DELIVERY_REWARD_SCALE * newly_delivered

        # 2. Waypoint BFS progress shaping / waypoint BFS 进度奖励
        waypoint_delta = None
        if (
            self.waypoint_reached
            and self.completed_waypoint_goal_key == self.prev_waypoint_goal_key
            and self.completed_waypoint_pos == self.prev_waypoint_pos
            and self.prev_waypoint_bfs_dist is not None
            and self.completed_waypoint_cur_dist is not None
        ):
            waypoint_delta = float(self.prev_waypoint_bfs_dist) - float(self.completed_waypoint_cur_dist)
        elif (
            self.waypoint_goal_key == self.prev_waypoint_goal_key
            and self.waypoint_pos == self.prev_waypoint_pos
            and self.prev_waypoint_bfs_dist is not None
            and self.waypoint_bfs_dist is not None
        ):
            waypoint_delta = float(self.prev_waypoint_bfs_dist) - float(self.waypoint_bfs_dist)

        if waypoint_delta is not None and np.isfinite(waypoint_delta):
            reward += self._get_progress_coef(self.mode) * np.clip(waypoint_delta, -1.0, 1.0)
            if waypoint_delta > Config.WAYPOINT_PROGRESS_TOL:
                self.waypoint_no_improve_steps = 0
                self.waypoint_stuck_replans = 0
            else:
                self.waypoint_no_improve_steps += 1

        if self.waypoint_reached:
            reward += Config.WAYPOINT_REACHED_REWARD

        # 3. Visitation penalty / 全局访问惩罚
        reward -= Config.VISIT_PENALTY_SCALE * self._get_visit_cost(self.cur_pos)

        # 4. Charge event / 充电事件奖励
        if charge_event and self.prev_mode == MODE_CHARGER:
            reward += Config.CHARGE_EVENT_REWARD

        # 5. Warehouse event / 仓库事件奖励
        if warehouse_event:
            if self.prev_mode == MODE_WAREHOUSE_REFILL and self.prev_package_count == 0 and self.package_count > 0:
                reward += Config.WAREHOUSE_REFILL_REWARD
            elif self.prev_mode == MODE_WAREHOUSE_CHARGE:
                reward += Config.WAREHOUSE_CHARGE_REWARD

        # 6. Smooth NPC risk penalty / 平滑 NPC 风险惩罚
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


        # 7. Step penalty / 每步惩罚
        reward -= Config.STEP_PENALTY

        # 8. Terminal penalty / 终局惩罚
        if self.terminated:
            if self.battery <= 0:
                reward += Config.TERMINAL_ENERGY_DEPLETED_PENALTY
            else:
                reward += Config.TERMINAL_COLLISION_PENALTY

        return [reward]
