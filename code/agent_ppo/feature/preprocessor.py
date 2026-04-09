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
    abs_norm = norm(np.array(target_pos), 128, -128)
    return np.array(
        [
            float(found),
            norm(relative_pos[0] / max(euclid_dist, 1e-4), 1, -1),
            norm(relative_pos[1] / max(euclid_dist, 1e-4), 1, -1),
            abs_norm[0],
            abs_norm[1],
            norm(chebyshev_dist, 128),
            float(extra_flag),
        ]
    )


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

        # Previous snapshot / 上一时刻快照
        self.prev_pos = None
        self.prev_delivered = None
        self.prev_package_count = None
        self.prev_charger_count = None
        self.prev_warehouse_count = None
        self.prev_mode = None
        self.prev_target_id = None
        self.prev_target_pos = None
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
        """Core feature extraction. Returns (feature_50d, legal_action, reward).

        核心特征提取方法，返回 50 维特征向量、合法动作掩码和奖励。
        """
        self._parse_obs(env_obs)
        self._update_goal_state()

        # 1. Hero state features (4D) / 英雄状态特征（4D）
        battery_ratio = norm(self.battery, self.battery_max)
        package_count_norm = norm(len(self.packages), 3)
        cur_pos_norm = norm(np.array(self.cur_pos, dtype=float), 128, -128)
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
            dist = self._chebyshev_distance(self.cur_pos, (s["pos"]["x"], s["pos"]["z"]))
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
            key=lambda c: self._chebyshev_distance(self.cur_pos, (c["pos"]["x"], c["pos"]["z"])),
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

        # Concatenate features (Total 50D / 合计 50D)
        feature = np.concatenate(
            [
                hero_feat,
                station_feat,
                charger_feat,
                npc_feat,
                np.array(legal_action, dtype=float),
                indicators,
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

    def _get_organ_pos(self, organ):
        """Extract organ position.

        获取物件位置。
        """
        # Assume organ["pos"] is already the semantic center point.
        # 假设 organ["pos"] 本身就是语义中心点，不再使用 w/h 做额外偏移。
        return (float(organ["pos"]["x"]), float(organ["pos"]["z"]))

    def _chebyshev_distance(self, pos_a, pos_b):
        """Chebyshev distance under 8-neighbor movement.

        8 邻接移动下的切比雪夫距离。
        """
        if pos_a is None or pos_b is None:
            return float("inf")
        return max(abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1]))

    def _can_reach_target(self, target_pos):
        """Check whether current battery can reach the target position.

        判断当前电量是否足以到达目标位置。
        """
        if target_pos is None:
            return False
        return self.battery >= self._chebyshev_distance(self.cur_pos, target_pos)

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
            key=lambda c: self._chebyshev_distance(from_pos, self._get_organ_pos(c)),
        )
        charger_pos = self._get_organ_pos(charger)
        return charger, charger_pos, self._chebyshev_distance(from_pos, charger_pos)

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
            key=lambda w: self._chebyshev_distance(from_pos, self._get_organ_pos(w)),
        )
        warehouse_pos = self._get_organ_pos(warehouse)
        return warehouse, warehouse_pos, self._chebyshev_distance(from_pos, warehouse_pos)

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
            key=lambda s: self._chebyshev_distance(self.cur_pos, self._get_organ_pos(s)),
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
        d_to_station = self._chebyshev_distance(self.cur_pos, station_pos)
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
            _, warehouse_pos, _ = self._get_nearest_warehouse()
            return self.package_count == 0 and warehouse_pos is not None and self._can_reach_target(warehouse_pos)

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
        elif self.mode == MODE_CHARGER:
            charger = self._get_charger_by_id(self.target_id)
            self.target_pos = self._get_organ_pos(charger) if charger is not None else None
        elif self.mode in (MODE_WAREHOUSE_REFILL, MODE_WAREHOUSE_CHARGE):
            _, warehouse_pos, _ = self._get_nearest_warehouse()
            self.target_pos = warehouse_pos
        else:
            self.target_pos = None

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
            _, warehouse_pos, _ = self._get_nearest_warehouse()
            if warehouse_pos is not None and self._can_reach_target(warehouse_pos):
                self.mode = MODE_WAREHOUSE_REFILL
                self.target_id = "warehouse"
                self.target_pos = warehouse_pos
                return

            charger, charger_pos, _ = self._get_nearest_charger()
            if charger_pos is not None:
                self.mode = MODE_CHARGER
                self.target_id = charger.get("config_id", 0)
                self.target_pos = charger_pos
                return

            self.mode = MODE_WAREHOUSE_REFILL
            self.target_id = "warehouse"
            self.target_pos = warehouse_pos
            return

        station = self._get_nearest_target_station()
        if station is not None and self._is_station_safe(station):
            self.mode = MODE_STATION
            self.target_id = station.get("config_id", 0)
            self.target_pos = self._get_organ_pos(station)
            return

        mode, target_id, target_pos, _ = self._get_nearest_charge_target()
        if target_pos is not None:
            self.mode = mode
            self.target_id = target_id
            self.target_pos = target_pos
            return

        if station is not None:
            self.mode = MODE_STATION
            self.target_id = station.get("config_id", 0)
            self.target_pos = self._get_organ_pos(station)
            return

        self.mode = None
        self.target_id = None
        self.target_pos = None

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
        if self.mode == self.prev_mode and self.target_id == self.prev_target_id and self.target_pos is not None:
            prev_dist = self._chebyshev_distance(self.prev_pos, self.target_pos)
            cur_dist = self._chebyshev_distance(self.cur_pos, self.target_pos)
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
