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
    dist = np.sqrt(relative_pos[0] ** 2 + relative_pos[1] ** 2)
    abs_norm = norm(np.array(target_pos), 128, -128)
    return np.array(
        [
            float(found),
            norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
            norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
            abs_norm[0],
            abs_norm[1],
            norm(dist, 1.41 * 128),
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
        self.npcs = []
        self.map_info = None
        self.env_legal_act = [1] * Config.ACTION_NUM

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

        self.last_delivered = self.delivered
        self.delivered = hero.get("delivered", 0)
        self.step_no = obs.get("step_no", 0)

        self.stations = []
        self.chargers = []
        for organ in frame_state.get("organs", []):
            st = organ.get("sub_type", 0)
            if st == 3:
                self.stations.append(organ)
            elif st == 2:
                self.chargers.append(organ)

        self.npcs = frame_state.get("npcs", [])

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
            dist = np.sqrt((s["pos"]["x"] - self.cur_pos[0]) ** 2 + (s["pos"]["z"] - self.cur_pos[1]) ** 2)
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
            key=lambda c: np.sqrt((c["pos"]["x"] - self.cur_pos[0]) ** 2 + (c["pos"]["z"] - self.cur_pos[1]) ** 2),
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
            key=lambda n: np.sqrt((n["pos"]["x"] - self.cur_pos[0]) ** 2 + (n["pos"]["z"] - self.cur_pos[1]) ** 2),
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

        return feature, legal_action, reward

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
        reward = 0.0

        # 1. Delivery reward / 投递奖励
        newly_delivered = max(0, self.delivered - self.last_delivered)
        if newly_delivered > 0:
            reward += 1.0 * newly_delivered

        # 2. Step penalty / 步数惩罚
        reward -= 0.001

        return [reward]
