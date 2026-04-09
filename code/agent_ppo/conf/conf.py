#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery PPO configuration.
智运无人机 PPO 配置。
"""


class Config:

    # Feature dimensions / 特征维度
    HERO_STATE_DIM = 4
    STATION_FEAT_DIM = 7
    STATION_TOPK = 3
    STATION_DIM = STATION_FEAT_DIM * STATION_TOPK
    CHARGER_DIM = 7
    NPC_DIM = 7
    LEGAL_ACT_DIM = 8
    INDICATOR_DIM = 3

    FEATURES = [
        HERO_STATE_DIM,
        STATION_DIM,
        CHARGER_DIM,
        NPC_DIM,
        LEGAL_ACT_DIM,
        INDICATOR_DIM,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间
    ACTION_NUM = 8
    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    # Value head (single) / 价值头（单头）
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    # Moderate entropy coeff / 中等熵系数
    BETA_START = 0.005
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
    GRAD_CLIP_RANGE = 0.5
    USE_GRAD_CLIP = True

    # Goal-switching and reward shaping / 目标切换与奖励塑形
    SAFETY_MARGIN = 8.0

    DELIVERY_REWARD_SCALE = 2.5
    STEP_PENALTY = 0.002
    BLOCK_PENALTY = 0.01

    CHARGE_EVENT_REWARD = 0.20
    WAREHOUSE_REFILL_REWARD = 0.30
    WAREHOUSE_CHARGE_REWARD = 0.15

    PROGRESS_REWARD_STATION = 0.05
    PROGRESS_REWARD_CHARGER = 0.04
    PROGRESS_REWARD_WAREHOUSE_REFILL = 0.05
    PROGRESS_REWARD_WAREHOUSE_CHARGE = 0.03

    NPC_PENALTY_SCALE = 0.20
    NPC_PENALTY_RADIUS = 8
    NPC_ESCAPE_TRIGGER_DIST = 4
    NPC_ESCAPE_REWARD_SCALE = 0.02

    TERMINAL_COLLISION_PENALTY = -2.0
    TERMINAL_ENERGY_DEPLETED_PENALTY = -1.2

    NUMB_HEAD = 1
