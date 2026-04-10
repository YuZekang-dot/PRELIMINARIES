#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery policy network.
智运无人机策略网络。
"""


import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc_layer(in_features: int, out_features: int):
    """Create and initialize a linear layer with orthogonal init.

    创建并初始化线性层（正交初始化）。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight)
    nn.init.zeros_(fc.bias)
    return fc


class MLP(nn.Module):
    """Multi-layer perceptron.

    多层感知器。
    """

    def __init__(
        self,
        fc_feat_dim_list: list,
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module(f"{name}_fc{i + 1}", fc)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_relu{i + 1}", non_linearity())

    def forward(self, x):
        return self.fc_layers(x)


class Model(nn.Module):
    """Actor-Critic model for Drone Delivery.

    智运无人机 Actor-Critic 模型。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "drone_delivery"
        self.device = device

        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM
        hidden_dim = 64
        fusion_dim = 128

        self.vector_feature_len = Config.VECTOR_FEATURE_LEN
        self.local_map_channels = Config.LOCAL_MAP_CHANNELS
        self.local_map_side = Config.LOCAL_MAP_SIDE

        # Vector branch / 标量特征分支
        self.vector_encoder = MLP(
            [self.vector_feature_len, hidden_dim, hidden_dim],
            "vector",
            non_linearity_last=True,
        )

        # Spatial branch / 局部地图分支
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(self.local_map_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        for module in self.spatial_encoder:
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)

        self.spatial_proj = MLP(
            [32 * 5 * 5, hidden_dim],
            "spatial",
            non_linearity_last=True,
        )

        # Fusion backbone / 融合主干
        self.backbone = MLP(
            [hidden_dim * 2, fusion_dim, hidden_dim],
            "backbone",
            non_linearity_last=True,
        )

        # Actor head (direct projection, no hidden layer) / Actor 输出头（直接投影，无隐藏层）
        self.actor_head = make_fc_layer(hidden_dim, action_num)

        # Critic head (direct projection, small init) / Critic 输出头（直接投影，小初始化）
        self.critic_head = make_fc_layer(hidden_dim, value_num)
        nn.init.orthogonal_(self.critic_head.weight, gain=0.01)

    def forward(self, s, inference=False):
        """Forward pass.

        前向传播。
        """
        feat = s.to(torch.float32)
        vector_feat = feat[:, : self.vector_feature_len]
        local_map_feat = feat[:, self.vector_feature_len :]
        local_map_feat = local_map_feat.view(
            -1,
            self.local_map_channels,
            self.local_map_side,
            self.local_map_side,
        )

        vector_hidden = self.vector_encoder(vector_feat)
        spatial_hidden = self.spatial_encoder(local_map_feat)
        spatial_hidden = spatial_hidden.reshape(spatial_hidden.shape[0], -1)
        spatial_hidden = self.spatial_proj(spatial_hidden)

        hidden = self.backbone(torch.cat([vector_hidden, spatial_hidden], dim=1))
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
