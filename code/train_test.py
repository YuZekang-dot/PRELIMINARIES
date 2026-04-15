#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.utils.train_test_utils import run_train_test

# To run the train_test, you must modify the algorithm name here. It must be one of algorithm_name_list.
# Simply modify the value of the algorithm_name variable.
# 运行train_test前必须修改这里的算法名字, 必须是 algorithm_name_list 里的一个, 修改algorithm_name的值即可
algorithm_name_list = ["ppo", "diy"]
algorithm_name = "ppo"


if __name__ == "__main__":
    # run_train_test 用于执行一次“最小闭环”的训练自检：
    # 1. 按 algorithm_name 加载对应智能体、模型、算法与工作流配置；
    # 2. 启动环境交互，采集少量样本；
    # 3. 触发至少一次训练更新；
    # 4. 检查代码包中的核心模块是否能正常联通运行。
    #
    # 这个脚本的目标不是正式训练出高质量模型，而是尽快验证：
    # - 配置文件是否正确；
    # - 特征处理、动作处理、样本处理是否可用；
    # - 模型前向、损失计算、反向传播是否能正常执行；
    # - 模型保存流程是否能够被触发。
    #
    # env_vars 中传入的是对训练配置项的临时覆盖值，
    # 这样可以避免直接修改 configure_app.toml，
    # 同时把训练门槛压到很低，让 train_test 尽快完成。
    run_train_test(
        # 指定本次自检使用的算法名称。
        # 这里会根据值去加载对应的 agent / workflow / model / algorithm 配置。
        algorithm_name=algorithm_name,

        # 可选算法名单，用于校验 algorithm_name 是否合法。
        # algorithm_name 必须是该列表中的一个，否则 train_test 无法正确定位算法配置。
        algorithm_name_list=algorithm_name_list,

        # 下面这些键值会临时覆盖训练相关配置，目的是让测试更快触发训练流程。
        env_vars={
            # 样本池容量设为 10000：
            # 当前采样慢于训练，容量保持中等，避免过多旧样本滞留。
            "replay_buffer_capacity": "10000",

            # preload_ratio 设为 0.6：
            # 样本池达到 60% 后启动训练，减少满池等待，同时保留足够 batch 多样性。
            "preload_ratio": "0.6",

            # 训练快于样本产出时限制重复消费，防止 ratio 继续升到 7+。
            "reverb_rate_limiter": "SampleToInsertRatio",
            "reverb_samples_per_insert": "2",
            "reverb_error_buffer": "2048",

            # 训练 batch size 设为 1024：
            # 降低 data_fetch 压力，让新 reward 下的策略更新更及时。
            "train_batch_size": "1024",

            # dump_model_freq 设为 100：
            # 维持当前模型保存频率，避免在采样瓶颈下增加额外 I/O。
            "dump_model_freq": "100",
        },
    )
