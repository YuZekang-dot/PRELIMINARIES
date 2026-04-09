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
            # 样本池容量设为 10：
            # 正式训练时这个值通常较大；这里故意设得很小，
            # 是为了让少量交互样本就能把样本池填到可训练状态。
            "replay_buffer_capacity": "10",

            # preload_ratio 设为 0.2：
            # 表示样本池中只要有 20% 的数据就允许开始训练。
            # 结合 replay_buffer_capacity = 10，相当于样本数达到约 2 条时即可触发训练，
            # 能显著缩短自检等待时间。
            "preload_ratio": "0.2",

            # 训练 batch size 设为 2：
            # Learner 每次只取极小批量样本做一次更新，
            # 这样在样本很少的情况下也能完成一次反向传播，
            # 适合快速验证训练链路是否畅通。
            "train_batch_size": "2",

            # dump_model_freq 设为 1：
            # 表示训练步每推进 1 次就触发一次模型保存检查。
            # 这样可以在 train_test 阶段顺便验证 save_model 逻辑是否正常。
            "dump_model_freq": "1",
        },
    )
