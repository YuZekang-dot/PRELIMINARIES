# Preliminaries

该仓库用于腾讯开悟 `drone_delivery` 赛题的 PPO 方案开发与训练。

- 赛题与环境说明见 [赛题文档](赛题.md)
- 当前特征工程、模型结构和奖励设计见 [README_feature_reward.md](README_feature_reward.md)

主要实现位于 `code/agent_ppo/`。训练运行过程中生成的日志和模型备份位于 `train/` 下，已通过 `.gitignore` 排除，不会提交到 GitHub。
