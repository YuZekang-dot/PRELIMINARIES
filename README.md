# 智运无人机：路径规划 - 物流配送

本仓库是腾讯开悟 `drone_delivery` 赛题代码包，目标是在规定时间内、能量耗尽前控制小悟无人机从仓库取货并配送到各个企鹅驿站，尽可能多地完成包裹配送。

本 README 结合 `区域初赛.docx` 与当前仓库代码整理而成，保留 Word 文档中的有效信息，并合并删除重复段落。

## 仓库概览

| 路径 | 说明 |
| --- | --- |
| `code/kaiwu.json` | 项目版本与项目码，当前 `project_code = drone_delivery` |
| `code/train_test.py` | 代码正确性测试入口，会使用当前代码包完成一步训练；默认 `algorithm_name = "ppo"`，可改为 `"diy"` |
| `code/conf/app_conf_drone_delivery.toml` | 项目策略配置，当前默认训练算法为 `ppo` |
| `code/conf/algo_conf_drone_delivery.toml` | `ppo` 与 `diy` 两套智能体、训练工作流、评估工作流配置 |
| `code/conf/configure_app.toml` | Learner、样本池、模型保存和同步等训练任务配置 |
| `code/agent_ppo/` | 当前可运行的 PPO 基线智能体实现 |
| `code/agent_diy/` | 用户自定义算法模板 |
| `dev/.docker-compose.yaml` | 本地开发容器配置，包含 `prometheus-pushgateway`、`gamecore`、`kaiwu_env`、`kaiwudrl` 服务 |
| `dev/.env` | 本地开发环境变量，默认 IDE 端口 `10000`、CPU 类型 `x86`、项目版本 `10.0.1` |
| `license.dat` | 本地开发容器挂载使用的 license 文件 |

当前 `code/conf/app_conf_drone_delivery.toml` 中的默认策略配置：

```toml
[drone_delivery]
rl_helper = "kaiwudrl.server.aisrv.kaiwu_rl_helper_standard.KaiWuRLStandardHelper"

[drone_delivery.policies.train_one]
policy_builder = "kaiwudrl.server.aisrv.async_policy.AsyncBuilder"
algo = "ppo"
```

在开悟/KaiwuDRL 环境内可通过 `code/train_test.py` 做代码正确性测试。该脚本将 `replay_buffer_capacity`、`preload_ratio`、`train_batch_size`、`dump_model_freq` 等参数临时设为较小值，用于快速完成一步训练检查。

如使用本地 Docker 开发配置，可在 `dev` 目录按 `.docker-compose.yaml` 启动容器；默认 IDE 地址由 `KAIWU_IDE_PORT=10000` 生成，映射到 `http://127.0.0.1:10000/?folder=/data/projects/drone_delivery`。

当前仓库实现状态：

| 模块 | 当前状态 |
| --- | --- |
| `agent_ppo` | 已实现完整 PPO 基线，可作为当前默认训练入口 |
| `agent_diy` | 仍为模板代码，`agent.py`、`algorithm.py`、`feature/definition.py`、`workflow/train_workflow.py` 中保留了待开发的 `pass` 或占位逻辑，不适合作为现成可运行基线直接切换使用 |

## 任务目标

小悟无人机在规定时间内和能量耗尽前，从仓库取货并配送到各个企鹅驿站，尽可能多地配送包裹。无人机初始 300 电量（可配置），每走一步消耗 1 格电量。

地图中心存在仓库，无人机进入仓库区域后，会自动补满包裹（3 个）并充满电量。

地图中存在充电桩，无人机进入充电桩范围后会充满电量。

地图中存在官方无人机（巡逻 NPC），小悟无人机需避免与其碰撞（距离 ≤ 1 格即为碰撞），碰撞即任务终止。

## 环境介绍

### 地图

地图大小为 128×128（栅格化地图），左上角为地图原点 `(0,0)`，x 轴向右为正，z 轴向下为正。为考察模型泛化能力，本赛题共提供十五张地图：十张开放给选手训练、评估，五张作为隐藏地图用于最终测评。

地图中包含小悟无人机、官方无人机、仓库、企鹅驿站、充电桩、道路、障碍物。

### 元素介绍

| 元素 | 说明 |
| --- | --- |
| 小悟无人机 | 智能体可以控制小悟无人机在地图中进行移动。每走一步消耗 1 格电量，可携带最多 3 个包裹进行配送。无人机抵达充电桩范围内，立即充满电量（默认 300 电量，可配置）。小悟机器人需避免撞到官方机器人（距离小于等于 1 格即为碰到，即走到以官方机器人为中心 3x3 的范围内），任务终止。 |
| 官方无人机 | 任务开始时，生成 4 个（可配置）官方无人机。官方无人机会在出生点附近 21×21 格（半径 10）范围内随机巡逻。小悟无人机与其距离 ≤ 1 格即判定为碰撞，任务终止。 |
| 仓库 | 位于地图中心区域。分为装货区与非装货区，当进入到装货区，无人机会把携带包裹补足到 3 个，并把能量充满。 |
| 企鹅驿站 | 任务开始时，根据配置生成对应数量的企鹅驿站（可配置，默认 10），每个驿站占 3×3 格。无人机进入驿站范围时，若携带包裹中存在对应驿站编号的包裹，则投递成功并得分。 |
| 充电桩 | 4 个（可配置），每个充电桩占 3×3 格。机器人抵达充电桩范围内，立即充满能量（300 电量，可配置）。 |
| 道路 | 智能体可以正常移动的区域。智能体有 8 个移动方向，每步移动 1 格。仓库、道路、草地都为可通行区域，即本环境中的“道路”。 |
| 障碍物 | 障碍物会阻碍智能体的移动。当智能体向有障碍物的方向移动时，将会停留在原地但仍消耗能量。仅楼房为障碍物。 |

### 小悟无人机

| 属性 | 说明 |
| --- | --- |
| 数量 | 1 |
| 默认动作空间 | 8（八个方向移动） |
| 视野域 | 以智能体为中心，分别向上、下、左、右四个方向拓宽 10 格的正方形区域，即 21×21 |
| 移动速度 | 1 格/步 |
| 初始电量 | 300（可配置） |
| 电量消耗 | 1/步 |
| 包裹容量 | 最多携带 3 个包裹 |

### 计分规则

任务得分 = 配送包裹数 × 100

每成功投递 1 个包裹到对应的企鹅驿站，得分 +100。

### 终止条件

| 终止原因 | 任务状态 | 说明 |
| --- | --- | --- |
| 达到最大步数 | completed | `current_step >= max_step` |
| 电量耗尽 | failed | `energy <= 0` |
| 碰撞官方无人机 | failed | 小悟无人机与官方无人机距离 ≤ 1 格（即进入官方无人机 3×3 范围） |

## 环境详述

在开始开发前，请仔细阅读腾讯开悟强化学习开发框架，深入理解环境、智能体、工作流等核心概念及其相关接口的使用方法。

### 环境配置

在智能体和环境的交互中，首先会调用 `env.reset` 方法，该方法接受一个 `usr_conf` 参数，这个参数通过读取 `train_env_conf.toml` 文件的内容来实现定制化的环境配置。

```python
# usr_conf 为用户传入的环境配置
observation, state = env.reset(usr_conf=usr_conf)
```

`train_env_conf.toml` 中包含以下配置信息：

| 数据名 | 数据类型 | 取值范围 | 默认值 | 数据描述 |
| --- | --- | --- | --- | --- |
| `map` | `list[int]` | `[1-10]` | `[1,2,...,10]` | 训练使用的地图编号列表 |
| `map_random` | `bool` | `true/false` | `false` | 是否随机抽取地图。`true` 表示每局从地图列表中随机抽取一张，`false` 表示按顺序抽取 |
| `drone_count` | `int` | `1-4` | `4` | 官方无人机数量（巡逻 NPC） |
| `station_count` | `int` | `3-10` | `10` | 企鹅驿站数量 |
| `charger_count` | `int` | `1-4` | `4` | 充电桩数量 |
| `battery_max` | `int` | `100-999` | `300` | 最大电量，满电状态下的电量 |
| `max_step` | `int` | `1-2000` | `1000` | 最大步数，单局任务预测步数达到最大步数时任务结束 |

补充说明：

`train_env_conf.toml` 文件中的配置仅在训练时生效，请按上表数据描述进行配置。若配置错误，训练任务会变为“失败”状态，此时可以通过查看 env 模块的错误日志进行排查。

若需调整模型评估任务时的配置，用户需要通过腾讯开悟平台创建评估任务并完成环境配置，详细参数见智能体详情模型评估。

`train_env_conf.toml` 采用的默认配置：

```toml
[env_conf]
# Maps used for training. Customize by keeping only desired map IDs, e.g. [1, 2] for maps 1 and 2.
# 训练使用的地图。可自定义选择期望用来训练的地图，如只期望使用1、2号地图训练数组内仅保留[1,2]即可
map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Whether to randomly select maps. true = randomly pick one from configured maps per episode, false = used sequentially.
# 是否随机抽取地图。布尔值，true表示每局从配置的地图中中随机抽取一张，false表示按顺序抽取地图训练
map_random = false

# Number of official drones. Range: 1~4 (integer).
# In each round, official drones will be randomly generated on the road according to the configured.
# 官方无人机数量。可配置范围为1～4（整数）。每局将按照配置数量在道路上随机生成官方无人机。
drone_count = 4

# Number of chargers. Range: 1~4 (integer). When less than 4, spawn points are randomly chosen.
# 充电桩数量。可配置范围为1～4（整数）。当配置小于4时，将从每张地图可生成充电桩的点位随机选择对应数量的点位生成。
charger_count = 4

# Number of post stations. Range: 3~10 (integer). In each round, post stations will be randomly generated on the road according to the configure.
# 企鹅驿站数量。可配置范围为3～10（整数）。每局将按照配置数量在道路上随机生成企鹅驿站。
station_count = 10

# Maximum steps. The task ends when the predicted steps in a single round reach the maximum. Range: 1~2000.
# 最大步数。单局任务预测步数达到最大步数时，任务结束。可配置范围为1～2000。
max_step = 1000

# Maximum battery. The battery level when fully charged. Range: 100~999.
# 最大电量。满电状态下的电量。可配置范围100～999。
battery_max = 300
```

### 环境信息

视野范围：以智能体为中心，分别向上、下、左、右四个方向拓宽 10 格的正方形区域，即 21×21 的观测范围。充电桩提供全局绝对位置信息。

在调用 `env.reset` 与 `env.step` 接口时，会返回环境当前的状态：

```python
# reset 返回
env_obs = env.reset(usr_conf=usr_conf)

# step 返回
env_reward, env_obs = env.step(hero_actions)
```

`env_reward` 包含以下信息：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `frame_no` | `int` | 当前帧号 |
| `env_id` | `string` | 环境标识 |
| `reward` | `float` | 当前得分 |

`env_obs` 包含以下信息：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `env_id` | `string` | 环境标识 |
| `frame_no` | `int` | 当前帧号 |
| `observation` | `Observation` | 观测信息 |
| `extra_info` | `ExtraInfo` | 额外信息 |
| `terminated` | `bool` | 任务是否终止（碰撞/电量耗尽） |
| `truncated` | `bool` | 任务是否因达到最大步数或异常而截断 |

得分信息：`env_reward` 是在当前状态下执行动作 `action` 所获得的分数。得分用于衡量模型在环境中的表现，也作为衡量强化学习训练产出模型的评价指标，与强化学习里的奖励 `reward` 要区别开。

`observation` 信息：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `step_no` | `int32` | 当前步数 |
| `frame_state` | `FrameState` | 帧状态数据（全局对象信息） |
| `env_info` | `EnvInfo` | 环境统计信息 |
| `map_info` | `MapInfo` | 视野内地图信息（21×21 可通行网格） |
| `legal_act` | `list[int]` | 合法动作列表，长度为 8 |

`extra_info` 信息：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `frame_state` | `FrameState` | 全局帧状态数据 |
| `map_id` | `int32` | 当前地图编号 |
| `result_code` | `int32` | 结果代码（0 = 正常） |
| `result_message` | `string` | 结果消息 |

### 动作空间

智能体具有 8 个移动方向，动作值范围为 0-7：

| 动作值 | 方向 | 向量 `(dx, dy)` | 说明 |
| --- | --- | --- | --- |
| 0 | 右 (→) | `(1, 0)` | 向右移动 1 格 |
| 1 | 右上 (↗) | `(1, -1)` | 向右上移动 1 格 |
| 2 | 上 (↑) | `(0, -1)` | 向上移动 1 格 |
| 3 | 左上 (↖) | `(-1, -1)` | 向左上移动 1 格 |
| 4 | 左 (←) | `(-1, 0)` | 向左移动 1 格 |
| 5 | 左下 (↙) | `(-1, 1)` | 向左下移动 1 格 |
| 6 | 下 (↓) | `(0, 1)` | 向下移动 1 格 |
| 7 | 右下 (↘) | `(1, 1)` | 向右下移动 1 格 |

合法动作：8 维整数数组，当前实现恒为 1，表示 8 个方向均可尝试移动。

执行逻辑：

| 类别 | 规则 |
| --- | --- |
| 移动规则 | 直线移动：目标格子对无人机“可通行”即可移动 |
| 移动规则 | 斜向移动防穿角：目标格子可通行，且相邻两条边（水平/垂直方向）至少有一条可通行，避免卡角穿墙。假设无人机当前在中间，想飞向右上角的格子，虽然右上角的格子是空的（可通行），但需要检查它正上方和正右方的两个格子，因为现实中无人机是有体积的，它无法从两堵墙对角线交汇的那个“零宽度的缝隙”里挤过去。 |
| 移动规则 | 撞墙处理：若目标不可通行，则停留原地，但仍计步并消耗能量 |
| 补给/充电规则 | 仓库：进入仓库区域后，每步都会把携带包裹补足到 3 个，并把电量充满 |
| 补给/充电规则 | 充电桩：进入充电桩范围后，每步都会把电量充满（计入“充电次数”的仅为进入范围的那一步） |

### 环境监控信息

监控面板中包含了 env 模块，表示环境指标数据，详细说明如下。

| 面板中文名称 | 面板英文名称 | 指标名称 | 说明 |
| --- | --- | --- | --- |
| 得分 | score | `total_score` | 任务结束时的总积分（配送得分） |
| 得分 | score | `delivery_score` | 配送得分 |
| 步数 | steps | `max_step` | 任务设置的最大步数 |
| 步数 | steps | `finished_steps` | 任务结束时所用的步数 |
| 充电 | charge | `remaining_charge` | 每局任务结束时剩余电量 |
| 充电 | charge | `charge_count` | 每局充电次数 |
| 充电 | charge | `total_charger` | 任务开始时，设置的充电桩个数 |
| 地图 | map | `total_map` | 地图总数 |
| 地图 | map | `map_random` | 是否随机地图（0/1） |
| 包裹 | package | `package_delivered` | 投递包裹数 |
| 驿站数量 | station | `total_post_station` | 任务设置的驿站数量 |
| 仓库 | warehouse | `back_warehouse_count` | 返回仓库次数 |

## 基线智能体

代码包提供的是一个最简基线版本，仅包含基础导航和投递能力。选手需要根据自己对环境的理解，自行扩展特征工程、奖励设计、模型结构等，不断提升智能体的能力。

当前仓库中默认可运行智能体位于 `code/agent_ppo/`，另有 `code/agent_diy/` 作为自定义算法模板。

### 观测处理

环境返回的 `observation` 信息包含了针对智能体的局部观测信息，可以在 `observation_process` 函数中对这些局部观测信息进行处理。推荐使用预处理器 `preprocessor` 对环境返回的 `observation` 信息进行预处理：

```python
def observation_process(self, env_obs):
    feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
    obs_data = ObsData(
        feature=list(feature),
        legal_action=legal_action,
    )
    remain_info = {"reward": reward}
    return obs_data, remain_info
```

### 特征处理

当前基线版本提供了 22 维的简化特征向量，布局如下：

| 部分 | 维度 | 含义 |
| --- | --- | --- |
| `hero_state` | 4 | 无人机自身状态（电量归一化, 包裹数归一化, `pos_x` 归一化, `pos_z` 归一化） |
| `station` | 7 | 最近 1 个驿站（`found`, `dir_x`, `dir_z`, `abs_x` 归一化, `abs_z` 归一化, `dist` 归一化, `is_target`；目标驿站优先） |
| `legal_action` | 8 | 移动方向合法掩码（8 维，0-7 方向） |
| `indicators` | 3 | 二值指示器（`has_package`: 是否携带包裹, `battery_low`: 电量是否低于 30%, `target_visible`: 目标驿站是否可见） |
| 合计 | 22 | 22 维简化特征向量 |

### 奖励处理

当前基线版本提供了稀疏奖励设计：

| 奖励项 | 含义 | 数值 |
| --- | --- | --- |
| `delivery_reward` | 成功投递包裹的奖励（稀疏） | +1.0/个 |
| `step_penalty` | 步数惩罚（鼓励效率） | -0.001/步 |

最终奖励：`reward = delivery_reward + step_penalty`

时序处理：使用 GAE（`Config.GAMMA`, `Config.LAMDA`）计算 `advantage` 与 `reward_sum`，作为 PPO 的训练目标。

代码包仅提供了最基础的智能体实现，用户可以仔细阅读环境详述和数据协议，根据自己对环境的理解，进行特征工程、奖励开发等工作，不断提升智能体的能力。

### 算法介绍

代码包中提供了基础 PPO 算法实现，同时还提供了一个 `diy` 模板算法文件夹，用户可在该文件夹中自定义算法实现。

| 算法 | 说明 |
| --- | --- |
| PPO | Proximal Policy Optimization，近端策略优化算法，训练稳定性高，收敛速度快 |
| DIY | 用户自定义算法实现模板 |

模型设计（Actor-Critic）：

当前基线版本使用单 MLP 骨干 + 双头结构：

```text
输入 (22D) → MLP骨干 (22→64→64) → Actor头 (64→8)  → 动作logits
                                  → Critic头 (64→1) → 状态价值
```

Actor 和 Critic 头均为单层线性投影（无中间隐藏层），整体参数量较小。选手可以自行设计更复杂的网络结构。

动作输出：

| 项 | 说明 |
| --- | --- |
| 动作空间 | 当前基线版本的动作空间为 8 维离散动作（仅移动方向 0-7） |
| 合法动作掩码 | 使用 mask 把非法选项从概率分布中排除（实现上为对 logits 做大负数惩罚再 softmax） |
| 采样策略 | 训练时随机采样（multinomial），评估时选最大概率（argmax） |

训练流程：

| 环节 | 说明 |
| --- | --- |
| 交互采样 | Agent 与环境交互生成采样（`SampleData`），包含 `obs`, `action`, `prob`, `value`, `reward` 等 |
| 后处理 | 填充 `next_value` 并用 GAE 计算 `advantage` 与 `reward_sum` |
| 更新 | 在采样数据上执行 PPO 更新（策略损失、价值损失、熵正则） |
| 保存/评估 | 定期保存模型并在验证地图上评估性能 |

损失函数：

```text
total_loss = vf_coef × value_loss + policy_loss - beta × entropy_loss
```

| 损失项 | 说明 |
| --- | --- |
| `value_loss` | Clipped 价值函数损失 |
| `policy_loss` | PPO Clipped surrogate 目标 |
| `entropy_loss` | 动作熵正则化（鼓励探索） |

算法上报了 `reward` 等指标，用户可以通过腾讯开悟平台/客户端的监控功能查看。针对当前基线算法的指标说明如下：

| 指标名称 | 说明 |
| --- | --- |
| `total_loss` | 总损失 |
| `policy_loss` | 策略损失 |
| `value_loss` | 价值损失 |
| `entropy_loss` | 熵损失 |
| `reward` | 累计回报 |

选手可以在 `monitor_builder.py` 中自行添加更多监控指标。

### 模型保存限制策略

为了避免用户保存模型的频率过于频繁，开悟平台对模型保存会有安全限制，不同的任务会有不同的限制。

默认提供合理的模型保存代码：每 30 min 保存一个模型。支持用户自行实现模型保存的代码，并且能正常按照用户的代码实现保存模型。

## 模型评估

用户可以在腾讯开悟平台上创建模型评估任务。

### 地图配置

为考察模型的泛化能力，本赛题共提供多张地图：

| 地图类型 | 说明 |
| --- | --- |
| 开放地图 | 开放给选手进行训练和评估 |
| 隐藏地图 | 用于最终测评，选手不可见 |

创建评估任务时，可通过平台界面勾选需要评估的地图。

泛化性建议：

| 建议 | 说明 |
| --- | --- |
| 多图训练 | 训练时建议使用多张地图进行训练，避免模型过拟合到单一地图 |
| 多图评估 | 评估时建议在多张不同地图上测试模型表现，确保策略具备跨地图适应性 |
| 隐藏图泛化 | 最终测评将在隐藏地图上进行，模型需要具备对未见过地图的泛化能力 |

### 评估环境配置

创建评估任务时，还需对该任务的环境进行配置：

```toml
[env_conf]
# 官方无人机数量。可配置范围为1～4（整数）
drone_count: 4

# 企鹅驿站数量。可配置范围为3～10（整数）
post_station: 10

# 充电桩数量。可配置范围为1～4（整数）
charger_count: 4

# 最大步数，单局任务预测步数达到最大步数时，任务结束。可配置范围为1～2000
max_step: 1000

# 最大能量，满能量状态下的能量。可配置范围100～999
battery_max: 300
```

任务状态：

| 状态 | 说明 |
| --- | --- |
| 已完成 | 小悟无人机碰到官方无人机、能量耗尽、或达到任务最大步数时该局任务终止 |
| 异常 | 各种原因导致的异常 |

## 选手开发指引

基线代码包提供了最简单的智能体实现，以下给选手提供部分优化提示，选手们可以自行探索更多优化方向。

### 特征工程

| 方向 | 说明 |
| --- | --- |
| 更多驿站信息 | 当前仅用最近 1 个驿站，可扩展至多个甚至全部 10 个 |
| 充电桩特征 | 添加充电桩位置、距离、优先级等信息 |
| NPC 详细特征 | 当前未使用 NPC 信息，可添加 NPC 的距离、位置、速度、威胁等级 |
| 路径规划 | 评估各方向的安全路线质量 |
| 地图记忆 | 记录已探索区域和障碍物分布 |
| 时序特征 | 利用历史轨迹预测 NPC 运动趋势 |

### 奖励设计

| 方向 | 说明 |
| --- | --- |
| 仓库奖励 | 鼓励无包裹时返回仓库补充 |
| 充电奖励 | 鼓励低电量时及时充电 |
| NPC 惩罚 | 接近官方无人机时给予惩罚 |
| 探索奖励 | 鼓励探索新区域 |
| 阶段奖励 | 根据游戏进程调整奖励权重 |

### 模型结构

| 方向 | 说明 |
| --- | --- |
| 网络加深 | 增加网络容量以学习复杂策略 |
| 注意力机制 | 动态关注重要实体（驿站、NPC 等） |
| 特征分离 | 不同类型特征使用不同编码器 |
| 辅助任务 | 添加预测任务增强特征学习（如 NPC 轨迹预测） |

### 高级技术

| 方向 | 说明 |
| --- | --- |
| RND 探索 | 使用随机网络蒸馏增强探索 |
| 多头价值 | 分解不同目标的价值估计（任务/生存） |
| 独立 Critic | Actor 和 Critic 使用独立网络 |

## 数据协议

为了方便同学们调用原始数据和特征数据，下面提供协议供大家查阅。

### 环境返回数据协议

`observation`（观测信息）：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `step_no` | `int32` | 当前步数 |
| `frame_state` | `FrameState` | 帧状态数据（全局对象信息） |
| `env_info` | `EnvInfo` | 环境统计信息 |
| `map_info` | `MapInfo` | 视野内地图信息（21×21 可通行网格） |
| `legal_act` | `list[int]` | 合法动作列表，长度为 8 |

`FrameState`（帧状态）：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `frame_no` | `int32` | 当前帧号 |
| `heroes` | `HeroState` | 小悟无人机状态 |
| `npcs` | `list[NpcState]` | 官方无人机状态列表 |
| `organs` | `list[OrganState]` | 物件状态列表（仓库/充电桩/驿站） |

`HeroState`（小悟无人机状态）：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `hero_id` | `string` | 实体 ID |
| `pos` | `Position` | 当前位置 `{x, z}` |
| `battery` | `int32` | 当前电量 |
| `battery_max` | `int32` | 电量上限 |
| `packages` | `list[int]` | 当前携带包裹列表（元素为驿站编号） |
| `score` | `int32` | 当前得分 |
| `delivered` | `int32` | 已投递包裹数量 |

`NpcState`（官方无人机状态）：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `npc_id` | `string` | 实体 ID |
| `pos` | `Position` | 当前位置 `{x, z}` |

`OrganState`（物件状态）：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `sub_type` | `int32` | 物件类型：1=仓库，2=充电桩，3=企鹅驿站 |
| `config_id` | `int32` | 配置/编号：仓库与充电桩为索引，驿站为 `station_id(1-10)` |
| `pos` | `Position` | 位置 `{x, z}`（绝对格子坐标） |
| `w` | `int32` | 宽度（格子数，仅仓库 `sub_type=1`） |
| `h` | `int32` | 高度（格子数，仅仓库 `sub_type=1`） |
| `range` | `float` | 充电范围（仅充电桩 `sub_type=2`） |

`EnvInfo`（环境信息）：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `total_score` | `int32` | 总得分 |
| `step_no` | `int32` | 当前步数 |
| `delivered` | `int32` | 已投递包裹数 |
| `battery` | `int32` | 当前电量 |
| `battery_max` | `int32` | 电量上限 |
| `package_count` | `int32` | 当前携带包裹数 |
| `pos` | `Position` | 小悟无人机当前位置 |
| `station_count` | `int32` | 驿站数量 |
| `charger_count` | `int32` | 充电次数（进入充电桩范围计一次） |
| `charger_station_count` | `int32` | 充电桩数量 |
| `warehouse_count` | `int32` | 返回仓库次数（起始算一次） |
| `total_map` | `int32` | 总地图数 |
| `map_random` | `int32` | 是否随机地图（0/1） |
| `max_step` | `int32` | 最大步数 |

`MapInfo`（地图信息）：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `map_info` | `list[list[int]]` | 21×21 可通行网格：1=可通行，0=不可通行 |

`Position`（位置）：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `x` | `int32` | X 坐标（栅格） |
| `z` | `int32` | Z 坐标（栅格） |

`RewardData`（奖励数据）：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `env_id` | `string` | 环境标识（地图编号字符串） |
| `frame_no` | `int32` | 当前帧号/步数 |
| `reward` | `float` | 本步即时奖励（仅投递成功时为正，等于投递包裹数 × `score_per_delivery`） |

`ExtraInfo`（额外信息）：

| 数据名 | 数据类型 | 数据描述 |
| --- | --- | --- |
| `frame_state` | `FrameState` | 帧状态数据 |
| `map_id` | `int32` | 当前地图编号 |
| `result_code` | `int32` | 结果代码（0=正常） |
| `result_message` | `string` | 结果消息 |

### 物件类型协议

| `sub_type` 值 | 物件类型 | 说明 |
| --- | --- | --- |
| 1 | 仓库 | 无人机可在此补充包裹和电量 |
| 2 | 充电桩 | 无人机可在此充电 |
| 3 | 企鹅驿站 | 配送目标，投递对应包裹可得分 |

### 任务状态协议

| 状态值 | 说明 |
| --- | --- |
| `running` | 任务进行中 |
| `completed` | 任务正常完成（达到最大步数） |
| `failed` | 任务失败（电量耗尽/碰撞官方无人机），`fail_reason` 字段说明具体原因 |

### 视野网格值协议

| 值 | 含义 |
| --- | --- |
| 0 | 不可通行（障碍物/地图边界） |
| 1 | 可通行道路 |

## 开发框架参考

完整训练流程包含以下关键环节：

| 环节 | 介绍 |
| --- | --- |
| 智能体-环境循环交互 | 智能体将环境提供的观测和奖励处理为符合预测函数输入要求的数据；<br>调用预测函数，生成动作指令；<br>将智能体输出的动作指令处理为符合环境输入要求的数据；<br>环境执行动作后完成状态转移，并反馈新的观测数据和奖励数据。 |
| 样本处理 | 每个环境有不同的开始与结束逻辑，智能体与环境从开始到结束的完整交互过程，称为 episode；<br>智能体与环境每一次交互产生的结构化数据，称为样本；一个 episode 产生的样本序列称为轨迹；<br>对轨迹数据进行处理，转换为规范化训练样本（`SampleData`）。 |
| 模型迭代优化 | 基于训练样本，通过算法持续更新模型参数，实现策略优化。 |
| 智能体模型更新 | 智能体加载最新模型，与环境继续循环交互。 |

该流程通过强化学习分布式计算框架提供的训练工作流实现。基于此，开发框架主要包含三大核心模块：

| 模块 | 说明 |
| --- | --- |
| 强化学习环境系统 | 提供标准的强化学习环境接口。开发者可以通过标准接口，实现智能体与环境的交互。 |
| 强化学习智能体开发套件 | 提供标准的强化学习智能体接口，以及算法库、模型组件库等工具函数库。开发者可以通过工具函数库快速完成智能体的构建。 |
| 强化学习分布式计算框架 | 提供标准接口，支持开发者按需实现训练工作流，运行单机或分布式的训练及评估任务。 |

### 代码包简介

开发者可以通过腾讯开悟平台所提供的强化学习项目使用开发框架。一个强化学习项目的代码目录如下：

| 目录名 | 介绍 |
| --- | --- |
| `agent/` | 智能体子目录，智能体相关内容均集中于该目录，是开发者核心工作目录。 |
| `conf/` | 配置文件目录，包含运行训练任务相关的配置，例如训练样本批处理大小 `batch_size` 等。 |
| `log/` | 日志目录，存放运行代码测试脚本时生成的日志文件。 |
| `train_test.py` | 代码正确性测试脚本，该脚本会使用当前代码包完成一步训练。建议开发者在启动训练任务前，确保代码已通过该脚本检测。 |

`agent` 目录：

| 目录/文件名 | 介绍 |
| --- | --- |
| `algorithm/` | 算法相关，开发者在该目录下完成算法实现，包含 loss 计算、模型优化等，详情见算法开发。 |
| `feature/` | 特征相关，开发者在该目录下完成数据结构定义和数据处理方法，以及样本处理和奖励计算，详情见数据处理与奖励设计。 |
| `model/` | 模型相关，开发者在该目录下完成模型实现，详情见模型开发。 |
| `workflow/` | 工作流目录，开发者在该目录下完成训练工作流的开发，详情见工作流开发。 |
| `agent.py` | 智能体核心代码文件，开发者在该文件中完成预测、训练等核心函数的实现，详情见智能体开发。 |

标准代码包中都存在一个 `agent_diy` 子文件夹，该文件夹是预定义的智能体模板，可供开发者进行智能体的开发。

`conf` 目录：

| 文件名 | 介绍 |
| --- | --- |
| `configure_app.toml` | 训练任务相关的配置，包括样本大小、样本池大小等。 |

通过对训练流程和代码包的介绍，相信开发者能够对腾讯开悟开发框架建立初步认知。

### 环境系统

强化学习训练流程离不开智能体与环境的持续交互。强化学习环境是基于输入动作，输出观测、奖励等反馈的功能模块，用于表达强化学习算法所求解的问题场景。

开发框架通过场景适配模块，对仿真器进行封装，将其特化的接口、协议转换为强化学习环境统一的接口和协议，供智能体调用。

强化学习环境系统主要提供如下功能：

| 功能 | 说明 |
| --- | --- |
| 接收配置信息 | 用于指定自身初始化方式，比如环境中各种元素的初始状态。 |
| 输出观测、奖励信息 | 可用于智能体预测、训练。 |
| 输出其他信息 | 输出观测、奖励之外的其他信息，供强化学习系统相关组件使用以实现特定功能。其他信息可包括可视化数据、日志数据等，实现的功能包括环境可视化、运行状况监测等。 |
| 接收动作指令 | 完成状态转移并产生新的观测和奖励。 |

开发框架通过场景适配模块，将问题场景进行标准化封装，为开发者提供统一的交互接口与通信协议。由于环境之间存在差异，接口中所涉及的观测、奖励等信息的具体数据结构也有所不同，开发者需查阅所使用环境的官方数据协议文档以获取准确信息。

开发者可以在训练工作流的 `workflow` 中获取到对应环境的实例，通过标准接口实现智能体与环境的交互。

#### `reset(usr_conf)`

`reset` 会将环境重置为环境配置文件中指定的状态，并且返回初始观测。

```python
# usr_conf 为开发者传入的环境配置
obs, state = env.reset(usr_conf=usr_conf)
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| `usr_conf` | `dict` 类型，环境配置文件 |

Returns：

| 参数名 | 介绍 |
| --- | --- |
| `obs` | `dict` 类型，环境观测信息 |
| `state` | `dict` 类型，环境全局信息 |

#### `step(act, stop_game=false)`

环境会执行传入的 `act` 动作指令，完成一次状态转移，并返回新的观测和奖励等信息。

```python
frame_no, _obs, score, terminated, truncated, _state = env.step(act, stop_game=false)
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| `act` | `dict` 类型，环境执行的动作 |
| `stop_game` | `bool` 类型，是否结束当前对局 |

Returns：

| 参数名 | 介绍 |
| --- | --- |
| `frame_no` | `int` 类型，当前环境实例运行时的帧号 |
| `_obs` | `dict` 字典类型，当前帧的观测信息 |
| `score` | `int` 类型，当前帧的奖励信息 |
| `terminated` | `bool` 类型，当前环境实例是否结束 |
| `truncated` | `bool` 类型，当前环境实例是否异常或中断 |
| `_state` | `dict` 字典类型，当前帧的全部状态信息 |

### 智能体开发套件

智能体是强化学习系统中的核心模块。基于训练流程，智能体开发分为四个部分：

| 部分 | 说明 |
| --- | --- |
| 数据处理及奖励设计 | 介绍基于环境观测数据进行特征处理、样本处理和奖励设计的方法。 |
| 模型开发 | 介绍模型开发接口及开发方法。 |
| 算法开发 | 介绍包括算法开发接口及开发方法。 |
| 工作流开发 | 介绍开发者开发自定义训练工作流的方法。 |

#### 数据结构

开发目录：`<智能体文件夹>/feature/definition.py`

开发者需要定义智能体可以使用的数据结构（类）。开发框架已经预先定义好了三种数据类型：`ObsData`, `ActData`, `SampleData`。

`ObsData` 和 `ActData` 分别表示智能体预测的输入和输出，将会由 `agent.predict()` 使用；`SampleData` 为训练样本的数据类型，训练样本将会被 `agent.learn()` 使用，进行模型训练。

`create_cls` 用于动态创建数据结构（类）。`ObsData`, `ActData`, `SampleData` 是训练流程必需的三类，但每一个类的数据结构包含哪些属性完全由开发者自定义，属性名称和属性数量没有限制。

```python
ObsData = create_cls(
    "ObsData",
    feature=None,
)
ActData = create_cls(
    "ActData",
    action=None,
    prob=None,
)
SampleData = create_cls(
    "SampleData",
    npdata=None,
)
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| 第一个参数 | 字符串类型，类的名称 |
| 其余参数 | 类的属性，默认值为 `None`，由开发者自行定义 |

#### 观测处理

开发目录：`<智能体文件夹>/agent.py`

由于环境的 `reset` 和 `step` 接口返回的数据属于原始观测数据，无法直接作为智能体预测时的输入，开发者需要将这部分数据进行特征化。

`observation_process` 将环境返回的观测数据转换成 `ObsData` 类型数据。很多情况下，特征工程包含了大量的数值处理、数据转换和领域知识，建议将大量的特征处理代码在 `<智能体文件夹>/feature/preprocessor.py` 文件中实现，然后由 `observation_process` 进行调用。

```python
def observation_process(self, obs, state=None):
    return ObsData(feature=feature, legal_act=legal_actions)
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| `obs` | `Observation` 类型，`env.reset` 和 `env.step` 返回的环境观测数据 |
| `state` | `EnvInfo` 类型，`env.reset` 和 `env.step` 返回的环境状态数据 |

Return：

| 参数名 | 介绍 |
| --- | --- |
| `ObsData` | 开发者定义的 `ObsData` 类型的数据，将作为 `agent.predict()` 函数的输入。 |

#### 动作处理

开发目录：`<智能体文件夹>/agent.py`

由于环境的 `step` 接口的输入须要满足环境的特定数据协议，开发者需要将智能体预测的输出转换为符合环境 `step` 接口输入要求的数据。

`action_process` 将智能体预测输出的 `ActData` 类型数据转换成环境可以接收的动作数据。

```python
def action_process(self, act_data):
    return act_data.act
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| `act_data` | 开发者定义的 `ActData` 类型的数据 |

Return：环境能处理的动作数据类型，作为 `env.step()` 的输入。

#### 奖励设计

开发目录：`<智能体名称>/feature/definition.py`

这里的奖励特指强化学习中的 Reward，注意要与环境反馈的 Score 进行区分。Score 通常用于衡量智能体在环境中的实际表现。开发者在设计 Reward 时，有非常大的灵活性，不仅可以基于环境返回的观测信息，还可以加入开发者对问题的理解、经验或者知识。

`reward_shaping` 是开发框架预设的奖励设计函数接口，开发者可以通过该函数实现复杂的奖励计算，在训练工作流中调用。

```python
def reward_shaping(obs, _obs, state, _state):
    return reward
```

Parameters：参数个数和类型不限制，可以是环境信息、智能体信息、开发者的经验和知识等。

Return：数值类型，计算出的 `reward` 值。

#### 样本处理

开发目录：`<智能体文件夹>/feature/definition.py`

由于环境与智能体交互过程中产生的轨迹数据无法直接作为智能体训练时的输入，开发者需要将轨迹数据转换为训练样本数据。

`sample_process` 将环境与智能体交互过程中产生的轨迹数据转换成开发者定义的 `SampleData` 类型数据。

```python
@attached
def sample_process(self, list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| `list_game_data` | `list(Frame)` 类型，使用开发者自定义的 `Frame` 作为输入，因为样本一般进行批处理，所以传入列表 |

Return：

| 参数名 | 介绍 |
| --- | --- |
| `list(SampleData)` | `SampleData` 类型的数据组成的列表 |

#### 算法开发

开发目录：`<智能体名称>/algorithm/algorithm.py`

在完成特征处理和奖励设计后，开发者还需要实现强化学习算法，以通过特定优化方法更新模型参数。`learn` 是实现强化学习优化算法的核心方法，该函数输入为训练样本数据，开发者需基于不同的算法完成相关实现，包括优化方法、损失计算等。

```python
def learn(self, list_sample_data):
    """
    Implementing the core method of the algorithm
    实现算法的核心方法
    """
    loss = 0                         # 基于不同算法实现loss计算 Calculate loss
    loss.backward()                  # 计算梯度 Calculate gradient
    self.optimizer.step()            # 通过梯度下降等方法更新模型 Update weights
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| `list_sample_data` | `list` 类型，训练样本（`SampleData`）列表 |

#### 模型开发

开发目录：`<智能体名称>/model/model.py`

一个强化学习模型是基于特征作为输入数据，输出策略的神经网络模型。开发者需要在 `model.py` 文件中实现神经网络模型。开发框架要求，模型类需继承 `torch.nn.Module` 类，即符合 Pytorch 模型的实现规范。

```python
class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
```

#### 工作流开发

训练工作流：在完成智能体开发后，需要进一步实现由分布式计算框架提供的训练工作流接口，使智能体和环境持续交互，收集训练样本，迭代模型参数，最终完成策略的优化。

开发目录：`<智能体名称>/workflow/train_workflow.py`

`workflow` 是训练工作流的核心函数，在 `workflow` 中可自定义训练流程。该函数通过调用智能体和环境提供的接口，完成环境交互、样本收集和模型更新。

```python
@attached
def workflow(envs, agents, logger=None, monitor=None):
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| `envs` | `list` 类型，环境列表，返回当前正在运行的环境。 |
| `agents` | `list` 类型，智能体列表，通过调用开发者实现的 `<智能体名称>/agent.py` 实例化 Agent，并作为输入传入 `workflow`。 |
| `logger` | `Logger` 类型，框架提供的日志组件，接口与 python 的 `logging` 库一致。 |
| `monitor` | `Monitor` 类型，框架提供的监控组件。 |

#### 智能体接口

开发目录：`<智能体名称>/agent.py`

在完成模型和算法后，开发者还需要实现强化学习智能体，智能体使用模型进行决策、与环境交互并通过算法更新模型参数。KaiwuDRL 也提供了智能体相关的接口函数，开发者可以按需实现以下接口函数，并在训练工作流中调用。

`learn`：该函数输入为训练样本数据，开发者需要在该函数中调用算法消费训练样本进行训练。

在不同的训练模式下，该函数使用方法有所不同：

| 训练模式 | 使用方法 |
| --- | --- |
| 单机训练 | 开发者需要在训练工作流中手动调用该函数以进行一步训练。 |
| 分布式训练 | 该函数作为训练函数会被循环执行，无需开发者手动调用；但该函数还作为样本发送函数，开发者需要在训练工作流中手动调用，以将样本发送至样本池。 |

```python
def learn(self, list_sample_data):
    self.algo.learn(list_sample_data)        # 调用算法消费训练样本进行训练 Call algorithm to train model
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| `list_sample_data` | `list` 类型，训练样本（`SampleData`）列表；在 Learner 中会从样本池按照配置项 `train_batch_size` 采样一批样本，作为输入传入 `learn()` 函数。 |

`predict`：该方法通过调用模型进行预测，通常在训练时调用该方法，依策略的概率分布采样或引入随机概率。

```python
@predict_wrapper
def predict(self, list_obs_data, list_state):
    return [ActData]
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| `list_obs_data` | `list` 类型，观测数据（`ObsData`）列表 |
| `list_state` | 可选参数，`list` 类型，环境返回的状态数据列表 |

Return：

| 参数名 | 介绍 |
| --- | --- |
| `List(ActData)` | `list` 类型，开发者定义的动作数据（`ActData`）列表 |

`exploit`：该方法通过调用模型进行预测，通常在评估时调用该方法，选取策略中概率最高的动作或者策略认为最优的动作。

```python
@exploit_wrapper
def exploit(self, observation):
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| `observation` | `dict` 类型，环境观测字典，评估工作流中将原始的环境观测信息作为输入传入 `agent.exploit()`。 |

Return：

| 参数名 | 介绍 |
| --- | --- |
| `action` | `list` 类型，动作列表，环境可以直接使用的动作指令 |

`load_model`：智能体通过该接口完成模型参数加载。Actor 会从模型池中获取最新模型参数文件，开发者需要手动调用 `load_model()` 函数，使智能体完成模型参数加载。

```python
@load_model_wrapper
def load_model(self, path=None, id="1"):
    # When loading the model, you can load multiple files,
    # and it is important to ensure that each filename matches the one used during the save_model process.
    # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
    model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
    self.model.load_state_dict(
        torch.load(model_file_path, map_location=self.device),
    )
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| `path` | `string` 类型，加载模型参数文件的路径，开发框架根据使用场景得到相应的路径，并作为输入传入 `load_model` |
| `id` | `string` 类型，模型参数文件的 id，使用 id 指定加载的模型参数文件 |

`save_model`：开发者可以通过该函数保存当前时刻的模型文件及智能体代码包，开发框架会将开发者需要保存的内容打包为 zip 格式的文件。

当开发者使用腾讯开悟客户端开发时，开发框架会在客户端指定目录下存储该 zip 文件。当开发者使用腾讯开悟平台时，开发框架会将该 zip 文件存储在云端，开发者可以通过平台的训练管理模块查看每一个训练任务的 zip 文件，即模型。

```python
@save_model_wrapper
def save_model(self, path=None, id="1"):
    # To save the model, it can consist of multiple files,
    # and it is important to ensure that each filename includes the "model.ckpt-id" field.
    # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
    model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

    # Copy the model's state dictionary to the CPU
    # 将模型的状态字典拷贝到CPU
    model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
    torch.save(model_state_dict_cpu, model_file_path)
```

Parameters：

| 参数名 | 介绍 |
| --- | --- |
| `path` | `string` 类型，模型文件保存的路径，开发框架根据使用场景得到相应的路径，并作为输入传入 `save_model` |
| `id` | `string` 类型，模型文件的索引，开发框架获取到模型池中最新模型的索引，并作为输入传入 `save_model` |

### 分布式计算框架

在强化学习项目的开发中，分布式计算框架是支撑大规模训练任务的核心基础设施。本开发框架提供了由腾讯王者荣耀团队自主研发的强化学习分布式计算框架 KaiwuDRL，通过并行化计算、高效资源调度和分布式协同优化，显著提升智能体训练的效率与稳定性。

KaiwuDRL 的整体架构包括 Environment、Aisrv、Actor、Learner 等强化学习组件（均支持多实例并行运行）。此外，还集成了通信、日志、监控、对象存储等基础组件。MemoryPool 和 ModelPool 仅在分布式训练时启用。

组件介绍：

| 组件名称 | 功能描述 |
| --- | --- |
| Environment | 环境服务组件，负责运行强化学习环境，支持通过标准接口与环境交互，并返回环境的观测 obs。 |
| Aisrv | 训练流程中枢，负责收集环境样本，运行训练、评估工作流，以及处理各个组件间的数据传输。 |
| Actor | 预测服务组件，负责响应 Aisrv 的预测请求，调用智能体 `predict()` 或 `exploit()` 函数生成动作决策结果。 |
| Learner | 训练服务组件，负责采集训练样本，调用智能体 `learn()` 函数完成梯度计算及模型迭代。 |
| MemoryPool | 样本存储组件，简称样本池。负责存储训练样本，接收 Aisrv 打包的训练样本，发往 Learner 用于智能体训练。 |
| ModelPool | 模型存储组件，简称模型池。负责存储模型参数文件，接收 Learner 产出的模型参数文件，将最新的模型参数文件发送给 Actor。 |
| 日志 | 日志采集组件，负责记录强化学习系统中各个组件的运行日志，支持通过标准接口上报日志。 |
| 监控 | 监控采集组件，负责采集系统资源使用率、训练指标趋势等数据，支持通过标准接口上报数据指标。 |

KaiwuDRL 提供预测服务和训练服务：

| 服务 | 流程 |
| --- | --- |
| 预测服务 | Aisrv → Environment：发送环境配置并创建新一局 episode；<br>Environment → Aisrv：返回原始观测数据；<br>Aisrv → Actor：Aisrv 基于原始观测数据，向 Actor 发送预测请求；<br>Actor：使用预测请求中的观测进行特征处理，智能体基于特征处理后的数据进行预测，并且将预测数据处理为环境可以识别的动作指令，发送给 Aisrv；<br>Aisrv：使用动作指令与环境 Environment 进行交互，Environment 返回新的观测。 |
| 训练服务 | Aisrv：预测服务不断产生轨迹数据，Aisrv 完成样本处理，并发送至样本池；<br>Learner：从样本池按批采集样本进行训练，并将最新的模型参数同步至 Actor。 |

KaiwuDRL 提供训练、评估工作流的接口函数，开发者可以按需灵活调用上述组件和服务，以实现模型的训练和评估。

评估工作流：在运行训练任务（训练工作流）并获得模型文件后，可以通过运行评估任务（评估工作流），对模型能力进行验证。当开发者在使用腾讯开悟平台所提供的强化学习项目时，评估工作流由腾讯开悟官方实现，开发者无法修改。评估工作流会调用开发者自定义的 `agent.exploit()` 函数。

#### 分布式训练配置项

开发目录：`/conf/configure_app.toml`

Learner 在每一次执行 `learn()` 函数时，会从样本池采样一批样本作为输入，并按照开发者配置的频次 `dump_model_freq` 保存模型参数文件，模型同步服务将按照配置 `model_file_sync_per_minutes` 将模型参数文件推送至模型池。

Actor 中的模型同步服务将按照配置 `model_file_sync_per_minutes`，从模型池获取最新模型参数文件。

Word 文档中的相关配置项示例：

```toml
[app]
# The time interval for executing the learn() function, configurable to throttle the Learner and balance sample production/consumption.
# 执行learn函数进行训练的时间间隔，可通过该配置让Learner休息以调节样本生产消耗比
learner_train_sleep_seconds = 0.001

# Replay buffer configurations
# 样本池容量
replay_buffer_capacity = 4096

# The ratio of the sample pool capacity that triggers training
# 当样本池中的样本占总容量的比例达到该值时，启动训练
preload_ratio = 1.0

# When new samples are added to the sample pool, the logic for removing old samples: reverb.selectors.Lifo, reverb.selectors.Fifo
# 当新样本加入样本池时，旧样本的移除逻辑，可选项：reverb.selectors.Lifo, reverb.selectors.Fifo
# reverb.selectors.Lifo：先进后出(Last In, First Out)
# reverb.selectors.Fifo：先进先出(First In, First Out)
reverb_remover = "reverb.selectors.Fifo"

# The sampling logic of the Learner from the sample pool: reverb.selectors.Fifo, reverb.selectors.Uniform
# Learner从样本池中采样的逻辑，可选项：reverb.selectors.Fifo, reverb.selectors.Uniform
# reverb.selectors.Uniform：Samples are selected uniformly at random from the replay buffer, with each sample having an equal probability of being chosen.
# reverb.selectors.Uniform：从回放缓冲区中随机均匀地选择样本，每个样本被选中的概率相同。
# reverb.selectors.Fifo：Samples are selected in the order they were added to the replay buffer.
# reverb.selectors.Fifo：按照先进先出从回放缓冲区中选择样本。
reverb_sampler = "reverb.selectors.Uniform"

# Training batch size limit for Learner
# Learner训练时样本批处理大小
train_batch_size = 2048

# Model dump frequency (steps)
# 训练间隔多少步输出模型参数文件
dump_model_freq = 1000

# The Learner pushes model updates, and the frequency at which Actors fetch the model (in minutes).
# Learner推送模型参数文件至模型池，以及Actor从模型池获取模型参数文件的频次（单位：分钟）
model_file_sync_per_minutes = 1

# he number of model updates pushed per learner iteration, and the maximum number of updates each actor can fetch at once (cap: 50).
# Learner每次推送模型参数文件，以及Actor每次获取模型参数文件的数量（上限：50）
modelpool_max_save_model_count = 1
```

当前仓库的 `code/conf/configure_app.toml` 与 Word 文档示例相比，保留了更多配置项，并将 `replay_buffer_capacity` 设为 `10000`、`dump_model_freq` 设为 `100`。
