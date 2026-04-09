# 特征工程与奖励设计说明

本文档是对当前工作区中 `agent_ppo` 的特征工程和奖励设计实现的补充说明，不替换原有 [README.md](C:/Users/26635/Desktop/Preliminaries/README.md)。

## 1. 相关文件

- `code/agent_ppo/feature/preprocessor.py`
- `code/agent_ppo/conf/conf.py`
- `code/agent_ppo/workflow/train_workflow.py`

## 2. 当前特征工程实现

### 2.1 总体特征维度

当前观测输入维度为 `50`，由以下部分拼接：

- `hero_state`: 4 维
- `station`: 21 维，表示最近 `3` 个驿站，每个驿站 7 维
- `charger`: 7 维，表示最近 `1` 个充电桩
- `npc`: 7 维，表示最近 `1` 个 NPC
- `legal_action`: 8 维
- `indicators`: 3 维

对应配置见 [conf.py](C:/Users/26635/Desktop/Preliminaries/code/agent_ppo/conf/conf.py)。

### 2.2 英雄自身特征

`hero_state` 包含：

- `battery_ratio`
- `package_count_norm`
- `cur_pos_x_norm`
- `cur_pos_z_norm`

其中 `battery_low` 的判断规则仍然保留为：

```text
battery / battery_max < 0.3
```

### 2.3 驿站特征

当前不是只取最近 1 个驿站，而是取最近 3 个驿站。

每个驿站特征为 7 维：

- `found`
- `dir_x`
- `dir_z`
- `abs_x_norm`
- `abs_z_norm`
- `dist_norm`
- `is_target`

当前排序规则不是简单最近，而是：

1. 先把当前包裹对应的目标驿站排在前面
2. 再按 Chebyshev 距离排序

这里的设计和“最多携带 3 个包裹”是对齐的，所以 `TopK=3` 是当前实现中的主方案。

### 2.4 充电桩特征

当前增加了最近 1 个充电桩特征，结构同样为 7 维：

- `found`
- `dir_x`
- `dir_z`
- `abs_x_norm`
- `abs_z_norm`
- `dist_norm`
- `battery_low_and_need_charge`

其中 `extra_flag` 直接使用 `battery_low`。

### 2.5 NPC 特征

当前增加了最近 1 个 NPC 特征，结构同样为 7 维：

- `found`
- `dir_x`
- `dir_z`
- `abs_x_norm`
- `abs_z_norm`
- `dist_norm`
- `is_threat_close`

当前 `is_threat_close` 的实现方式是：

```text
abs(npc_x - cur_x) <= 10 且 abs(npc_z - cur_z) <= 10
```

也就是把“当前无人机是否处于该 NPC 的 21x21 观测范围内”作为威胁标志。

### 2.6 距离与方向的编码方式

当前实现采用了两套不同但互补的编码：

- `dir_x, dir_z`：使用欧式归一化方向向量
- `dist_norm`：使用 Chebyshev 距离归一化

这样做的原因是：

- 方向向量更适合用欧式归一化表达“朝哪个方向走”
- 距离更适合用 Chebyshev 表达“还需要多少步”

### 2.7 合法动作掩码

当前合法动作不是只信环境返回的 `legal_action`，而是采用：

```text
final_mask = env_mask ∩ map_mask
```

其中：

- `env_mask` 来自 `obs["legal_action"]`
- `map_mask` 基于 `map_info` 的 `21x21` 可通行网格计算

`map_mask` 的规则如下：

- 直线动作：目标格必须可通行
- 斜向动作：目标格必须可通行，且水平相邻格与垂直相邻格至少一个可通行

这对应“防穿角”规则，避免无人机从两堵墙对角缝隙中挤过去。

如果交集结果全 0，当前实现会回退到环境 `env_mask`；若环境掩码也不可用，则回退为全 1。

### 2.8 当前未纳入特征的仓库信息

仓库没有进入显式特征向量，但仓库信息已经进入目标选择和奖励逻辑。

当前对仓库位置的处理假设是：

```text
warehouse.pos 本身就是语义中心点
```

因此当前不会再使用 `w/h` 对仓库位置做偏移修正。

## 3. 当前奖励设计实现

### 3.1 设计目标

奖励设计不再把任务看成单阶段“走到驿站”，而是按循环任务处理：

```text
补货 -> 配送 -> 补能 -> 继续配送
```

因此当前实现引入了锁定目标状态机和事件触发重规划。

### 3.2 目标状态机

当前模式分为 4 类：

- `warehouse_refill`
- `station`
- `charger`
- `warehouse_charge`

锁定目标使用：

```text
g_t = (mode_t, target_id_t)
```

当前不是每步重选目标，而是事件触发重选。

### 3.3 触发重规划的时机

当前实现中，以下情况会触发目标重选：

- episode 开始
- `package_count` 变化
- `delivered` 变化
- 成功进入充电桩
- 成功进入仓库
- 当前目标失效
- 当前锁定驿站不再安全

其他时刻保持当前目标不变。

### 3.4 模式切换逻辑

当前逻辑如下：

1. `package_count == 0`
   先判断当前电量是否足以直接到达仓库
   - 若可到达：进入 `warehouse_refill`
   - 若不可到达：进入 `charger`
   - 充电完成后重新规划，此时回到 `warehouse_refill`

2. `package_count > 0`
   优先判断最近目标驿站是否仍可安全配送

3. 若安全
   进入或保持 `station`

4. 若不安全
   在“最近充电桩”和“最近仓库”之间选择最近补能点
   - 充电桩更近：`charger`
   - 仓库更近：`warehouse_charge`

其中，空载回仓的可达性判定使用：

```text
battery >= dist(cur, warehouse)
```

到达仓库后会立即补满电量和包裹，因此这里不再额外叠加撤离距离。

### 3.5 安全配送判定

当前安全性判定基于 Chebyshev 距离近似：

```text
d1 = dist(cur, station)
d2 = min(dist(station, nearest_charger), dist(station, warehouse))
need = d1 + d2 + SAFETY_MARGIN
```

若：

```text
battery >= need
```

则认为当前驿站仍然可安全配送。

安全边际配置在 [conf.py](C:/Users/26635/Desktop/Preliminaries/code/agent_ppo/conf/conf.py)：

- `SAFETY_MARGIN = 8.0`

### 3.6 奖励组成

当前奖励函数位于 [preprocessor.py](C:/Users/26635/Desktop/Preliminaries/code/agent_ppo/feature/preprocessor.py)，总体形式为：

```text
r =
  delivery_reward
  + progress_reward
  + charge_event_reward
  + warehouse_event_reward
  - npc_penalty
  + npc_escape_reward
  - block_penalty
  - step_penalty
  + terminal_penalty
```

各项含义如下。

#### 3.6.1 投递主奖励

```text
delivery_reward = DELIVERY_REWARD_SCALE * delta_delivered
```

当前配置：

- `DELIVERY_REWARD_SCALE = 2.5`

#### 3.6.2 同目标进度奖励

只有在以下条件成立时才计算：

```text
mode_t == mode_{t-1}
target_id_t == target_id_{t-1}
```

然后按当前锁定目标的 Chebyshev 距离变化给势函数奖励：

```text
progress_reward = alpha(mode) * clip(prev_dist - cur_dist, -1, 1)
```

当前系数：

- `station = 0.05`
- `charger = 0.04`
- `warehouse_refill = 0.05`
- `warehouse_charge = 0.03`

#### 3.6.3 充电事件奖励

仅在：

```text
charger_count_t > charger_count_{t-1}
```

且上一时刻模式为 `charger` 时生效。

当前配置：

- `CHARGE_EVENT_REWARD = 0.20`

#### 3.6.4 仓库事件奖励

仅在：

```text
warehouse_count_t > warehouse_count_{t-1}
```

时生效，并区分两种语义：

- `warehouse_refill`
  - `prev_package_count == 0`
  - 当前 `package_count > 0`
  - 奖励 `0.30`

- `warehouse_charge`
  - 奖励 `0.15`

#### 3.6.5 NPC 平滑风险惩罚

当前使用最近 NPC 的 Chebyshev 距离做平滑惩罚：

```text
if d_npc <= 8:
    penalty = 0.20 * exp(-(d_npc - 1) / 1.5)
```

同时保留轻微的脱险奖励：

```text
若上一时刻 d_prev <= 4：
    reward += 0.02 * clip(d_npc - d_prev, -1, 1)
```

#### 3.6.6 阻塞惩罚

若：

- `cur_pos == prev_pos`
- 且本步没有 `charge_event`
- 且本步没有 `warehouse_event`

则给轻量阻塞惩罚：

- `BLOCK_PENALTY = 0.01`

#### 3.6.7 每步惩罚

当前配置：

- `STEP_PENALTY = 0.002`

#### 3.6.8 终局惩罚

当前终局惩罚已经收口到 `preprocessor` 中：

- 碰撞失败：`-2.0`
- 电量耗尽：`-1.2`

对应配置：

- `TERMINAL_COLLISION_PENALTY = -2.0`
- `TERMINAL_ENERGY_DEPLETED_PENALTY = -1.2`

## 4. 训练流程侧配合修改

当前训练流程已同步调整，见 [train_workflow.py](C:/Users/26635/Desktop/Preliminaries/code/agent_ppo/workflow/train_workflow.py)。

关键变化有两点：

- 奖励直接使用 `observation_process()` 返回的 `remain_info["reward"]`
- workflow 末尾的 `final_reward` 不再重复追加失败惩罚，统一设为 `0.0`

这样可以避免：

- 奖励函数重复调用
- 状态快照错位
- 终局惩罚重复计算

## 5. 当前实现与原始基线的主要差异

相较原始基线，当前工作区已经完成了以下增强：

- 驿站特征从“最近 1 个”扩展为“最近 3 个，目标优先”
- 新增最近 1 个充电桩特征
- 新增最近 1 个 NPC 特征
- 合法动作掩码从单纯依赖环境返回，增强为 `env_mask ∩ map_mask`
- 距离度量统一偏向 Chebyshev，更贴近 8 邻接移动环境
- 奖励从“投递奖励 + 步惩罚”扩展为“状态机 + 事件奖励 + 平滑 shaping”
- workflow 中的奖励接入方式已经和新奖励逻辑对齐

## 6. 当前实现里的注意点

以下是当前实现状态，不是新的修改建议：

- 仓库未加入显式特征，但参与目标选择和补能判定
- `is_threat_close` 当前基于“最近 NPC 是否覆盖当前无人机”判断，不是“任意 NPC 覆盖”
- 进度奖励当前判断的是“模式和目标 id 未变”，没有进一步检查目标位置是否发生刷新
- 合法动作掩码当前不是纯 `map_mask`，而是 `env_mask ∩ map_mask`

## 7. 结论

当前工作区的 `agent_ppo` 已经不再是原始稀疏奖励基线，而是一个带有以下特征的增强版 PPO 基线：

- 50 维观测输入
- 显式充电桩与 NPC 特征
- 地图约束合法动作
- 锁定目标状态机
- 事件触发重规划
- 面向循环配送任务的奖励设计

这份文档仅描述当前已经落地到代码中的实现，不包含尚未执行的候选方案。
