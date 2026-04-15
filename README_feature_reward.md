# 特征工程与奖励设计说明

本文档描述当前工作区中 `agent_ppo` 的最新特征工程、局部地图建模、模型输入结构与奖励设计实现，用于补充赛题文档，不替换原有 [赛题.md](赛题.md)。

## 1. 相关文件

- `code/agent_ppo/feature/preprocessor.py`
- `code/agent_ppo/conf/conf.py`
- `code/agent_ppo/model/model.py`
- `code/agent_ppo/workflow/train_workflow.py`

## 2. 当前特征工程实现

### 2.1 总体输入维度

当前观测输入由两部分组成：

- 标量向量特征：`64` 维
- 局部空间图特征：`21 x 21 x 3 = 1323` 维

总输入维度为：

```text
64 + 1323 = 1387
```

其中标量向量特征由以下部分拼接：

- `hero_state`: 4 维
- `station`: 21 维，表示最近 `3` 个驿站，每个驿站 7 维
- `charger`: 7 维，表示最近 `1` 个充电桩
- `npc`: 7 维，表示最近 `1` 个 NPC
- `legal_action`: 8 维
- `indicators`: 3 维
- `warehouse`: 5 维
- `mode_onehot`: 4 维
- `target`: 5 维

### 2.2 坐标与距离编码

当前实现统一采用以下规则：

- 绝对坐标按地图范围 `[0, 127]` 归一化到 `[0, 1]`
- Chebyshev 距离按最大距离 `127` 归一化到 `[0, 1]`
- 方向仍采用欧式归一化方向向量编码为 `dir_x, dir_z`

这样做的目的：

- 方向向量用于表达“往哪个方向走”
- Chebyshev 距离用于表达“按 8 邻接移动还差多少步”

### 2.3 保留的原有特征

以下原有特征全部保留：

- `hero_state`: `battery_ratio`, `package_count_norm`, `cur_pos_x_norm`, `cur_pos_z_norm`
- `station`: 最近 `3` 个驿站，每个驿站为
  - `found`
  - `dir_x`
  - `dir_z`
  - `abs_x_norm`
  - `abs_z_norm`
  - `dist_norm`
  - `is_target`
- `charger`: 最近 `1` 个充电桩
- `npc`: 最近 `1` 个 NPC
- `legal_action`: 环境合法动作掩码与地图合法动作掩码的交集
- `indicators`: `has_package`, `battery_low`, `target_visible`

其中：

- 驿站排序仍然是“目标驿站优先，再按距离排序”
- `battery_low` 的判断仍为

```text
battery / battery_max < 0.3
```

### 2.4 新增仓库显式特征

当前新增 `warehouse` 5 维特征：

- `warehouse_abs_x_norm`
- `warehouse_abs_z_norm`
- `warehouse_h_norm`
- `warehouse_w_norm`
- `warehouse_dist_to_area_norm`

其中距离不是到仓库中心点的距离，而是到仓库区域的最小 Chebyshev 距离：

```text
dx = max(x_min - x, 0, x - x_max)
dz = max(z_min - z, 0, z - z_max)
dist = max(dx, dz)
```

当前仓库区域边界的实现假设为：

- `warehouse.pos` 表示仓库区域的语义中心点
- 结合协议中的 `w/h` 还原仓库矩形边界

### 2.5 新增模式 one-hot 特征

当前新增 `mode_onehot` 4 维特征，对应锁定目标状态机中的四种模式：

- `warehouse_refill`
- `station`
- `charger`
- `warehouse_charge`

编码方式为标准 one-hot。

### 2.6 新增当前目标显式特征

当前新增 `target` 5 维特征：

- `target_dir_x`
- `target_dir_z`
- `target_abs_x_norm`
- `target_abs_z_norm`
- `target_dist_to_area_norm`

其中：

- 方向不是指向“目标中心点”，而是指向“当前点投影到目标区域上的最近点”
- 绝对位置使用目标的锚点位置
  - 驿站：驿站中心
  - 充电桩：充电桩中心
  - 仓库：仓库中心
- 距离使用“到目标区域的最小 Chebyshev 距离”

### 2.7 目标区域定义

当前实现中，不同目标的“目标区域”定义如下：

- `station`
  - 按 `3x3` 区域处理
- `charger`
  - 按充电桩 `range` 近似为轴对齐矩形区域，最小半径不小于 `1`
- `warehouse_refill` / `warehouse_charge`
  - 按仓库 `w/h` 对应的矩形区域处理

### 2.8 局部空间图特征

当前新增局部空间图特征，尺寸为：

```text
21 x 21 x 3
```

实现上采用 `channel-first` 排布，即：

```text
(3, 21, 21)
```

再展平后拼接到最终观测向量中。

三个通道分别如下。

#### 2.8.1 通道 1：障碍图

通道 1 基于环境返回的 `map_info` 构造：

- `0` 表示可通行
- `1` 表示障碍/不可通行

即直接把协议中的

```text
map_info: 1=可通行, 0=不可通行
```

转换为障碍图：

```text
obstacle_map = 1 - traversable_map
```

#### 2.8.2 通道 2：NPC danger map

通道 2 不是简单地标注 NPC 当前位置，而是对局部 `21x21` 每个格子都计算一个风险值。

当前风险场使用与 NPC 奖励惩罚一致的衰减思想，对所有 NPC 取最近风险：

```text
danger(cell) = max_i exp(-(max(d_inf(cell, npc_i), 1) - 1) / 1.5)
```

并在 `NPC_PENALTY_RADIUS` 外截断为 `0`。

这样做的意义：

- 不仅告诉模型 NPC 在哪
- 还告诉模型“哪些区域危险更高”
- 与奖励中的 NPC 风险惩罚语义一致

#### 2.8.3 通道 3：target potential map

通道 3 表示当前局部区域中，每个格子相对当前 waypoint 的“接近潜力”。

当前 waypoint 来自目标可见时的局部目标点，或来自全局 A* 路径中当前视野内 BFS 可达的前方路径点。对该 waypoint 反向做局部 BFS 后，定义：

```text
target_potential(p) = 1 - bfs_dist(p, waypoint) / max_reachable_bfs_dist
```

其含义是：

- 值越大：从该格子到 waypoint 的局部 BFS 距离越短
- 不可达格子为 `0`

这不是 reward 本身，而是局部导航势场的观测表达。

### 2.9 合法动作掩码

当前合法动作仍然不是只信环境返回的 `legal_action`，而是：

```text
final_mask = env_mask ∩ map_mask
```

其中：

- `env_mask` 来自环境返回
- `map_mask` 基于 `21x21 map_info` 计算

规则如下：

- 直线动作：目标格必须可通行
- 斜向动作：目标格必须可通行，且相邻水平格与垂直格至少一个可通行

若交集全 0：

- 优先回退到环境掩码
- 若环境掩码也不可用，则回退为全 1

## 3. 当前模型输入结构

当前模型不再把全部特征直接送进单一 MLP，而是采用双分支：

```text
标量特征 64D  -> MLP -> 64D
局部地图 3x21x21 -> CNN -> 64D
拼接 -> 融合 MLP -> Actor / Critic
```

其中：

- 标量分支负责任务状态、目标、仓库、电量等高层语义
- 空间分支负责局部障碍、NPC 风险场、目标势场

## 4. 当前目标状态机

当前目标状态机仍分为 4 类：

- `warehouse_refill`
- `station`
- `charger`
- `warehouse_charge`

锁定目标使用：

```text
g_t = (mode_t, target_id_t)
```

当前依旧采用“事件触发重规划”，不是每步重选目标。

触发重规划的时机：

- episode 开始
- `package_count` 变化
- `delivered` 变化
- 成功进入充电桩
- 成功进入仓库
- 当前目标失效
- 当前锁定驿站不再安全

## 5. 当前目标切换逻辑

### 5.1 无包裹时

若 `package_count == 0`：

1. 判断是否能到达仓库区域
2. 若能到达，则进入 `warehouse_refill`
3. 若无法到达，则进入最近充电桩 `charger`

### 5.2 有包裹时

若 `package_count > 0`：

1. 选择最近目标驿站
2. 若当前电量足以安全送达并撤离，则保持 `station`
3. 否则在最近充电桩和仓库之间选择最近补能点

## 6. 当前安全配送判定

当前安全性判定依旧基于 Chebyshev 距离近似，但已部分改为“到区域的最小距离”：

```text
d1 = dist(cur, station_area)
d2 = min(dist(station_center, nearest_charger_area), dist(station_center, warehouse_area))
need = d1 + d2 + SAFETY_MARGIN
```

若：

```text
battery >= need
```

则认为当前目标驿站仍可安全配送。

当前配置：

- `SAFETY_MARGIN = 8.0`

## 6.1 全局地图与重规划

每局维护一张 `128x128` 全局地图，状态包括：

- `GLOBAL_MAP_UNKNOWN = -1`
- `GLOBAL_MAP_FREE = 0`
- `GLOBAL_MAP_BLOCKED = 1`
- `GLOBAL_MAP_PRIOR_FREE = 2`

其中 `PRIOR_FREE` 是弱先验，只在仓库、驿站、充电桩区域仍为 `UNKNOWN` 时写入；真实 `map_info` 观测会覆盖该先验。A* 代价当前为：

```text
BLOCKED = inf
FREE = 1.0
PRIOR_FREE = 1.2
UNKNOWN = 1.5
```

并额外叠加：

```text
cell_cost += VISIT_GLOBAL_COST_WEIGHT * visit_cost
```

全局 A* 不再维护 NPC 历史热点图，也不把 NPC 风险大范围加入全局路径代价。NPC 仍通过最近 NPC 标量特征、局部 `21x21` danger map 和 NPC 奖励项参与策略学习。

全局路径不是每步重算，stuck 触发已扩展为：

- 连续不移动
- waypoint BFS 距离无进展
- waypoint 连续脱离局部视野
- 短窗口震荡
- 重复访问且无进展
- 当前位置偏离缓存全局路径过远

## 7. 当前奖励设计实现

奖励函数位于 `code/agent_ppo/feature/preprocessor.py`，总体形式为：

```text
r =
  delivery_reward
  + waypoint_progress_reward
  + waypoint_reached_reward
  - visitation_penalty
  + charge_event_reward
  + warehouse_event_reward
  - npc_penalty
  + npc_escape_reward
  - step_penalty
  + terminal_penalty
```

### 7.1 投递主奖励

```text
delivery_reward = DELIVERY_REWARD_SCALE * delta_delivered
```

当前配置：

- `DELIVERY_REWARD_SCALE = 2.5`

### 7.2 进度 shaping

当前进度奖励只在以下条件下计算：

```text
mode_t == mode_{t-1}
target_id_t == target_id_{t-1}
```

当前进度 shaping 使用 waypoint BFS 距离差，而不是全局直线距离：

```text
waypoint_progress_reward = alpha(mode) * clip(prev_bfs_dist_to_waypoint - cur_bfs_dist_to_waypoint, -1, 1)
```

当前系数：

- `station = 0.05`
- `charger = 0.04`
- `warehouse_refill = 0.05`
- `warehouse_charge = 0.03`

### 7.3 充电事件奖励

仅在：

```text
charger_count_t > charger_count_{t-1}
```

且上一时刻模式为 `charger` 时生效。

当前配置：

- `CHARGE_EVENT_REWARD = 0.20`

### 7.4 仓库事件奖励

仅在：

```text
warehouse_count_t > warehouse_count_{t-1}
```

时生效，并区分两类语义：

- `warehouse_refill`
  - `prev_package_count == 0`
  - 当前 `package_count > 0`
  - 奖励 `0.30`

- `warehouse_charge`
  - 奖励 `0.15`

### 7.5 NPC 平滑风险惩罚

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

### 7.6 Waypoint 与访问惩罚

当前进度奖励改为局部 BFS waypoint 进度。waypoint 优先取局部可达目标；目标不在局部可达范围内时，从缓存全局 A* 路径中选择当前 `21x21` 视野内、BFS 可达且距离不超过 `WAYPOINT_LOOKAHEAD_RADIUS` 的最远前方路径点。

若当前 waypoint 和上一时刻 waypoint 一致：

```text
waypoint_progress_reward =
    alpha(mode) * clip(prev_bfs_dist_to_waypoint - cur_bfs_dist_to_waypoint, -1, 1)
```

到达 waypoint 时额外增加：

- `WAYPOINT_REACHED_REWARD = 0.04`

全局访问矩阵为 `128x128`，每局重置。访问惩罚为：

```text
visit_cost = min(visit_count[z][x], VISIT_COUNT_CAP) / VISIT_COUNT_CAP
visitation_penalty = VISIT_PENALTY_SCALE * visit_cost
```

当前配置：

- `WAYPOINT_REPLAN_STUCK_STEPS = 3`
- `WAYPOINT_PROGRESS_TOL = 0.1`
- `WAYPOINT_LOOKAHEAD_RADIUS = 8`
- `NO_MOVE_STUCK_STEPS = 2`
- `WAYPOINT_MISSING_REPLAN_STEPS = 2`
- `PATH_DEVIATION_REPLAN_DIST = 4`
- `VISIT_COUNT_CAP = 10`
- `VISIT_PENALTY_SCALE = 0.01`
- `VISIT_GLOBAL_COST_WEIGHT = 0.3`

### 7.7 每步惩罚

当前配置：

- `STEP_PENALTY = 0.002`

### 7.8 终局惩罚

当前终局惩罚已收口到 `preprocessor`：

- 碰撞失败：`-2.0`
- 电量耗尽：`-1.2`

对应配置：

- `TERMINAL_COLLISION_PENALTY = -2.0`
- `TERMINAL_ENERGY_DEPLETED_PENALTY = -1.2`

## 8. 与旧版实现相比的主要变化

相较此前的 50 维增强版 PPO 基线，当前实现新增：

- 显式仓库特征
- 显式模式 one-hot
- 显式当前目标特征
- 到目标区域的最小 Chebyshev 距离编码
- `21x21x3` 局部空间图
- 向量分支 + CNN 空间分支的双分支模型
- 进度奖励从“点距离”改为“区域距离”

## 9. 结论

当前工作区中的 `agent_ppo` 已经不再是单纯的向量型 PPO 基线，而是一个包含以下能力的增强版实现：

- 保留原始 50 维手工特征
- 新增仓库 / 模式 / 当前目标显式特征
- 引入 `21x21x3` 局部空间图
- 使用风险场表达 NPC 威胁
- 使用目标势场表达局部导航方向
- 使用区域距离统一目标选择、特征编码与进度奖励
- 用双分支网络分别处理高层状态与局部空间结构
