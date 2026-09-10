# LunarLander: Powered-Braking Onset

日期：2026-09-10。优先级：第一，先实施本实验。状态：实验计划，尚未实现、训练或冻结确认性协议。

## 前置项目 and 结果 and paper初稿

/Users/cattie/myProjects/highway_transition_timing

## Purpose and Claim

检验 ONSET 的匹配暴露、行为确认和 seed 级比较能否扩展到非驾驶物理控制：不同奖励条件的着陆器，何时由下降进入持续动力制动？

本文新增的证据是第二领域中的方法适用性，不是 highway 策略的零样本迁移。新环境必须重新训练策略。即使结果成立，也不宣称 ONSET 自动发现事件、从行为唯一反推出奖励，或比 HIGHLIGHTS/DISAGREEMENTS 更有助于人类理解。

本计划中的权重、网格、窗口和训练预算是 pilot 起点，不是已验证的最佳配置。正式运行前生成独立的 `protocol.md` 和带校验和的 manifest。不得更改 highway A/B/C、held-out v2 的模型、网格或判定规则。

## Research Questions and Hypotheses

| ID             | 问题与预先声明的方向                                                                                              | 解释边界                                                         |
| -------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| L1，主要       | 在匹配的可恢复制动场景中，高推力成本条件 ECON 的制动 onset 晚于高下降速度代价条件 DESC：`tau_ECON - tau_DESC > 0` | 待检验假设；不能以出现此排序为模型选择条件                       |
| L2，机制支持   | 对同一固定策略降低初始下降速度，制动 onset 倾向于延后，或在观察期内不再发生                                       | 低压力下点火方式可能改变；同时报告事件率，不能仅看成功配对的时间 |
| L3，物理支持   | 时间对比伴随可解释的高度、下降速度轨迹差异                                                                        | 跨策略轨迹已经分化，不强求 onset 高度一定具有与时间相反的排序    |
| L4，测量有效性 | ONSET 排除一次性点火、地面接触造成的减速，以及未产生制动的指令                                                    | 与简单检测器比较事件语义，不预设一定有更大的组间差异             |

均衡条件 BAL 用来检查关系形状，不要求三组严格单调，也不要求复制 highway 的排序反转。

## Environment and Exposure

以 Gymnasium `LunarLander-v3` 离散版本为基础，主实验关闭风，固定重力，保留四动作控制。它提供发动机动作和着陆器运动状态；原始环境的燃料并非有限库存，因此本文用“推力使用成本”表述经济性。[官方文档](https://gymnasium.farama.org/main/environments/box2d/lunar_lander/)

需要明确标为 **controlled-reset, reward-modified LunarLander**，不能把结果当作未经修改的标准 benchmark 分数。

- 暴露时刻 `t0` 是固定场景完成初始化、策略取得控制权的边界。所有条件从同一物理状态开始，不能让每个策略先飞到某个高度再各自计时。
- 基础场景位于着陆区上方，下行、无地面接触、姿态接近竖直；起始横向速度和角速度固定，垂直速度与高度变化。
- 自定义 reset 必须一致设置机身与起落架的位置、速度和关节关系，并同步观测、奖励历史、接触状态和随机数状态。不能只改八维 observation；也不能只移动机身而留起落架在原位。实现时对照锁定版本源码。[源码入口](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py)
- 时钟使用模拟物理秒，动作保持步长固定。记录仿真步长、动作重复数及换算；不能沿用 highway 的 5 Hz 和速度阈值。
- 使用模拟器原生坐标并明确单位，不把归一化观测直接标成真实米或米每秒。
- 固定暴露 seed、地形与外生扰动。若动作路径会改变随机数消耗，使用按时间索引的扰动序列，或明确只实现初态匹配；不得仅凭相同 reset seed 声称全过程噪声相同。

**候选采样结构：** 高度 3 档 × 下降速度 3 档 × 水平偏移 2 档，共 18 个开发暴露和另一个 18 个 held-out 暴露。具体数值由物理可恢复性检查确定；不根据策略排序选格点。训练连续分布覆盖两套网格及低下降速度干预，held-out 的各变化轴取值与开发集不重合。两套 manifest 分别记录完整初态、地形 seed 和噪声 seed。

初态可恢复性先用固定参考控制器和运动检查检验；参考控制器不是最优可达性的证明。若多数场景迫使所有策略在 `t0` 制动，或还没到可能的制动阶段就撞地，则修改 pilot 初态范围并留痕。正式 held-out 后不得删掉这些困难结果。

## Reward Conditions and Agents

建议自定义且逐项记录奖励：

`r = r_common - w_D * clipped_downward_speed_squared - w_E * main_engine_usage`。

下降速度项以固定物理参考速度归一化；主发动机使用项按单位物理时间累计。`r_common` 包含所有条件相同的着陆/坠毁、位置/姿态、侧发动机和时间项。重写奖励分解时移除被替换的原有速度项和主发动机成本，避免重复计入。保留原生环境 return 仅作旁路诊断；不能用跨权重的总 return 比较优劣。

| 条件 | pilot 相对权重 `(w_D, w_E)` | 与其他策略的唯一区别及意义           |
| ---- | --------------------------- | ------------------------------------ |
| DESC | `(1.0, 0.25)`               | 更重视控制下降速度，检验较早制动假设 |
| BAL  | `(0.625, 0.625)`            | 中间权衡，检查分组或非单调关系       |
| ECON | `(0.25, 1.0)`               | 更重视减少推力使用，主要配对比较对象 |

三组使用同一 Double DQN、网络、优化器、训练步数和 seed 内场景序列，仅奖励权重变化。Double DQN 沿用项目算法基础，出处为 [van Hasselt et al., Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)。它是被解释策略的训练算法，不是与 ONSET 竞争的解释方法。

初始采用现有 MLP/DDQN 接口，检查锁定的 SB3/Gymnasium 版本兼容性。不得假设安装现有 `pilot` extra 就具备 Box2D 依赖；为新环境单独锁定依赖，不直接升级已有 highway 运行环境。

## ONSET Definition

主要事件：**第一次经运动确认的持续主发动机制动**。这是短时窗事件确认扩展，应在论文方法中明确说明；不是原 highway 三个连续 `SLOWER` 动作的原样复用。

候选定义如下，pilot 只检验语义与可测性：

1. 候选 onset `t` 必须对应下降状态下的一次主发动机点火，且尚未接触地面。
2. 前向窗口 `W = 0.30 s` 内，主发动机点火时间占比至少 `rho = 0.60`。
3. 同一窗口末，相比窗口起点，向下速度幅值下降至少 `epsilon_v`；该阈值从无推力/固定推力的物理校准确定，冻结为明确数值与单位。
4. 窗口内不得发生地面接触、坠毁或越界。最早满足全部条件的候选 `t` 为 onset，窗口末为 confirmation time；不是把窗口末当作 onset。

记录每次点火候选与未确认原因。立即制动 `tau=0` 是有效结果，不设人为最小延迟。窗口未结束就终止时不得回填一个“已确认事件”。已确认后坠毁仍保留 onset，并在终局字段报告坠毁。

辅助检测器：第一次主发动机点火、仅点火占比阈值而无运动确认。二者是本实验定义的测量对照，无独立论文出处；不把它们包装成已有 XRL baseline。

**敏感性分析：** 预先保留 `W = 0.20/0.40 s` 两个辅助窗口，主窗口固定为校准后的单一值。完整报告各窗口事件率和主要对比，不能挑显著的窗口。若 pilot 表明有效着陆普遍使用短脉冲、与该事件语义不符，应在冻结前重写事件定义或停止该方向，而非强迫策略学习长点火。

## Experimental Battery

| 实验           | 策略/样本                                                      | 目的与主要读数                                            |
| -------------- | -------------------------------------------------------------- | --------------------------------------------------------- |
| 主实验         | 三奖励条件，每组 20 个正式训练 seed；匹配 held-out 网格        | DESC–ECON 的 seed 级配对 onset 时间差，完整事件和终局统计 |
| 低下降速度干预 | 同一批固定模型与 held-out 场景，下降速度幅值乘以候选系数 `0.5` | 相同策略是否随制动压力改变 onset；其他初态与扰动匹配      |
| 检测器对照     | 同一批原始轨迹，三种检测器                                     | 检查首次点火是否误代表持续制动，无需重训                  |
| 阈值敏感性     | 同一批轨迹，冻结的辅助窗口                                     | 评估结论对事件定义的依赖                                  |
| 固定动作探针   | 无点火、单次点火、持续主推力等合成/模拟轨迹                    | 校准和验证检测器；不是用于证明奖励条件排序的训练 baseline |

不把关闭重力或禁用发动机作为必要机制对照，因为它们大幅改变任务可行性。风扰动、连续动作算法和第二套重力作为未来扩展，不进入本次最小完整实验。

## Training environment

写 pbs 在kanata运行20seeds训练

## Training, Freeze, and Inference

1. **实现与校准：** 检查 reset 一致性、动作时钟、奖励分解、故障/接触标记和固定动作探针。预先定义独立着陆成功判据，不能将任意 `terminated=True` 当成功。
2. **Pilot：** 3 个独立 pilot seed × 3 条件，起点每策略 500,000 训练步。只判断可学性、奖励尺度、事件语义和初态范围；不检验论文假设。预算和数值可在此阶段调整并记录，pilot 不进入正式推断。
3. **冻结：** 锁定依赖、训练预算、权重、归一化、主/辅助端点、manifest、所有 seed、终局判据与分析脚本。以所有条件的任务学习质量决定是否继续，不以 onset 排序或显著性决定。
4. **正式训练：** 20 seed × 3 条件，固定步数的最后 checkpoint 是主要模型；所有 seed 保留。开发评估不用于选择最有利时间差的 checkpoint。基础设施失败从同一配置重试，不替换训练不佳的 seed。
5. **封闭评估：** 校验 manifest 与模型 hash 后运行 held-out 和配对干预；仅根据技术完整性决定是否重跑，不能根据统计结果扩样或调参。
6. **汇总与入稿：** 生成数据完整性报告、seed 级表和图稿所需数据。负结果、失败率过高或大量即时响应均按预定边界解释。

每个 seed、每对条件，在两者均观察到事件的匹配暴露上计算 `median(tau_ECON - tau_DESC)`。主要效应为 20 个 seed 估计的中位数，10,000 次 seed bootstrap 的 95% CI；固定 bootstrap seed。区间完全大于 0 支持 L1。两侧 exact sign test 作为方向一致性补充，BAL 比较为辅助，不能替代主检验。

候选可估计门槛为每 seed 至少 12/18 个共同事件，且全部 20 个 seed 可估计，正式冻结前确定。门槛未满足时仍完整报告事件率和失败，不删 seed 宣称确认性时间优势。条件时间差始终解释为“双方事件均发生的场景中的差异”，即使达到门槛也不称为无条件响应优势。

时间终点与 episode 终局分开存储：事件已确认、到时限仍无事件、无事件而成功着陆、无事件而坠毁、越界、技术错误。后四类不能一律作无信息右删失；不使用把坠毁当普通删失的 Kaplan–Meier 推断。技术错误不得混入行为失败，未修复则整批标记不完整。

干预分析先报告每 seed 的事件率差和所有事件/终局转移计数，再报告共同事件中的延迟；无事件不赋予 horizon 延迟。L2 是支持性证据，不把“延迟/抑制任选其一显著”设为额外确认性成功规则。

## Deliverables and Implementation Boundary

拟建文件（目前只存在本计划）：

| 路径                                                                 | 职责                                          |
| -------------------------------------------------------------------- | --------------------------------------------- |
| `experiments/lunar_lander_braking/env.py`                            | 场景 reset、奖励分解、状态/终局记录           |
| `experiments/lunar_lander_braking/events.py`                         | 窗口与运动确认、辅助检测器                    |
| `experiments/lunar_lander_braking/run.py`                            | pilot/训练/固定模型评估入口                   |
| `experiments/lunar_lander_braking/protocol.md`、`configs/`、`grids/` | 冻结配置、物理单位、开发与 held-out manifest  |
| `scripts/aggregate_lunar_lander.py`                                  | 事件完整性、配对、seed bootstrap 和支持性分析 |
| `scripts/katana_lunar_*.pbs`                                         | 集群训练与评估                                |
| `tests/test_lunar_lander_*.py`                                       | reset、端点、终局、聚合关键验证               |

复用 DDQN 与统计思想，先检查共享汇总模块是否硬编码 highway 字段，不为复用而伪造 `front_distance` 或 `collision`。新事件提取器独立放置，不能直接改变现有 highway 检测行为。

输出保存在独立 `outputs/lunar_lander_braking_<version>/`：逐步轨迹、候选/确认事件、episode outcome、reward components、seed contrasts、配置与模型 hash。代表性轨迹按最接近总体中位效应的 seed/暴露确定，不挑最大差异。

论文目标约 0.75–1.25 页：一个小节、一张匹配高度/下降速度/点火轨迹图、一张含事件分母、终局、时间差 CI 的表。若页数紧，窗口敏感性和完整配置放补充材料或仓库。绘图后另行做可视化 QA，本计划不生成图。

## Acceptance and Resource Estimate

**技术完成：** 锁定配置可重现；所有正式模型、暴露和干预齐全；事件探针验证正确；无按结果删 seed/场景；原始数据可重算表格。

**科学判断：** L1 区间支持方向且物理轨迹、任务质量和机制干预相容，才表述“在所测第二领域支持机制敏感的时间比较”。若方向未成立，则报告未获得预期分离；若学不会或事件语义不适用，则不能作为成功跨域验证。技术完成与假设成立是两个独立标准。

预算仅供排程：pilot 9 个策略 × 0.5M 步 = 4.5M 步；若冻结预算仍为 0.5M，正式 60 个策略 = 30M 步。正式 held-out 原始+干预共 `60 × 18 × 2 = 2,160` episodes，开发评估另计。单策略峰值内存、每秒步数与训练总时长用首个 pilot 实测；当前不承诺 GPU 小时。

工程粗估 3–5 个工作日，包括环境/校准、pilot 排查、冻结与汇总接入，不含 Katana 排队、多 seed 训练及失败修复。长训练上 Katana，本地仅运行短 smoke 和必要验证。若时间不足，减少可选扰动扩展，不能把少量 pilot seed 当正式确认性证据。

## Sources

- [Gymnasium LunarLander 文档](https://gymnasium.farama.org/main/environments/box2d/lunar_lander/)：环境接口及原生任务说明。
- [Gymnasium LunarLander 源码](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py)：实施时锁定版本、核对 reset/奖励/时钟；本计划未完成该实现验证。
- [Double DQN 原论文](https://arxiv.org/abs/1509.06461)：训练算法出处。
- [当前论文](/Users/cattie/myProjects/highway_transition_timing/docs/manuscripts/acra2026/paper.tex)、[现有实现契约](/Users/cattie/myProjects/highway_transition_timing/docs/implementation.md)：ONSET 研究定位与现有运行边界。
