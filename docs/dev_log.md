# 开发日志 — LunarLander Powered-Braking Onset

> 创建时间：2026-09-10 | 最后更新：2026-09-10
> 关联实现指南：`docs/implementation.md`
> 本文件只追加，不删除。

## 项目概览

| 项目 | 内容 |
|---|---|
| 研究方向 | 非驾驶物理控制中的持续动力制动 onset |
| 实现策略 | 从头构建，复用 Gymnasium LunarLander-v3 与 Double DQN 思想 |
| 框架 | Python 3.11、PyTorch、Gymnasium/Box2D |
| Git 仓库 | `https://github.com/cattiesu2025/lunarlander_transition_timing.git` |
| 运行策略 | 本地 CPU 测试/smoke，Katana 正式训练 |

## 实现进度

| 模块 | 文件 | 状态 | 完成时间 | 备注 |
|---|---|---|---|---|
| 设计与约束 | `docs/user_requirements.md`, `docs/implementation.md` | ✅ Done | 2026-09-10 | 已通过三项设计校验 |
| 初始化与配置 | requirements/configs/grids | ✅ Done | 2026-09-10 | manifests 各 18 场景 |
| 环境与事件 | `env.py`, `events.py` | ✅ Done | 2026-09-10 | Box2D 集成与 probe 已验证 |
| DDQN 与入口 | `agent.py`, `run.py` | ✅ Done | 2026-09-10 | train/evaluate smoke 已验证 |
| 汇总与集群 | aggregate/PBS | ✅ Done | 2026-09-10 | end-to-end incomplete gate 已验证 |
| 测试与 README | tests/README | ✅ Done | 2026-09-10 | 12 tests passed |

## 开发日志

### 2026-09-10 — 配置、manifest 与协议

- **完成内容**：新增 pilot/formal 配置、互不重合的 18+18 场景 manifest、候选冻结协议和包入口。
- **遇到的问题**：`epsilon_v` 与初态范围尚未经过 Box2D probe。
- **解决方案**：明确标记为 pilot candidate，只有 probe 与任务质量检查后才允许 `freeze`。

### 2026-09-10 — 配置与校验工具

- **完成内容**：实现 YAML/manifest 验证、canonical/file SHA-256 和不可覆盖的 freeze bundle。
- **遇到的问题**：无。
- **解决方案**：无。

### 2026-09-10 — ONSET 事件模块

- **完成内容**：实现主检测器、首次点火和仅持续点火辅助检测器；保留全部候选与拒绝原因。
- **遇到的问题**：窗口的动作数与时间边界易出现 off-by-one。
- **解决方案**：窗口固定包含 `W/dt` 个动作，onset 为首动作起点，confirmation 为 `onset+W`。

### 2026-09-10 — Controlled LunarLander 环境

- **完成内容**：实现机身/起落架刚体一致 reset、独立 terrain/noise seed、物理状态日志、奖励分解及终局分类。
- **遇到的问题**：原生 reset 会先执行一个零动作 physics step，且 native reward 混合速度与主推力成本。
- **解决方案**：原生创建场景后整体重定位并重新播种噪声；自算 common shaping 和替换成本，仅保留 native return 作诊断。

### 2026-09-10 — 配置模块语法复核

- **完成内容**：修正 manifest 读取函数的括号、错误消息与 ID 集合录入拼写。
- **遇到的问题**：首次静态编译发现三个机械转写错误。
- **解决方案**：逐项修正并重新执行全模块 `py_compile`。

### 2026-09-10 — Double DQN 训练模块

- **完成内容**：实现 MLP、replay buffer、Double-DQN target、固定预算 final checkpoint 与 episode-indexed 训练场景序列。
- **遇到的问题**：三个条件必须共享 seed 内场景序列，同时不能共享策略探索 RNG。
- **解决方案**：场景由 `(seed, episode, 固定标签)` 独立派生；探索 RNG 留在 agent 内。

### 2026-09-10 — 实验 CLI

- **完成内容**：实现 probe/freeze/train/evaluate 四入口、模型与配置 hash 校验、原始和低下降速度成对评估、压缩逐步轨迹及候选记录。
- **遇到的问题**：ONSET 的 `tau=0` 必须可表达。
- **解决方案**：step 记录使用动作起点时间 `(step-1)*dt`，首动作明确为 0 秒。

### 2026-09-10 — 聚合与完整性门控

- **完成内容**：实现递归 JSONL 读取、重复/缺失/额外/技术错误检查、seed 中位差、固定 seed bootstrap、exact sign test 和干预转移统计。
- **遇到的问题**：无事件和缺失 episode 不能被赋 horizon 或自动剔除。
- **解决方案**：只对 joint events 计算时间差；完整性或 20-seed 门槛不满足时确认性结果为 null。

### 2026-09-10 — Katana PBS 编排

- **完成内容**：新增 60-task formal train/eval 数组脚本，确定性映射条件和 seed。
- **遇到的问题**：Katana 的具体 module/queue 因账户配置而异。
- **解决方案**：仅固定资源需求和数组逻辑，环境名可覆盖；README 要求提交前核对集群路径。

### 2026-09-10 — 测试与 README

- **完成内容**：新增事件、环境 reset/奖励、聚合门控测试，并记录完整运行命令。
- **遇到的问题**：本地 base 环境缺少 Box2D。
- **解决方案**：纯逻辑测试直接运行，Box2D 集成测试在创建的 Python 3.11 环境中运行。

### 2026-09-10 — 首轮静态审查修正

- **完成内容**：修正聚合脚本的 import/JSON 异常处理/list comprehension、评估 PBS 的 Conda 命令，并加入测试路径 bootstrap。
- **遇到的问题**：首次 `py_compile` 与 pytest collection 暴露机械语法错误及 `pytest` 启动方式下的包路径差异。
- **解决方案**：逐项修正；保留该日志以记录审查过程，不把失败验证标成 Done。

### 2026-09-10 — Conda 平台兼容修正

- **完成内容**：从环境定义移除只在 PyTorch channel/特定平台提供的 `cpuonly` 元包。
- **遇到的问题**：macOS arm64 的 conda-forge/defaults 无 `cpuonly`，首次环境求解失败且未创建环境。
- **解决方案**：保留 conda-forge `pytorch`；该平台自然安装 CPU/MPS 构建，Katana 仍由 PBS 在运行时检测 CUDA。

### 2026-09-10 — Freeze 与汇总审查加固

- **完成内容**：checkpoint 外部 metadata 新增 SHA-256 并在评估前验证；formal freeze 强制实际 Conda/pip lock；聚合新增全部 detector/window 的事件分母与时间摘要。
- **遇到的问题**：候选 freeze 只复制声明式依赖且聚合最初只展开 primary endpoint。
- **解决方案**：formal CLI 缺 lock 即拒绝执行；保留 nested detector 原始值并生成 detector comparison。

### 2026-09-10 — 物理校准覆盖增强

- **完成内容**：probe 默认覆盖完整开发 manifest，并输出无推力/单脉冲/持续推力的首窗速度变化范围和分离中点。
- **遇到的问题**：初版 probe 只保留一个场景的最终速度，不足以冻结运动阈值。
- **解决方案**：逐场景记录 0.30 s 首窗 reduction；单场景参数仅保留作快速调试。

### 2026-09-10 — 物理阈值校准与可恢复性入口

- **完成内容**：18 场景固定动作探针显示无推力/单脉冲与持续推力清晰分离；将候选 `epsilon_v` 校准为 0.40，并新增参考控制器 recoverability 命令。
- **遇到的问题**：原 0.08 仅是计划占位值，且尚无开发/held-out 网格可恢复性批量入口。
- **解决方案**：使用非负下界 0 与最弱持续推力 0.80 的中点；筛查只看任务可行性，不看条件排序。

### 2026-09-10 — 冻结输入与技术错误门控

- **完成内容**：formal train/evaluate 强制校验 freeze manifest；评估逐 episode 记录 technical error 并让任务非零退出；轨迹补充动作前位置/速度。
- **遇到的问题**：仅输出哈希而不校验，无法阻止冻结后配置或 held-out manifest 漂移。
- **解决方案**：在任何 formal physics step 前验证文件 SHA-256；checkpoint 继续使用外部 metadata hash 校验。

### 2026-09-10 — 网格可恢复性结果

- **完成内容**：参考控制器在 development 18/18 与 held-out 18/18 场景均成功着陆，每场景均产生非即时 primary onset。
- **遇到的问题**：无。
- **解决方案**：结果写入独立 JSONL/summary，明确不作为最优可达性证明。

### 2026-09-10 — Pilot 集群入口与推断隔离

- **完成内容**：新增 9-task pilot train/eval PBS；聚合器根据 `experiment.phase` 禁止 pilot 生成确认性 L1 结果。
- **遇到的问题**：共用聚合器若仅看完整性，完整的 3-seed pilot 可能被误读为确认性分析。
- **解决方案**：只有 `phase: formal` 才计算 bootstrap/sign test；pilot 仍输出全部诊断统计。

### 2026-09-10 — Pilot 推断门控回归测试

- **完成内容**：新增“完整 pilot 仍不得产生确认性结果”的单元测试。
- **遇到的问题**：无。
- **解决方案**：无。

### 2026-09-10 — 最终代码审查

- **完成内容**：Python 全量编译、4 个 PBS shell 语法检查、9 项 pytest、18 场景物理校准、两个 18 场景可恢复性筛查、100-step train、checkpoint/hash evaluate、freeze bundle 和 aggregate 全链路通过。
- **遇到的问题**：本机没有 `qsub`，无法直接提交 Katana 长训练；formal protocol 仍须等待 9 个 pilot 模型的任务质量审查。
- **解决方案**：交付 pilot/formal PBS 数组和严格 freeze gate；不把 smoke 或参考控制器结果冒充策略训练结果。

### 2026-09-10 — 实现契约同步与 freeze 回归测试

- **完成内容**：同步 recoverability、pilot PBS、formal input/hash 门控到 implementation contract；新增 manifest 轴不重合、freeze 漂移检测和不可覆盖测试。
- **遇到的问题**：实现审查发现契约仍保留初版四命令/两 PBS 描述。
- **解决方案**：先更新实现契约，再加入三项回归测试并执行全套验证。

### 2026-09-10 — 最终端到端复验

- **完成内容**：在锁定的 Python 3.11 环境重跑 12 项测试、pip dependency check、全量 py_compile、4 个 PBS bash 语法检查，以及 train→hash-verified evaluate→pilot aggregate smoke。
- **遇到的问题**：无；pip 仅报告用户 cache 不可写并自动禁用，不影响环境依赖完整性。
- **解决方案**：无。最终 smoke 正确标记 `analysis_phase=pilot` 且确认性结果为 null。

### 2026-09-10 — Git 发布准备

- **完成内容**：按用户要求切换为整项目 GitHub 发布；确认全局身份为 `cattiesu2025 <cattiesu2025@gmail.com>`，输出、模型、cache 和字节码均由 `.gitignore` 排除。
- **遇到的问题**：无超过 50 MB 的待跟踪文件。
- **解决方案**：初始化 main 分支、审查 staged 清单后提交并推送指定远端。

## 待执行的外部作业

- [ ] 在 Katana 提交 9-task pilot train/eval，并检查三条件任务质量与事件语义。
- [ ] pilot 通过后从 Katana 环境导出 lock、正式 freeze，再提交 60-task formal train/eval。
- [ ] 聚合 2,160 个正式 episodes；在完整性和 12/18 门槛均通过时才解释 L1。

### 2026-09-10 — 编码前约束与实现契约

- 完成内容：记录用户确认的默认环境；将研究计划转换为逐文件、函数级实现契约。
- 遇到的问题：项目只有实验计划，缺少阶段 E 所需的实现契约。
- 解决方案：先完成从头构建的实现设计，并通过实验覆盖、逻辑一致性和完整性检查。

### 2026-09-10 — 初始化、配置与 manifests

- 完成内容：添加环境定义、依赖清单、包入口、pilot/formal YAML、各 18 个场景的 development/held-out manifest，以及候选冻结协议。
- 遇到的问题：本机 base 环境是 Python 3.13 且缺少 Box2D，不适合作为锁定运行环境。
- 解决方案：定义独立 Python 3.11 `lunar-lander` 环境；正式配置仍标记为必须通过 pilot freeze gate 后才可冻结。

## 运行说明

### 环境准备

```bash
conda env create -f environment.yml
conda activate lunar-lander
```

安装 Python 3.11、PyTorch CPU、Gymnasium Box2D 和分析/测试依赖。正式集群环境可由 Katana module 替代 Conda 中的 PyTorch。

### 快速验证与物理探针

```bash
pytest -q
python -m experiments.lunar_lander_braking.run probe --output outputs/probe.jsonl
python -m experiments.lunar_lander_braking.run recoverability --manifest experiments/lunar_lander_braking/grids/development.json --output outputs/development_recoverability.jsonl
```

- `pytest` 验证事件边界、受控 reset、奖励唯一差异和确认性门控。
- `probe` 输出三条固定动作探针到 JSONL；用于冻结前校准，不检验 L1。
- `recoverability` 输出参考控制器的逐场景终局和事件结果；不证明最优可达性。

### 单模型训练

```bash
python -m experiments.lunar_lander_braking.run train --config experiments/lunar_lander_braking/configs/pilot.yaml --condition DESC --seed 101 --output outputs/lunar_lander_braking_pilot_v0/train/DESC/seed_101
```

- `--condition` 选择唯一奖励差异；`--seed` 必须在配置中；`--steps` 只用于 smoke。
- 输出 final checkpoint、metadata 与逐 episode training metrics。

### Katana pilot

```bash
qsub scripts/katana_lunar_pilot_train.pbs
# 9 个训练任务全部成功后：qsub scripts/katana_lunar_pilot_eval.pbs
```

- 两个数组各 9 个任务，映射 3 conditions × 3 pilot seeds；pilot 聚合器禁止确认性推断。

### 冻结、Katana 正式训练和评估

```bash
mkdir -p outputs/lunar_lander_braking_v1/locks
conda list --explicit > outputs/lunar_lander_braking_v1/locks/conda-explicit.txt
python -m pip freeze > outputs/lunar_lander_braking_v1/locks/pip-freeze.txt
python -m experiments.lunar_lander_braking.run freeze --config experiments/lunar_lander_braking/configs/formal.yaml --development-manifest experiments/lunar_lander_braking/grids/development.json --held-out-manifest experiments/lunar_lander_braking/grids/held_out.json --conda-lock outputs/lunar_lander_braking_v1/locks/conda-explicit.txt --pip-lock outputs/lunar_lander_braking_v1/locks/pip-freeze.txt --output outputs/lunar_lander_braking_v1/frozen --confirm-calibrated
qsub scripts/katana_lunar_train.pbs
# 训练数组全部成功后：qsub scripts/katana_lunar_eval.pbs
```

- `--confirm-calibrated` 和两个环境 lock 是正式 freeze 的显式门；输出不可覆盖的配置、manifest、protocol 和 checksums。
- 两个 PBS 数组分别产生 60 个 final checkpoints 和 2,160 个封闭评估 episodes。

### 聚合

```bash
python scripts/aggregate_lunar_lander.py --config experiments/lunar_lander_braking/configs/formal.yaml --manifest experiments/lunar_lander_braking/grids/held_out.json --input outputs/lunar_lander_braking_v1/eval --output outputs/lunar_lander_braking_v1/aggregate
```

- 输出完整性报告 `summary.json` 和 seed 级 `seed_contrasts.csv`。
