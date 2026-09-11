# LunarLander Powered-Braking Onset

本项目实现 controlled-reset、reward-modified LunarLander 实验，用三个独立训练的 Double DQN 奖励条件检验持续动力制动 onset。代码区分 pilot 与 formal freeze，保留逐步轨迹、所有点火候选、行为终局与技术错误，并只在完整性和共同事件门槛满足后计算确认性 seed-level 结果。

## Environment

```bash
conda env create -f environment.yml
conda activate lunar-lander
```

`requirements.txt` 不固定版本。实际 formal freeze 前先导出环境锁文件，且不得在 freeze 后升级 Gymnasium/Box2D：

```bash
mkdir -p outputs/lunar_lander_braking_v1/locks
conda list -n lunar-lander --explicit > outputs/lunar_lander_braking_v1/locks/conda-explicit.txt
conda run -n lunar-lander python -m pip freeze > outputs/lunar_lander_braking_v1/locks/pip-freeze.txt
```

### Katana（推荐）

Katana 没有默认的 `conda` 命令。环境安装也通过普通 PBS batch job 完成，不需要等待交互节点：

```bash
cd ~/projects/lunarlander_transition_timing
qsub scripts/katana_lunar_setup.pbs
```

默认加载已在账户上确认可用的 `python/3.11.3`，环境位于 `/srv/scratch/$USER/environments/lunar-lander`。setup job 会安装依赖、执行 `pip check` 并验证 Box2D、Gymnasium、Stable-Baselines3 和 PyTorch。可用下面的命令检查 setup 日志：

```bash
qstat -u "$USER"
tail -n 50 lunar_setup.o*
```

安装脚本使用 PyTorch 官方 CPU wheel，因为本实验的主要瓶颈是单进程 Box2D 仿真，并不需要 GPU。如果需要覆盖 module 或 venv 路径，setup 和后续作业必须传入相同的 `PYTHON_MODULE`、`LUNAR_VENV_DIR`。

环境安装完成后先做快速验证：

```bash
source /srv/scratch/$USER/environments/lunar-lander/bin/activate
pytest -q
```

正式 freeze 前，在同一个 Katana venv 中记录环境：

```bash
mkdir -p outputs/lunar_lander_braking_v1/locks
python --version > outputs/lunar_lander_braking_v1/locks/python-version.txt
python -m pip freeze > outputs/lunar_lander_braking_v1/locks/pip-freeze.txt
```

## Fast validation

```bash
pytest -q
python -m experiments.lunar_lander_braking.run probe \
  --output outputs/probe.jsonl
python -m experiments.lunar_lander_braking.run recoverability \
  --manifest experiments/lunar_lander_braking/grids/development.json \
  --output outputs/development_recoverability.jsonl
```

probe 默认在开发集 18 个场景上运行无点火、单次点火和持续主推力轨迹，输出逐场景 JSONL 和 `probe.summary.json` 的首窗速度变化范围，用于检查 `epsilon_v`、持续窗语义和候选拒绝原因。快速单场景检查可加 `--scenario-index 8`。`recoverability` 使用参考控制器筛查初态可恢复性；它不是最优可达性证明。当前参数仍是 pilot candidate。

## Pilot

单个条件/seed：

```bash
python -m experiments.lunar_lander_braking.run train \
  --config experiments/lunar_lander_braking/configs/pilot.yaml \
  --condition DESC --seed 101 \
  --output outputs/lunar_lander_braking_pilot_v3/train/DESC/seed_101
```

本地 smoke 可追加 `--steps 200 --allow-smoke-seed` 并使用任意测试 seed。Pilot 的 9 个模型用于可学性、尺度和事件语义检查，不进入正式推断。

Katana 上首次提交 pilot，可一次性排队 setup 和 training；training 仅在 setup 成功后启动：

```bash
bash scripts/submit_katana_pilot.sh
# 9 个训练任务全部成功后：
qsub scripts/katana_lunar_pilot_eval.pbs
python scripts/aggregate_lunar_lander.py \
  --config experiments/lunar_lander_braking/configs/pilot.yaml \
  --manifest experiments/lunar_lander_braking/grids/development.json \
  --input outputs/lunar_lander_braking_pilot_v3/eval \
  --output outputs/lunar_lander_braking_pilot_v3/aggregate
```

聚合器对 `phase: pilot` 永远不产生确认性 L1 结果，只输出任务质量、事件语义和完整性诊断。

Pilot v1 在所有条件中加入相同的 `time_scale=5.0 reward-units/s`（每步 `-0.1`，1000-step horizon 累计 `-100`），修复 pilot v0 中 timeout 没有共同时间代价的问题。Pilot v2 进一步加入共享的 `settling_actuation_scale=5.0`：仅当动作开始时双腿已接触、策略仍执行非零动作时额外扣 `-0.1`，用于抑制落地后的持续作动，不影响接触前 ONSET。Pilot v3 保留 v2 奖励，将训练器替换为继承 Stable-Baselines3 `DQN` 并覆盖 `train()` 的 Double DQN；online network 选择 next action，target network 评估该 action。各版本使用独立输出目录，互不覆盖。

### ECON 1M training-budget diagnostic

Pilot v4 仅将 ECON 的训练预算从 500k 增加到 1M environment steps，其余配置与 v3 完全一致。它用于判断 ECON 是否因 500k 过早停止而未收敛，不能与 v3 DESC/BAL 直接组成 ONSET 推断。

```bash
qsub scripts/katana_lunar_econ_1m_train.pbs
# 3 个 train array tasks 全部成功后：
qsub scripts/katana_lunar_econ_1m_eval.pbs
```

输出位于 `outputs/lunar_lander_braking_pilot_v4_econ_1m/`。若三个 ECON seed 在 1M 都通过任务质量门槛，后续才用同一 1M 预算从头重训三个条件。

### V5 height × checkpoint diagnostic

V5 只训练 ECON，并同时运行 current height（训练 4.8–7.2，评估 5/6/7）与 high height（训练 7.8–12.2，评估 8/10/12）两个实验臂。每次训练自动保存 600k、700k、800k、900k、1M checkpoints，因此只需要 6 个训练任务，而不是为五个预算分别训练。

```bash
qsub scripts/katana_lunar_v5_train.pbs
# 6 个 train array tasks 全部成功后：
qsub scripts/katana_lunar_v5_eval.pbs
```

Evaluation array 共 30 个任务，覆盖 `2 arms × 3 seeds × 5 checkpoints`，输出位于 `outputs/lunar_lander_braking_pilot_v5_height_budget/`。选择规则只使用 original-scenario landing quality：每个 seed 至少 15/18 landed，并在下一个 checkpoint 保持门槛；不得使用 ONSET 选择高度或预算。V5 不与旧版本 DESC/BAL 组合推断。

### V6 high-height latest-eligible pilot

V6 在 high-height 训练分布 `7.8–12.2` 下从头训练 DESC/BAL/ECON × 3 seeds。每个模型从 1M checkpoint 向前选择第一个通过 gate 的 checkpoint，和 highway 项目的 latest-eligible 结构一致。Gate 仅检查 original/low-speed 各自 `landed >= 15/18`、`primary events >= 15/18`、完整性与 hashes；选择代码不读取 ONSET 时间。

```bash
qsub scripts/katana_lunar_v6_train.pbs
# 9 个 train tasks 全部成功后：
qsub scripts/katana_lunar_v6_dev_eval.pbs
# 45 个 development evaluation tasks 全部成功后：
qsub scripts/katana_lunar_v6_select.pbs
```

输出位于 `outputs/lunar_lander_braking_pilot_v6_high_latest/`。`selection/model_selection.csv` 保留全部 checkpoint gate audit，`selection/selected_checkpoints.json` 记录九个模型的选择。`held_out_high.json` 与 development 的固定轴和 seeds 均不重合；v6 pilot 不运行 held-out。

若安装时选择的不是默认 Python module 或 venv 路径，请这样传给作业：

```bash
qsub -v PYTHON_MODULE=python/3.11.3,LUNAR_VENV_DIR=/srv/scratch/$USER/environments/lunar-lander scripts/katana_lunar_pilot_train.pbs
```

## Freeze and formal run

校准和 pilot 审查完成后才可明确确认 freeze：

```bash
python -m experiments.lunar_lander_braking.run freeze \
  --config experiments/lunar_lander_braking/configs/formal.yaml \
  --development-manifest experiments/lunar_lander_braking/grids/development.json \
  --held-out-manifest experiments/lunar_lander_braking/grids/held_out.json \
  --output outputs/lunar_lander_braking_v1/frozen \
  --python-lock outputs/lunar_lander_braking_v1/locks/python-version.txt \
  --pip-lock outputs/lunar_lander_braking_v1/locks/pip-freeze.txt \
  --confirm-calibrated
qsub scripts/katana_lunar_train.pbs
```

训练数组完成且 60 个 final checkpoint/hash 均齐全后，再提交 `qsub scripts/katana_lunar_eval.pbs`。PBS 数组各有 60 个任务，映射 `3 conditions × 20 seeds`。默认加载 `python/3.11.3` 并激活 scratch 中的 venv；可用 `PYTHON_MODULE` 和 `LUNAR_VENV_DIR` 覆盖。

Formal 的 train/evaluate 命令必须提供 `--freeze-manifest`；PBS 已指向冻结目录。任何配置或 held-out manifest 哈希不匹配都会在运行前失败。评估中的 episode 级异常会写成 `technical_error` 后让 PBS task 返回失败，便于同配置重试。

## Aggregate

```bash
python scripts/aggregate_lunar_lander.py \
  --config experiments/lunar_lander_braking/configs/formal.yaml \
  --manifest experiments/lunar_lander_braking/grids/held_out.json \
  --input outputs/lunar_lander_braking_v1/eval \
  --output outputs/lunar_lander_braking_v1/aggregate
```

输出 `summary.json` 与 `seed_contrasts.csv`。缺 episode、重复键、unexpected row、技术错误或任一 seed 少于 12/18 个 DESC/ECON 共同事件时，确认性 bootstrap 和 sign test 不会计算。
