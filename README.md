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

Katana 没有默认的 `conda` 命令。先从登录节点申请交互计算节点：

```bash
qsub -I -l select=1:ncpus=4:mem=16gb,walltime=2:00:00
```

作业启动、主机名从 `kdm` 变为计算节点后，进入项目并创建 venv：

```bash
cd ~/projects/lunarlander_transition_timing
module avail python
PYTHON_MODULE=python/3.10.8 bash scripts/setup_katana_venv.sh
```

默认环境位于 `/srv/scratch/$USER/environments/lunar-lander`。如果 `module avail python` 显示了更新版本，可通过 `PYTHON_MODULE=python/<version>` 指定；提交 PBS 时使用相同变量。安装脚本使用 PyTorch 官方 CPU wheel，因为本实验的主要瓶颈是单进程 Box2D 仿真，并不需要 GPU。

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
  --output outputs/lunar_lander_braking_pilot_v0/train/DESC/seed_101
```

本地 smoke 可追加 `--steps 200 --allow-smoke-seed` 并使用任意测试 seed。Pilot 的 9 个模型用于可学性、尺度和事件语义检查，不进入正式推断。

Katana 上提交 pilot：

```bash
qsub scripts/katana_lunar_pilot_train.pbs
# 9 个训练任务全部成功后：
qsub scripts/katana_lunar_pilot_eval.pbs
python scripts/aggregate_lunar_lander.py \
  --config experiments/lunar_lander_braking/configs/pilot.yaml \
  --manifest experiments/lunar_lander_braking/grids/development.json \
  --input outputs/lunar_lander_braking_pilot_v0/eval \
  --output outputs/lunar_lander_braking_pilot_v0/aggregate
```

聚合器对 `phase: pilot` 永远不产生确认性 L1 结果，只输出任务质量、事件语义和完整性诊断。

若安装时选择的不是默认 Python module 或 venv 路径，请这样传给作业：

```bash
qsub -v PYTHON_MODULE=python/3.10.8,LUNAR_VENV_DIR=/srv/scratch/$USER/environments/lunar-lander scripts/katana_lunar_pilot_train.pbs
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

训练数组完成且 60 个 final checkpoint/hash 均齐全后，再提交 `qsub scripts/katana_lunar_eval.pbs`。PBS 数组各有 60 个任务，映射 `3 conditions × 20 seeds`。默认加载 `python/3.10.8` 并激活 scratch 中的 venv；可用 `PYTHON_MODULE` 和 `LUNAR_VENV_DIR` 覆盖。

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
