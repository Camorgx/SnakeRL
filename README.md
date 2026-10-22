# SnakeRL — 从零学习，直到填满世界

一个可以训练、观看、回放和比较贪吃蛇智能体的研究原型。使用 **Gymnasium + PyTorch CNN + Stable-Baselines3 PPO**，通过 Pygame 观察真实策略，通过 TensorBoard 和 CSV 追踪训练。

神经网络决策只读取观察张量，输出左转、直行、右转的概率。推理时没有动作屏蔽、寻路、专家调用或碰撞兜底。Random、贪心 BFS、哈密顿回路是明确标注的传统算法基线；专家只参与模仿学习的数据采集。

## 纯 PPO 研究迭代（2026-10-14）

已增加长视野 PPO、学习率/熵衰减、KL 限制、独立 actor/critic CNN、计时与显式身体观察，以及仅从 PPO 自身经验构造的收尾课程。`--pure-ppo` 会沿初始化模型链检查来源，拒绝专家衍生和未知来源的权重。课程中的长蛇开局只参与训练，标准评估始终从长度 3 开始。最新真实结果见 [纯 PPO 迭代报告](reports/ppo_iteration_2026-10-14.md)。

观看已取得稳定全覆盖的小棋盘纯 PPO：

```powershell
uv run snakerl play --policy neural --checkpoint runs/ppo_iteration/curriculum4/checkpoints/best.zip --size 4 --fps 30
```

分析成功轨迹是否始终遵循一个固定的位置回路：

```powershell
uv run snakerl analyze --checkpoint runs/ppo_iteration/curriculum4/checkpoints/best.zip --size 4 --episodes 100 --seed-start 70000 --out runs/my_path_analysis
```

同一局中，一个位置若出现多个不同后继位置，就不能用一个固定位置回路解释该局。仅仅不匹配某个哈密顿基线，不能证明新策略不是另一条固定回路。分析只在游戏结束后进行，不参与网络决策。

观察版本 1 保留原来的 9 通道；版本 2 加入未进食计时和总步数；版本 3 再加入二值蛇身占用和尾巴位置，总计 13 通道。这些都来自实际棋盘，不提供推荐动作。扩大观察时，新输入权重初始化为 0，兼容已有纯 PPO 权重；播放器自动识别模型的观察版本。

`coverage` 奖励配置：每步 −0.003，吃食物加 `1+2×覆盖率²`，全覆盖额外 +20，碰撞扣 `1+4×覆盖率²`，超时额外 −1。对同样从长度 3 开始的成功局，未折扣的进食/满盘总奖励固定，步数越少收益越高；有限训练不构成全局最短路径证明。标准比赛仍用 `legacy` 奖励记录，并比较成功率和步数。

CSV 保存每局是否使用课程开局，曲线主均值只统计正常开局；TensorBoard 的 `snake/standard_success` 和 `snake/curriculum_success` 分开记录。旧日志中的 `snake/success` 为全部训练局的混合指标，`eval/success_rate` 才是标准初始状态的验证成功率。

可复用训练设置：

```powershell
uv run snakerl train --size 8 --steps 1048576 --envs 32 --n-steps 256 --batch-size 1024 --gamma 0.9995 --gae-lambda 0.99 --target-kl 0.01 --learning-rate 0.00003 --final-learning-rate 0.000005 --entropy 0.004 --final-entropy 0.0002 --observation-version 3 --reward-profile coverage --distance-reward 0 --separate-encoders --self-curriculum 0.5 --curriculum-min-coverage 0.85 --pure-ppo --out runs/my_pure_ppo
uv run python scripts/retrain.py --config runs/ppo_iteration/curriculum4/config.json --out runs/my_reproduced4
```

上述第一条命令从随机权重启动；复现具体阶段时使用其 `config.json`，必要时通过 `--init-checkpoint` 替换本地初始化模型路径。`evaluate --stochastic` 可以独立测试 PPO 动作采样，使用按局固定的随机数源，并与默认 argmax 结果分开记录。

## 使用 uv 启动

```powershell
uv sync --frozen
uv run snakerl play --policy hamiltonian --size 8
```

Python 由 `.python-version` 选择 3.13；依赖由 `pyproject.toml` 声明并锁在 `uv.lock`，安装进项目 `.venv`。默认使用 PyTorch CPU wheel，适合在普通 Windows 电脑上复现实验。无需手动激活环境或调用 pip。项目内缓存位于 `.uv-cache`；新机器首次同步需要联网。

当前工作区已保存训练模型，可直接观看神经网络：

```powershell
uv run snakerl play --policy neural --checkpoint runs/imitation8/final.zip --size 8 --fps 60 --run-dir runs/imitation8
```

`runs/` 是本地实验产物，不加入 Git；新克隆需要训练来生成模型。真实实验结论与轻量记录见 [首次实验报告](reports/first_experiment.md)。

## 完整实验闭环

一条命令执行从零 PPO、模仿学习 + PPO、传统算法和同种子比赛，并导出模型、曲线和回放：

```powershell
uv run snakerl experiment --size 8 --steps 131072 --bc-samples 30000 --bc-epochs 30 --finetune-steps 8192 --episodes 100 --out runs/my_experiment8
```

BC 默认学习率为 `1e-3`，PPO 微调默认为 `1e-5`。每次实验使用一个新的输出目录，避免覆盖已有记录。训练步数是多个环境的**总步数**，SB3 按完整 rollout 更新，实际步数可能向上取整不到一个 rollout。

可以分别运行：

```powershell
# 从随机权重开始；渲染不参与训练
uv run snakerl train --size 8 --steps 131072 --envs 8 --seed 0 --out runs/my_ppo8

# 哈密顿专家示范 → 监督模仿 → PPO 微调
uv run snakerl train --mode imitation_ppo --size 8 --steps 8192 --bc-samples 30000 --bc-epochs 30 --learning-rate 0.00001 --entropy 0 --out runs/my_imitation8

# 100 或 1000 局，确定性神经网络推理，独立测试种子
uv run snakerl evaluate --policy neural --checkpoint runs/my_imitation8/final.zip --size 8 --episodes 1000 --seed-start 10000 --out runs/my_test1000

# 所有策略都使用完全相同的测试种子集合
uv run snakerl compare --size 8 --episodes 100 --model ppo=runs/my_ppo8/final.zip --model imitation_ppo=runs/my_imitation8/final.zip --out runs/my_race
```

每个策略的输出包含 `evaluation.json`、逐局 `episodes.csv`，以及存在相应结果时的 `replay_success.json` / `replay_failure.json`。指标包括全覆盖成功率、95% Wilson 区间、平均/最小/最大覆盖率、成功局平均步数和失败原因。相同种子固定各环境的随机数源；策略改变可用格子后，食物坐标不必相同。

## 观看训练与失败回放

```powershell
uv run snakerl play --policy neural --checkpoint runs/my_ppo8/final.zip --size 8 --run-dir runs/my_ppo8
uv run snakerl play --replay runs/my_race/ppo/replay_failure.json
uv run tensorboard --logdir runs --host localhost
```

播放器显示棋盘覆盖率、食物数、步数、动作概率和真实训练曲线。它执行推理；训练在另一个进程运行。附加 `--run-dir` 后，按 R / N 开始新一局时加载该训练目录发布的新模型，当前局保持同一组权重。保存的神经网络回放包含所用 checkpoint 和逐步概率。

| 操作 | 快捷键 |
|---|---|
| 暂停 / 继续 | Space |
| 单步前进 | → |
| 回放退一步 | ← |
| 加速 / 减速 | ↑ / ↓ 或 + / − |
| 同种子重开 | R |
| 下一种子 | N |
| 保存当前回放 | S |
| 退出 | Esc |

回放支持点击进度条跳到任意一步。传统算法的动作条展示已选动作，不将其标成神经网络概率。图表可独立导出：

```powershell
uv run snakerl plot --run-dir runs/my_ppo8
uv run snakerl plot --comparison runs/my_race/comparison.json --out runs/my_race/chart.png
```

## 环境与奖励

棋盘默认 8×8，也支持其他大小及环境 API 中的矩形。初始长度 3（小棋盘会缩短到宽度），蛇在顶行朝东；初始蛇身固定，食物按种子均匀生成在空格中。此设置让三种路线公平比较，并保证哈密顿基线从有效的回路排列开始。**当前实验没有证明随机初始蛇身或障碍棋盘上的泛化。**

观察为 `(9, height, width)` 的 `[0,1]` 浮点张量：蛇头、含顺序信息的蛇身、食物、四个朝向通道、x / y 坐标。CNN 融合全局空间特征和蛇头附近特征，再由 actor / critic 网络输出动作分布与价值。坐标为观察的一部分，未编码任何路径规则。

| 事件 | 奖励 / 行为 |
|---|---|
| 每步 | −0.001 |
| 吃食物 | 额外 +1，长度 +1 |
| 全覆盖 | 额外 +10，`terminated=True` |
| 撞墙或撞身体 | −1，`terminated=True` |
| 非进食移动 | `distance_reward × 曼哈顿距离减少量 / (width+height−2)` |
| 超时 | 额外 −0.25，`truncated=True` |

默认 `distance_reward=0.02`，可用 `--distance-reward 0` 消融距离塑形。移动到正在让位的尾巴合法。连续未吃食物的上限为 `4×面积`，总步数上限为 `4×面积²`；成功优先于时间限制。训练和标准比赛默认使用同一环境设置，报告保存评估所用设置。

模仿数据按整局种子划分训练/验证集。数据集记录专家来源和种子；大棋盘按局进行有界采样，避免保存过多状态。动作预测准确率与整局成功率分别记录。`checkpoints/bc.zip` 保存 PPO 微调前的网络，`final.zip` 保存训练结束时的网络，`checkpoints/best.zip` 按内部验证集选择；测试集不参与选择。

## 课程学习与多进程

```powershell
uv run snakerl curriculum --sizes 4 6 8 --steps 262144 --out runs/my_curriculum
uv run snakerl train --size 8 --steps 131072 --vec-env subproc --envs 8 --out runs/my_parallel
uv run snakerl train --size 8 --steps 524288 --init-checkpoint runs/my_ppo8/final.zip --out runs/my_extended
```

默认 `DummyVecEnv` 在单进程批量收集多个小环境；`subproc` 使用 Windows 兼容的 spawn 真正并行。CPU 默认限制 PyTorch 为 2 线程，避免小卷积中过度调度。CNN 输出维度与棋盘大小无关，因此课程学习能迁移权重。迁移和 `--init-checkpoint` **重新建立优化器和步数计数器**，不是精确恢复中断的训练。

4→6 的权重迁移与 Windows 双工作进程已做运行验证。6×6、12×12、20×20 都可继续训练，但本次可靠通关结论只覆盖 8×8 示范学习路线。纯 PPO 的长期规划、缩短通关步数和 DQN 对比仍是后续研究方向。

## 验证与项目结构

```powershell
uv run python -m pytest -q
uv lock --check
```

测试覆盖 Gymnasium 接口、食物和尾巴逻辑、完整终止、超时截断、矩形回路、同种子比赛、完整回放重演、神经推理独立性、模型保存和训练权重更新。Pygame 另有 SDL dummy 的实时、回放及键盘交互烟雾验证。

| 文件 | 用途 |
|---|---|
| `src/snakerl/env.py` | Gymnasium 棋盘与完整观察 |
| `src/snakerl/baselines.py` | Random、BFS、哈密顿算法 |
| `src/snakerl/model.py` | PyTorch CNN 与纯神经推理 |
| `src/snakerl/training.py` | PPO、模仿、课程权重迁移、日志 |
| `src/snakerl/evaluation.py` | 同种子评估、比赛、回放 |
| `src/snakerl/viewer.py` | Pygame 播放与训练监控 |
| `src/snakerl/reporting.py` | 真实数据图表导出 |

接口设计参考 [Gymnasium 自定义环境](https://gymnasium.farama.org/introduction/create_custom_env/)、[SB3 自定义策略](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)；uv 配置参考 [官方配置文档](https://docs.astral.sh/uv/configuration/files/)。
