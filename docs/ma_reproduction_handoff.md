# 交接：在 HPC 上复现多智能体 PCGRL 论文实验并生成 GIF

> 写给在 HPC 上接手这项工作的 Claude（以及 Zehua）。写于 2026-10-02，原本在一台 2×RTX 4090 的机器上推进，
> 那台机器的 GPU 被别的任务占着，所以训练改到 HPC 上跑。

## 目标

复现论文 *Video Game Level Design as a Multi-Agent Reinforcement Learning Problem*（Earle, Jiang, Vinitsky,
Togelius, AIIDE 2025）里**有代表性、效果最好**的设置，外加两组新的对比实验，然后**渲染 GIF 用于 presentation**。
最终产出是一个文件夹，里面是挑选过、命名清楚的 GIF（及 `best.png`）。数值结果能对上论文就够了，不需要多 seed 统计。

## 论文要点（选这些实验的依据）

- 表 1 / 图 3：binary（迷宫）domain，turtle 表示，3×3 局部观察，每个 episode 1 个 board scan
  （16×16 地图上为 512 个环境步），训练 9e8 步。智能体数量 1 → 2 → 3 时，分布内和分布外的结果都逐步变好。
  - 平均 episode reward（训练用的 16×16 固定形状地图）：1 个智能体 46.30，3 个智能体 62.57。
    在 32×32 固定形状地图上：156.10 对 181.81。
- 表 4：dungeon domain，同样是 3 个智能体最好（16×16 固定形状：66.43 对 99.91）。
- 表 6：3×3 局部观察最好。
- 论文配图就是我们想要的画面：图 2（1 个智能体对 3 个智能体的生成过程）、图 4（3 个智能体在没见过的 32×32
  地图上协作）、图 5（dungeon）。
- 注意：论文正文说用的是 RNN，但仓库里对应的 sweep（`ma_board_scans_binary_aiide`、
  `ma_n_agents_dungeon_conv2_obs_3`）用的是 `model: conv2`。我们按 sweep 文件用 conv2。
  另外 `train_ma.py` 用 `model=rnn` 时本来就会报错（见下文“已知问题”）。

## 要跑的实验（共 7 个训练任务，每个配置 1 个 seed）

所有任务共用：turtle、conv2、3×3 观察（`obs_size_hid_dims: 3`）、`max_board_scans: 1.0`、`n_envs: 400`、
`total_timesteps: 9e8`、`seed: 0`。sweep 配置文件已写好，提交后它们都会被 commit。

| sweep 名（`conf/sweeps/<name>.yaml`） | domain | 训练地图 | 智能体数 | 说明 |
|---|---|---|---|---|
| `ma_repro_binary` | binary | 16×16 | 1, 3, 20 | 1 和 3 复现表 1 和图 2；20 是新增的实验 |
| `ma_repro_dungeon` | dungeon | 16×16 | 1, 3 | 复现表 4 和图 5 |
| `ma_repro_binary_w32` | binary | 32×32 | 1, 3 | 新增：在 32×32 地图上训练，比较 1 个和 3 个智能体 |

渲染和评估时，每个模型会在多种地图宽度（16×16 训练的是 8/16/24/32，32×32 训练的是 16/24/32）上测试，
每种宽度分固定形状和随机形状两种。

关于新增的两个实验：
- **20 个智能体**：在 turtle 表示下，episode 长度只看地图大小（`宽 × 高 × 2 × board_scans`），跟智能体数量无关。
  所以 20 个智能体同样是 512 个环境步，但总编辑次数是单个智能体的 20 倍。每个环境的 actor 数是 3 个智能体时的约 7 倍，
  训练会慢很多。
- **32×32 训练**：每个 episode 有 2048 步，路径计算也更贵。9e8 步可能不够收敛。看一下 `progress.csv` 里的曲线，
  如果还在上升，就把 yaml 里的 `total_timesteps` 调大再重新提交（会从 checkpoint 续训）。

## 当前仓库状态

- 依赖改用 uv 管理（`pyproject.toml` + `uv.lock`，Python 3.12，JAX 0.6.2）。`gymnax` 锁定在 `<1.0`：
  gymnax 1.0 会导致 `train.py` 在 `is_terminated` 处抛 `NotImplementedError`。
- 在原来那台机器的 CPU 上（`JAX_PLATFORMS=cpu`）已经验证，以下都能跑通几个 update：
  `train.py`（含 GIF 渲染）；用 `train_ma.py` 训练 binary 3 个智能体、binary 20 个智能体、dungeon 3 个智能体，
  以及 binary 在 32×32 上 3 个智能体（全部为 conv2，3×3 观察）；三个 sweep 都能正确展开。
  **还没有在 GPU 上跑过。**
- 已知问题：`train_ma.py` 用 `model=rnn` 时（`MultiAgentConfig` 的默认值原本是 rnn，现已改为 conv2），在 GAE 的 scan 那一步会报
  `float32[N]` 和 `float32[1,N]` 形状不匹配。在原来的 conda 环境里也会出现，所以是代码原本就有的问题，不影响本计划
  （本计划全部用 conv2）。

## HPC 上的步骤

### 0. 环境准备

```bash
cd /scratch/$USER && git clone git@github.com:JiangZehua/pcgrl-jax.git   # 或者在已有的 clone 里 git pull
cd pcgrl-jax
curl -LsSf https://astral.sh/uv/install.sh | sh        # 如果还没装 uv
export UV_CACHE_DIR=/scratch/$USER/.cache/uv            # home 目录配额小，缓存放到 scratch
uv sync --extra cuda12
echo "SLURM_ACCOUNT=<你的账户，例如 pr_174_tandon_advanced>" > .env   # sweep.py 从这里读取账户
```

W&B：要么在 `.venv` 里执行一次 `wandb login`，要么给下面所有命令加上 `wandb_mode=offline`。
默认的 project 是 `smearle_pcgrl_mappo`（在 `conf/config.py` 里定义）。

### 1. 在 GPU 上做冒烟测试并测吞吐量

先申请一个交互式 GPU 节点（参考 `scripts/hpc/srun_gpu.sh`），然后：

```bash
C="model=conv2 arf_size=3 vrf_size=3 obs_size_hid_dims=3 max_board_scans=1.0 n_envs=400 wandb_mode=disabled save_dir=saves_smoke overwrite=True"
uv run python train_ma.py $C problem=binary n_agents=3  hidden_dims=[477,512] total_timesteps=20000000
uv run python train_ma.py $C problem=binary n_agents=20 hidden_dims=[477,512] total_timesteps=20000000
uv run python train_ma.py $C problem=binary n_agents=3  hidden_dims=[477,512] map_width=32 total_timesteps=20000000
```

从日志里的 `FPS:` 估算 9e8 步要多久。sweep 的训练作业时限是 24 小时（`sweep.py` 里 `timeout_min=1440`）。
如果超时，重新提交同一条命令即可，它会从 checkpoint 续训（每 100 个 update 存一次）。
**不要加 `overwrite=True`，否则会删掉已有的实验目录。**
如果显存不够（20 个智能体最可能出现），可以把 `n_envs` 降到 200，但要在 yaml 里改，并在结果里注明。

### 2. 提交训练

```bash
uv run python sweep.py name=ma_repro_binary     mode=train
uv run python sweep.py name=ma_repro_dungeon    mode=train
uv run python sweep.py name=ma_repro_binary_w32 mode=train
```

每个配置提交为一个 SLURM 数组作业（通过 submitit），日志在 `submitit_logs/train/`。
输出在 `saves/<实验名>/`（有 `progress.csv`、`ckpts/`，训练中的渲染在 `vids/`）。
注意：sweep yaml 里的超参数会覆盖命令行上的同名参数，所以要改步数之类的设置，请改 yaml。

### 3. 评估（拿到能和论文对照的数字）

```bash
uv run python sweep.py name=ma_repro_binary mode=eval     # dungeon 和 w32 同理
uv run python cross_eval.py name=ma_repro_binary          # 表格和图输出到 cross_eval/
```

拿 16×16 固定形状地图上的 `mean_ep_reward` 和上面列的论文数字比较。只有 1 个 seed，差几分很正常，
重点看趋势：3 个智能体应该比 1 个好，在大地图和随机形状上差距更明显。

### 4. 渲染 GIF

```bash
uv run python sweep.py name=ma_repro_binary     mode=enjoy n_eval_envs=8
uv run python sweep.py name=ma_repro_dungeon    mode=enjoy n_eval_envs=8
uv run python sweep.py name=ma_repro_binary_w32 mode=enjoy n_eval_envs=8
```

对多智能体模型，`mode=enjoy` 调用的是 `eval_ma.main_eval_ma(render=True)`。每种评估设置会写出
`saves/<实验名>/vids_randMap-<True|False>_w-<宽度>_seed-0/{0..7}.gif` 和 `best.png`
（`best.png` 是 loss 最低的那一帧）。

一定要加 `n_eval_envs=8`：`SweepConfig` 的默认值是 50，而每个环境都会生成一个 GIF，所有帧要先在显存里渲染出来。
32×32 地图每个 episode 有 2048 帧，用 50 个环境很容易 OOM。如果还是 OOM，就降到 4。
渲染作业的时限是 60 分钟（见 `sweep.py`）。

### 5. 整理成 presentation 素材

把挑出来的 GIF 复制到 `presentation_gifs/`（这个目录不要 commit 进仓库，GIF 很大），按下面的格式命名：
`binary_w16_3agents_eval-w32-fixed.gif`。每组至少要有：

1. binary，1 个和 3 个智能体，在 16×16 固定形状上（对应图 2）。
2. binary，3 个智能体，在 32×32 上（16×16 训练的模型泛化到大地图，对应图 4）。
3. binary，20 个智能体，在 16×16 和 32×32 上。
4. dungeon，1 个和 3 个智能体，在 16×16 上（对应图 5）。
5. 32×32 训练的模型，1 个和 3 个智能体，在 32×32 上。
6. 选几个随机形状地图（`randMap-True`）的例子，展示对形状的泛化。

挑选方法：每组先看 `best.png`，再在 8 个 GIF 里挑最终地图最像样的一个（迷宫路径长、只有一个连通区域；
dungeon 里玩家、钥匙、门都在并且连通）。最后写一个简短的 `presentation_gifs/README.md`，说明每个文件
对应的配置和 eval reward。

### 可选

- 论文图 2 是从**全墙地图**开始生成的（“for illustration”），视觉效果更好。现在的代码里 `full_start`
  会被写进实验目录名，所以不能在渲染时单独打开。要实现的话，得仿照 `eval_map_width` 加一个 `eval_full_start`
  选项（改 `conf/config.py`、`eval_ma.init_config_for_eval` 和 `get_eval_name`）。动手前先问 Zehua 要不要做。
- GIF 太大的话，可以用 `gifsicle -O3 --lossy=80` 压缩，或者抽帧。

## 完成后要汇报的内容

- 每个实验的训练时长、最终 return，以及评估表格（贴出 `cross_eval/` 里的路径或数字），并和论文数字对照。
- `presentation_gifs/` 的位置和文件清单。
- 遇到的问题和对代码的改动（如果改了代码，单独 commit）。
