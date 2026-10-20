# Isaac Lab 迁移

此入口迁移原默认任务 `combo_go2ARX5_pickle_reaching_extreme`：Go2 + ARX5 在平地跟踪世界坐标系的末端轨迹，包括训练、续训、推理和 TorchScript 导出。

**当前阶段：工程链路验证已通过，任务性能与原版行为一致性仍待验收。** 下述短训练和推理结果用于检查运行链路，尚不能证明有效跟踪、策略收敛或无缝迁移。

## 环境与依赖

适配本机已下载的 **Isaac Lab v1.2.0 + Isaac Sim 4.2.0.2**，使用 `omni.isaac.lab` API。Python 为 3.10，PyTorch 为 2.4.0。Isaac Lab 2.x/3.x 的 API 不在此次验证范围内。

项目专用环境位于 `mani-centric-wbc/.venv-isaaclab`，不继承其他环境的 site-packages。依赖版本见 [requirements-isaaclab.txt](../requirements-isaaclab.txt)。以下命令均在 `mani-centric-wbc/` 下执行：

```bash
python3.10 -m venv .venv-isaaclab
source .venv-isaaclab/bin/activate
python -m pip install --no-cache-dir -r requirements-isaaclab.txt
```

另外安装 Isaac Lab 的核心 Python 包。本机已有 `/home/linux/PycharmProjects/IsaacLab`，对应 v1.2.0：

```bash
python -m pip install -e /home/linux/PycharmProjects/IsaacLab/source/extensions/omni.isaac.lab
python -m pip check
```

其他机器先下载指定版本，再将上述路径替换为自己的路径：

```bash
git clone --branch v1.2.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git /path/to/IsaacLab
python -m pip install -e /path/to/IsaacLab/source/extensions/omni.isaac.lab
```

运行需要 Linux 和支持 Isaac Sim 4.2 的 NVIDIA 驱动/GPU。首次启动会编译着色器、转换 URDF。阅读并接受 [Isaac Sim EULA](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/installation/install_python.html) 后设置：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

无需安装 Isaac Gym、PyTorch3D、外部 rsl_rl 或原版 `isaac.yml`。训练直接复用本仓库的 PPO。

## 数据

使用作者发布的数据：

```bash
curl -L https://real.stanford.edu/umi-on-legs/wbc/data.zip -o /tmp/umi-wbc-data.zip
unzip /tmp/umi-wbc-data.zip -d .
```

得到 `data/tossing.pkl`、`data/pushing.pkl`、`data/umi_cup_in_wild.pkl`。数据格式为字典列表，每项包含 `ee_pos` 和 `ee_axis_angle`，均为 `(T, 3)` 数组，默认 200 Hz。加载时使用原版的 XY 居中方式，截取前 7 秒；较短序列和超出末帧的预览使用最后一帧。数据保存在 CPU，仅将当前查询结果传到 GPU。只加载可信的 pickle 文件。

## 训练、续训与推理

默认使用 4096 个环境、每轮 24 步、32 个 PPO epoch、4 个 mini-batch：

```bash
python scripts/train_lab.py --headless --device cuda:0 \
  --trajectory data/tossing.pkl --output logs/lab_tossing
```

短训练验证：

```bash
python scripts/train_lab.py --headless --device cuda:0 \
  --num_envs 32 --trajectory data/tossing.pkl --iterations 2 --epochs 1 \
  --output logs/lab_smoke
```

续训加载模型、Adam 状态、学习率和迭代编号。`--iterations` 表示本次新增轮数；仿真状态、随机化和课程统计重新初始化：

```bash
python scripts/train_lab.py --headless --device cuda:0 \
  --num_envs 32 --trajectory data/tossing.pkl --iterations 2 --epochs 1 \
  --checkpoint logs/lab_smoke/model_last.pt --output logs/lab_resume
```

确定性推理并导出 `logs/lab_play/actor.pt`，去掉 `--headless` 可打开视窗：

```bash
python scripts/train_lab.py --headless --device cuda:0 \
  --num_envs 4 --trajectory data/tossing.pkl --play --steps 360 \
  --no_randomization --no_noise \
  --checkpoint logs/lab_resume/model_last.pt --output logs/lab_play
```

训练输出包括 `config.json`、`metrics.jsonl`、`model_last.pt` 和定期 checkpoint。不使用 W&B。没有数据时可显式传 `--synthetic` 做链路验证，该选项与 `--trajectory` 互斥。

## 控制与学习设计

| 项目 | 设置 |
| --- | --- |
| 物理 / 策略频率 | 200 Hz / 50 Hz |
| 动作 | 18 维关节位置增量，缩放 0.25；每个物理步执行 PD 并限幅力矩 |
| 动作延迟 | 20 ms，即 4 个物理步 |
| Actor / Critic | 96 / 225 维输入；128、64、32 隐层，ELU |
| 特权信息 | Critic 额外获得质量、质心、摩擦、阻尼、PD 增益和机身线速度 |
| 目标预览 | 0.02、0.04、0.06、1.0 秒后的末端位姿 |
| PPO | clip 0.2，gamma 0.99，GAE lambda 0.95，entropy 0.01，KL 自适应学习率 |
| 重置 | 7 秒超时，或指定机身/腿/机械臂部件发生接触 |
| 随机化 | 质量、质心、接触摩擦、关节阻尼/摩擦、PD 增益、初始位姿；每 4 秒位置扰动、每 5 秒速度扰动 |

奖励沿用原任务的系数：位姿跟踪为 `4 * exp(-位置误差平方 / pos_sigma - 旋转角误差 / orn_sigma)`；叠加动作变化、关节加速度、力矩、关节限位、碰撞、足端负载分配、足髋距离、机械臂对齐、机身高度和倾斜惩罚。总和截断到非负后乘控制周期 `0.02`。误差课程逐渐收紧 `pos_sigma` 和 `orn_sigma`，配置见 [config.py](../umi_lab/config.py)，奖励见 [env.py](../umi_lab/env.py)。

关节通过名字映射为 FR、FL、RR、RL 的 hip/thigh/calf，随后 joint1～joint6，避免 PhysX 遍历顺序改变动作含义。URDF 由原资源转换，生成文件保存在 `.cache/isaaclab/`。

## 验证与边界

2026-10-20，在上述独立虚拟环境和 RTX 4080 SUPER 上完成以下验证。测试脚本仅留本机，不纳入提交。

| 验证 | 结果 |
| --- | --- |
| 依赖检查 | `pip check` 通过 |
| CPU 检查 | 4 项通过，覆盖旋转约定、关节顺序、轨迹末帧、PPO 更新、超时 bootstrap 和 checkpoint 恢复 |
| GPU 集成 | 4 个环境通过观测布局、属性随机化、四步延迟、力矩限幅、局部重置、扰动、超时与接触终止检查；8 次超时、3 次接触终止 |
| 默认规模训练 | `tossing.pkl`，4096 环境，24 步采样，共 98,304 条 transition；32 epoch、4 mini-batch，完成 128 次 Adam 更新；观测、奖励和损失均为有限值 |
| 续训 | 恢复上述模型和 Adam 状态，32 环境完成 1 轮；迭代编号从 1 变为 2，Adam step 从 128 变为 132，模型参数发生变化 |
| 推理与导出 | 4 环境运行 360 步，导出 `actor.pt`；重新加载后，在 256 组相同输入上与迁移版 actor 的最大绝对输出差为 0 |

**短训模型尚未取得可用跟踪效果的证据。** 此次推理平均位置误差为 `0.318589 m`（约 31.9 厘米），发生 18 次接触终止、0 次超时结束；累计重置 22 次包含首次 4 个环境重置。各环境运行 7.2 秒，未取得完整 7 秒回合的超时结束记录。误差统计包含自动重置后的片段，使用的仍是训练数据，不能作为正式任务性能评估。

该模型仅经过上述一轮默认规模训练和一轮小规模续训。后续需对照两后端的观测、奖励、控制及物理行为，并在固定评估轨迹上记录完整回合的跟踪误差和终止情况，通过多个随机种子的训练曲线验证收敛与性能。

可用上面的短训练和推理命令复查链路。日志、数据、虚拟环境和生成资产均不纳入提交。

原 Isaac Gym checkpoint 的关节及碰撞形状顺序与此后端不同，加载器要求迁移版的关节约定，需重新训练。导出的 actor 接收上述 96 维观测，输出 18 维动作；实机部署仍需核对观测与硬件接口。

此次范围是默认 reaching 主任务。原来的 locomotion、Perlin 地形、其他消融配置、Blender 记录接口和实机部署脚本仍使用原入口。Isaac Sim 的资产转换和接触求解与 Isaac Gym 存在差异；此次验证没有比较两者的长期训练结果。短训练用于验证仿真和优化链路，不能证明策略收敛、论文性能或实机效果。
