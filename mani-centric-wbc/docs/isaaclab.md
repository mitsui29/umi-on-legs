# Isaac Lab 迁移

此入口迁移原默认任务 `combo_go2ARX5_pickle_reaching_extreme`：Go2 + ARX5 在平地跟踪世界坐标系的末端轨迹，包括训练、续训、推理和 TorchScript 导出。

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
