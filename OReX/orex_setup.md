完成 ✅ `/home/nay/paper/setup_orex_env.sh` 已创建并**从零跑通验证**，新环境 `orex` 也建好了。

## 验证结果（一键脚本实际执行）

```
python : 3.10.21
numpy  : 1.24.4 | scipy: 1.9.3 | sklearn: 1.0.2 | shapely: 1.8.5 | trimesh: 3.12.9
numpy-stl OK
torch  : 1.13.1+cu117 | cuda: True | gpu: NVIDIA GeForce RTX 3090 Ti   (真实算子通过)
csl smoke OK: Elephant | planes: 24     (内置示例数据解析通过)
```

## 脚本用法（任意 clone 路径）

```bash
bash setup_orex_env.sh                  # 放在 OReX 内 / 任意父目录，自动定位仓库
bash setup_orex_env.sh /path/to/OReX    # 或显式指定
bash setup_orex_env.sh --recreate       # 删除重建
bash setup_orex_env.sh --dry-run        # 只预览
```
覆盖项：`OREX_ENV_NAME` / `OREX_PYTHON`（默认 3.10）/ `OREX_TORCH_CUDA`（默认 117，可 `113/116/102/cpu`）。

## 为什么这样配（几个关键决策）

| 点 | 处理 | 原因 |
|---|---|---|
| python | **3.10** | requirements 里 numpy~=1.24（无 py3.7 wheel）且 torch==1.13.1（无 py3.11 wheel），3.10 是全部能装 wheel 的交集 |
| torch | 从官方索引装 **1.13.1+cu117**（不是 PyPI 的 CPU 版） | PyPI 上 `torch==1.13.1` linux 是 CPU 版，而 OReX 靠 `--cuda_device` 用 GPU；cu117 支持 RTX 30 系 sm_86 |
| requirements.txt | 先过滤掉 `torch` 行再装其余 | 避免先装 CPU torch 再被覆盖的浪费/歧义 |
| numpy-stl | 额外安装 | requirements 漏了它，但 `Slicer.py` 顶层 `from stl import mesh`，不装会报错 |
| 验证 | import 全家 + torch CUDA 算子 + 解析示例 `.csl` | 不启动完整训练，快速确认链路可用 |

## 运行示例

```bash
conda activate orex
cd /home/nay/paper/OReX
python Main.py ./Artifacts ./Data/csl_with_ref/eight_15.csl --cuda_device 0
```

以后换新服务器只需把这个脚本拷到 OReX clone 所在处（或任意父目录/加路径参数）执行一次即可；`--recreate` 可在本机随时重建。