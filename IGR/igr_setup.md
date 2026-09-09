# IGR 一键环境配置说明（igr_setup.md）

> 项目：**IGR: Implicit Geometric Regularization for Learning Shapes**（ICML 2020）
> 仓库：`/home/nay/paper/IGR` ｜ 官方 README：见本目录 `README.md`
> 本机验证环境：Ubuntu + conda + NVIDIA RTX 3090 Ti（sm_86）

---

## 1. 推荐方式：一键脚本

一键脚本位于仓库外一层：`/home/nay/paper/setup_igr_env.sh`（可拷到任意位置使用，脚本会自动定位 IGR 仓库）。

```bash
# 方式 A：脚本放在 IGR 仓库根目录 / 任意父目录（会自动往下找到 IGR）
bash setup_igr_env.sh

# 方式 B：脚本在别处，显式指定仓库路径
bash /home/nay/paper/setup_igr_env.sh /home/nay/paper/IGR

# 从 IGR 目录内直接调用父目录脚本：
cd /home/nay/paper/IGR && bash ../setup_igr_env.sh
```

常用选项：

```bash
bash setup_igr_env.sh --dry-run      # 只打印计划，不安装
bash setup_igr_env.sh --recreate     # 删除已有 igr 环境后完整重建
bash setup_igr_env.sh --repo /path/to/IGR
```

可用环境变量覆盖默认值：

| 变量 | 默认 | 说明 |
|---|---|---|
| `IGR_ENV_NAME` | `igr` | conda 环境名 |
| `IGR_PYTHON` | `3.7.9` | Python 版本 |
| `IGR_TORCH_VERSION` | `1.13.1` | PyTorch 版本 |
| `IGR_TORCH_CUDA` | `117` | torch wheel 的 CUDA 标识（`113/116/117/102/cpu`） |

脚本会自动完成：创建环境 → 安装依赖 → 安装 torch → 验证（import + CUDA 算子 + IGR 模型前向/反向冒烟）。

---

## 2. 手动安装步骤（等价于脚本做的事）

```bash
conda create -n igr -c conda-forge python=3.7.9 -y
conda activate igr
python -m pip install -U "pip<24"            # pip 24 起不再支持 python 3.7

python -m pip install numpy==1.19.2 scipy==1.6.2 pyhocon plotly==5.9.0 \
    scikit-image==0.18.1 trimesh==3.9.14 GPUtil

# GPU 版 torch（python 3.7 可用且支持 RTX 30 系的最高版本）
python -m pip install torch==1.13.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

# 无 GPU 时：
# python -m pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. 为什么这样配（版本结论）

| 项 | 选择 | 原因 |
|---|---|---|
| python | **3.7.9** | 官方 README 指定 python 3.7 |
| torch | **1.13.1+cu117**（替代官方 1.2） | torch 1.2 没有 Ampere/RTX 30 系(sm_86) kernel；py3.7 能装的最高 torch 是 1.13.1，`+cu117` 支持 sm_86，且 IGR 代码只用基础 `torch.nn` / `torch.autograd.grad`，实测兼容 |
| numpy / scipy | 1.19.2 / 1.6.2 | python 3.7 的 wheel 组合（与 digs 环境一致） |
| scikit-image | **0.18.1**（必须 <0.19） | IGR 用 `skimage.measure.marching_cubes_lewiner`，该函数在后续版本被改名/移除 |
| plotly | 5.9.0 | `code/utils/plots.py` 顶层 `import plotly`，训练/可视化路径必需 |
| pyhocon / trimesh / GPUtil | 最新/3.9.14 | 配置解析 / 点云网格读写 / GPU 选择 |

> 依赖清单不是照抄 README，而是按代码 import 扫描确定的：numpy、scipy、pyhocon、plotly、scikit-image、trimesh、GPUtil + torch。

---

## 4. 本机验证结果（脚本实际输出）

```
python : 3.7.9
numpy  : 1.19.2 | scipy: 1.6.2 | skimage: 0.18.1 | plotly: 5.9.0 | trimesh: 3.9.14
torch  : 1.13.1+cu117 | cuda: True | gpu: NVIDIA GeForce RTX 3090 Ti
cuda op 通过
model smoke OK | eikonal loss 正常反传   (ImplicitNet + gradient)
utils.plots import OK
```

---

## 5. 使用示例

### 5.1 单形状表面重建（需要自备点云）

编辑 `code/reconstruction/setup.conf`，把 `train.input_path` 改成你自己的点云路径（支持 `.xyz/.npy/.npz/.ply`），并按维度设置 `d_in`（3D=3，2D=2）：

```bash
conda activate igr
cd /home/nay/paper/IGR/code

# 训练
python reconstruction/run.py --conf reconstruction/setup.conf --gpu 0

# 导出网格（CHECKPOINT 可为具体 epoch 或 latest）
python reconstruction/run.py --conf reconstruction/setup.conf --eval --checkpoint latest --gpu 0
```

> 仓库未自带重建用点云数据，`setup.conf` 里默认路径是作者机器的绝对路径，必须先改。

### 5.2 D-Faust 预训练模型推理（仓库自带 trained_models）

```bash
conda activate igr
cd /home/nay/paper/IGR/code

python shapespace/eval.py \
    --checkpoint 1200 \
    --exp-name dfaust_pretrained \
    --split dfaust/test_all.json \
    --exps-dir ../trained_models
```

（拆分 json 与 D-Faust 点云需按官方 README 准备；仅看插值演示可参考 `shapespace/interpolate.py`。）

---

## 6. 常见问题

1. **换到更新的 GPU（如 H100/sm_90）跑不了？**
   py3.7 下 torch 最高 1.13.1，不支持 sm_90。需要整体升级（python≥3.8 + torch 2.x），并重新验证 skimage/plotly 版本。
2. **`marching_cubes_lewiner` 报错？**
   说明 scikit-image 被升到 ≥0.19 了，锁回 `scikit-image==0.18.1`。
3. **装 torch 后 CUDA 不可用？**
   确认装的是 `+cu117` 版本（`pip list | grep torch`），且驱动足够新；无 GPU 环境用 `cpu`。
4. **重建时 `GPUtil` 找不到空闲 GPU？**
   `run.py` 默认用 GPUtil 自动挑卡；显式 `--gpu 0` 可绕过。
