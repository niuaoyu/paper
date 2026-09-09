# 3DShape2VecSet 复现 + 心脏数据适配操作说明（shape2vecset_setup.md）

> 官方仓库：`/home/nay/paper/3DShape2VecSet`（本目录，已打 torch2 兼容补丁）
> 论文：3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models (SIGGRAPH 2023)
> 本机：conda 环境 **`shape`**（python 3.9 + torch 2.2.0 + cu118）｜ GPU RTX 3090 Ti 24GB

---

## 1. 结论：能不能用你的心脏数据跑通？

**能跑通。** 已在本机用 `shape` 环境 + 官方代码 + 心脏数据完成最小冒烟验证：

```
dataset OK   | n_train=34064 (2129 模型 × replica16)
forward/backward OK | logits (1,2048) | loss_vol 0.6938 loss_near 0.6973 | kl 0.036
```

> 注意：仓库原始代码是 torch1.x 时代的，直接装 torch2.x 会报 `from torch._six import inf` 错误。
> 本目录已做两处最小兼容补丁（一键脚本会自动重复执行，幂等）：
> 1. `util/misc.py`：`from torch._six import inf` → `from math import inf`
> 2. `util/shapenet.py`：兼容点数据放在 `<cat>/4_pointcloud/` 的布局（官方布局与转换脚本布局都能读）

---

## 2. 你的数据现状与所需格式

### 2.1 项目需要的数据格式（每样本）
`--data_path DATA` 下需满足：

```
DATA/
├── ShapeNetV2_point/<category>/
│   ├── <model>.npz          # 键: vol_points(N,3), vol_label(N), near_points(M,3), near_label(M)
│   ├── <model>.npy          # 归一化 scale（标量）
│   └── train.lst / val.lst / test.lst   # 每行一个 <model>，如 687_ES
└── ShapeNetV2_watertight/<category>/4_pointcloud/
    └── <model>.npz          # 键: points(S,3) 表面采样点
```
（类别 `<category>` 用 ShapeNet id 占位即可，如 `02691156`；AE 第一阶段不用真实类别标签。）

### 2.2 你现有数据的性质（关键提醒）
- `/home/nay/datasets/MNMS2/raw/points/*.ply`：是 **2D 切片轮廓顶点**（带 label/layer 等属性，**没有面、没有法线、不是水密网格**）——**不能直接**喂给 3DShape2VecSet。
- `/home/nay/datasets/MNMS2/preprocessed/{deepsdf,onet,dif_net}`：DeepSDF 等格式的预采样 SDF/occupancy，**可以**作为转换源。
- `/home/nay/github/DeepSDF/3d_data`：水密网格 + 归一化 SDF（1331 例 × ES/ED），是此前转换的真实数据源。

### 2.3 已经转好的心脏数据（可直接用）
`/home/nay/github/3DShape2VecSet/data/` 下已按上面格式放好：
- `ShapeNetV2_point/02691156/4_pointcloud/*.npz(+.npy)`，约 3287 个（ED/ES）
- `ShapeNetV2_watertight/02691156/4_pointcloud/*.npz`（每样本 5 万表面点）
- `train.lst / val.lst / test.lst`

冒烟测试用的就是这份数据（`--data_path /home/nay/github/3DShape2VecSet/data`）。

### 2.4 转换脚本（若想从你自己的 DeepSDF 数据重新生成）
参考已适配的转换器：
```
/home/nay/github/3DShape2VecSet/convert_deepsdf_to_shape2vecset.py
```
按需改顶部 `INPUT_ROOT / OUTPUT_ROOT / CATEGORY_ID / TRAIN_RATIO…` 后运行即可；
若你的数据目录不是 DeepSDF 布局，把该脚本里 `load_sdf_data / load_mesh` 部分换成你的读取方式，输出仍写上面 2.1 的格式即可（两处布局官方 loader 都兼容）。

---

## 3. 一键环境配置

脚本：`/home/nay/paper/setup_shape2vecset_env.sh`（用法同 setup_bipt_env.sh 系列）

```bash
# 任意 clone 路径都行（自动定位 3DShape2VecSet 仓库）
bash setup_shape2vecset_env.sh
bash setup_shape2vecset_env.sh /path/to/3DShape2VecSet
bash setup_shape2vecset_env.sh --recreate    # 删除重建 shape 环境
bash setup_shape2vecset_env.sh --dry-run     # 只预览
```

覆盖项：

| 变量 | 默认 | 说明 |
|---|---|---|
| `S2VS_ENV_NAME` | `shape` | conda 环境名 |
| `S2VS_PYTHON` | `3.9` | Python |
| `S2VS_TORCH_VERSION` / `S2VS_TORCHVISION_VERSION` | `2.2.0` / `0.17.0` | torch / torchvision |
| `S2VS_CUDA` | `118` | 如 `117/121/cpu` |

脚本做的事：建/复用 env → conda 装 pytorch → pip 装依赖（含 `torch_cluster`，来自 data.pyg.org 对应 torch 版本的 wheel）→ 幂等打 torch2 兼容补丁 → import 自检（torch/cuda/torch_cluster.fps/仓库模块）。

手动等价命令（环境 `shape` 已就绪时可跳过）：
```bash
conda create -n shape python=3.9 -y && conda activate shape
conda install pytorch=2.2.0 torchvision=0.17.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install numpy==1.26.4 scipy==1.13.1 Pillow==11.3.0 h5py==3.14.0 PyYAML==6.0.2 \
    PyMCubes==0.1.6 scikit-image==0.24.0 tensorboard==2.21.0 tqdm==4.68.4 psutil==7.2.2 \
    einops==0.8.2 timm==1.0.27 trimesh==4.12.2
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

---

## 4. 运行

### 4.1 训练 autoencoder（心脏数据，单卡 3090 请缩小 batch）
```bash
conda activate shape
cd /home/nay/paper/3DShape2VecSet

python main_ae.py \
    --model kl_d512_m512_l8 \
    --data_path /home/nay/github/3DShape2VecSet/data \
    --output_dir output/ae/heart \
    --log_dir output/ae/heart \
    --batch_size 4 \
    --accum_iter 4 \
    --num_workers 4 \
    --epochs 200 \
    --warmup_epochs 5
```
> 原 README 用 `torchrun --nproc_per_node=4` + batch 64 + num_workers 60 是 8×A100 配置；
> 单卡 24GB 时把 `batch_size` 调小、用 `accum_iter` 补有效 batch、`num_workers` 别太大。
> 断点续训：加 `--resume output/ae/heart/checkpoint-XXX.pth`。

### 4.2 评估（IoU + 重建）
```bash
python eval.py \
    --model kl_d512_m512_l8 \
    --pth output/ae/heart/checkpoint-199.pth \
    --data_path /home/nay/github/3DShape2VecSet/data
```
> 本目录为官方原版 eval.py（只算指标）；如需导出 `.obj` 网格、按类别过滤、分块 query 防爆显存，
> 可参考已适配版 `/home/nay/github/3DShape2VecSet/eval.py`（支持 `--categories/--export_dir/--query_batch_size`）。

### 4.3 数据快速自检
```python
import types
from util.datasets import build_shape_surface_occupancy_dataset
args = types.SimpleNamespace(data_path='/home/nay/github/3DShape2VecSet/data', point_cloud_size=2048)
ds = build_shape_surface_occupancy_dataset('train', args)
points, labels, surface, cat = ds[0]
print(points.shape, labels.shape, surface.shape)   # (2048,3) (2048,) (2048,3)
```

---

## 5. 注意事项 / FAQ

1. **必须用打补丁后的代码**：任何新 clone 都要跑一次一键脚本或手动补 `util/misc.py` / `util/shapenet.py`，否则 torch2.x 直接 import 报错。
2. **原始 MNMS2 轮廓点不是网格**：要训练必须有水密网格/SDF 采样（`deepsdf` 预采样数据或水密 mesh 都行），否则先生成 mesh（如用 open3d/其它重建）。
3. **AE 第一阶段用不到类别标签**：`02691156` 只是占位 id；第二阶段 category-conditioned 生成若做单类心脏，占位 id 也能用（或把 `category_ids` 扩成你的 id）。
4. **显存**：`kl_d512_m512_l8` 不小；单卡请 batch≤4~8 + accum_iter；评估若爆显存用分块 query（见 4.2 提及的增强 eval）。
5. **训练策略**：过拟合/KL 权重等经验见已适配仓库的
   `/home/nay/github/3DShape2VecSet/{HEART_RECONSTRUCTION_GUIDE.md, FIXES_APPLIED.md, README_CUSTOM_DATA.md}`，
   里面记录了心脏数据下的 batch、KL warmup、best-checkpoint 选择等实践结论。

---

## 6. 相关路径速查
| 用途 | 路径 |
|---|---|
| 本仓库（官方代码，已补丁） | `/home/nay/paper/3DShape2VecSet` |
| 已适配 fork（转换器/增强 eval/训练记录） | `/home/nay/github/3DShape2VecSet` |
| 已转换心脏数据 | `/home/nay/github/3DShape2VecSet/data` |
| DeepSDF 源数据（mesh+sdf） | `/home/nay/github/DeepSDF/3d_data` |
| 原始心脏数据 | `/home/nay/datasets/MNMS2` |
| conda 环境 | `shape`（python3.9 / torch2.2.0 / cu118） |
| 一键环境脚本 | `/home/nay/paper/setup_shape2vecset_env.sh` |
