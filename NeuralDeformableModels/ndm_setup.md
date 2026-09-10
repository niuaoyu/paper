# NeuralDeformableModels (NDM) 环境与心脏数据操作说明（ndm_setup.md）

> 论文：Neural Deformable Models for 3D Bi-Ventricular Heart Shape Reconstruction and Modeling from 2D Sparse Cardiac MRI (ICCV 2023)
> 官方仓库：`/home/nay/paper/NeuralDeformableModels`（本目录）
> 旧复现/适配版：`/home/nay/github/NeuralDeformableModels`
> 本机：conda 环境 **`ndm`**（python 3.9 + torch 2.2.0 + cu118）｜RTX 3090 Ti 24GB

---

## 1. 结论：能否用你的数据跑通？

**训练/评估管线本身已验证可跑通**（本机用 `ndm` 环境实测，见第 5 节）。
但要区分两种“数据”：

| 数据 | 能否直接跑 | 说明 |
|---|---|---|
| `/home/nay/github/DeepSDF/3d_data`（已转换，1331 例） | ✅ 可以直接跑 | 旧复现训练用的就是它，本节的冒烟测试也用它（含 GT 标签网格 + 稀疏输入点 + 归一化参数） |
| `/home/nay/datasets/MNMS2`（你的原始数据） | ❌ 不能直接跑 | 格式与 NDM 两条管线都不同，需要先转换（见第 3 节） |

> 一句话：**NDM 项目 + 心脏数据是能跑通的；但你 /home/nay/datasets 里的 MNMS2 需要先转成 DeepSDF-3d_data 或官方 NDM-NPZ 格式。**

---

## 2. 项目需要什么数据

### 2.1 已验证的“两表面心脏管线”（本仓库已同步）
数据根目录（`--data-root`）需为：

```
<root>/
├── mesh/<case>/ED.ply, ES.ply     # 带 per-vertex label 的三角网格：1=LV, 2=RV（+rgb）
├── points/<case>/ED.ply, ES.ply   # 稀疏输入点云（心脏 MRI 轮廓点），binary PLY: x,y,z float32 + label uchar + rgb uchar
└── norm/<case>/
    ├── pose.npy                   # 4x4 姿态/方向矩阵
    ├── unit_sphere.npz            # keys: offset(3), scale(标量) —— 归一化到单位球
    └── aabb.npz                   # keys: offset(3), scale(标量) —— 归一化到 [-1,1]^3
```
训练时：输入点云重采样到 5600 点，GT 从 `mesh` 按 label 取 LV/RV 表面各采样 3000 点（`NeuralDeformableModel/dataset/heart_dataset.py`）。

### 2.2 官方 NDM 原始管线（train_ndm.py）所需的 NPZ
`tools/convert_deepsdf_3d_data_to_ndm.py` 的说明：每 case/phase 需要 4 个未命名 NPZ：

```
HR_<PHASE>_pc.npz        (5600, 3)   # 稀疏输入
HR_<PHASE>_epi_myo.npz   (3000, 3)   # LV epicardium
HR_<PHASE>_endo_myo.npz  (3000, 3)   # LV endocardium
HR_<PHASE>_rv.npz        (3000, 3)   # RV
```
注意：DeepSDF `3d_data` 只有 LV/RV 两套标签，没有独立的 LV 心外膜/心内膜，
因此该转换器默认拒绝臆造缺失表面；`--approximate-missing-lv` 仅用于管线冒烟。

---

## 3. 你的 MNMS2 数据现状与转换要求

`/home/nay/datasets/MNMS2` 现有内容：

| 目录 | 内容 | 问题 |
|---|---|---|
| `raw/points/*.ply` | 2D 切片轮廓顶点（binary PLY：x/y/z double + label/view/layer/contour_id/rgb） | 不是 3D 表面、PLY 属性布局也和 NDM 期望不同 |
| `raw/mesh/*.ply` | 网格（ascii PLY，x/y/z + rgb，**无 per-vertex label**） | 无法直接按 label 取 LV/RV GT |
| `preprocessed/deepsdf/*.npz` | 键：`pcd`, `structure`, `view_labels`, `sax_layers`, `query_points`, `query_sdf`, `center`, `scale` | 既不是 DeepSDF-3d_data 布局，也不是官方 NDM NPZ 布局 |

**要跑通，需要做（三选一）：**

1. **最省事**：直接用已转换好的 `/home/nay/github/DeepSDF/3d_data`（1331 例，格式见 2.1），本仓库的 `train_simple.py` 直接可跑。
2. **从 MNMS2 转成 2.1 的布局**：
   - `mesh/<id>/{ED,ES}.ply`：把 `raw/mesh/<id>_<phase>.ply` 加上 per-vertex LV/RV label（可从原始分割掩膜/颜色编码重建；参考 `NeuralDeformableModel/dataset/script_generate_input_sparse_point_cloud_from_seg_masks_sax_and_lax_*.py`）；
   - `points/<id>/{ED,ES}.ply`：从 `raw/points`（轮廓点）清洗/重采样成输入点云，写成 `x,y,z float32 + label uchar + rgb uchar` 的 binary PLY；
   - `norm/<id>/{pose.npy, unit_sphere.npz, aabb.npz}`：由点云/网格统一计算（offset=质心、scale=1/max半径 等）。
3. **转成官方 NDM NPZ（2.2）**：用 `tools/convert_deepsdf_3d_data_to_ndm.py`（需先有 2.1 布局的数据），或自行从分割掩膜生成三个 dense 目标表面。

> 转换后务必先跑对齐检查（`test_input_gt_alignment.py` / 训练时的自动门禁）：输入点与 GT 的 LV-RV 连线方向夹角应小于阈值（默认 20°），否则训练会直接报错停止。

---

## 4. 环境：一键配置（环境名 `ndm`）

脚本：`/home/nay/paper/setup_ndm_env.sh`

```bash
bash setup_ndm_env.sh                 # 任意路径自动定位仓库；默认创建 env=ndm
bash setup_ndm_env.sh --recreate      # 删除重建
bash setup_ndm_env.sh --dry-run       # 只预览
bash setup_ndm_env.sh --no-sync       # 不同步心脏管线文件
bash setup_ndm_env.sh --with-official-deps   # 追加官方管线/评估依赖（open3d/SimpleITK/pykeops/pytorch3d…）
```
覆盖变量：`NDM_ENV_NAME`（默认 ndm）、`NDM_PYTHON`（3.9）、`NDM_TORCH_VERSION`（2.2.0）、`NDM_CUDA`（118）、`NDM_HEART_SOURCE`（默认 `/home/nay/github/NeuralDeformableModels`）。

脚本做的事：
1. 建/复用 `ndm` 环境（python 3.9 + pytorch 2.2.0 + torchvision 0.17.0 + pytorch-cuda 11.8）；
2. 安装训练/评估所需 pip 依赖（numpy/scipy/torchdiffeq/trimesh/tqdm/scikit-image/matplotlib 等）；
3. 从旧复现仓库同步**已验证的心脏管线**：`train_simple.py`、`evaluate_two_surface.py`、`chamfer_loss.py`、`test_input_gt_alignment.py`、`NeuralDeformableModel/dataset/{heart_dataset,acquisition_geometry}.py`、`model/{model,pointnet_util}.py`、`tools/*`；
4. 自动打 torch≥2.0 兼容补丁（`torch.meshgrid(..., indexing='ij')`、`square_distance` 的显存优化实现）；
5. 自检：torch CUDA、NDM 模型前向（5600 点输入，输出 primitive 15500×3）。

> 注：`shape` 环境是此前 3DShape2VecSet 用的；NDM 建议用本次新建的独立 `ndm` 环境。

---

## 5. 本机验证记录（ndm 环境）

```
torch: 2.2.0 | cuda: True | gpu: NVIDIA GeForce RTX 3090 Ti
NDM model forward OK | outputs: 29 | primitive: (1, 15500, 3)

# 训练冒烟（translation 阶段，4 训练样本 / 2 验证样本，2 epochs）
采集对齐 0/ED: 2.65° , 0/ES: 2.45°   （通过 <20° 门禁）
Epoch 0 Train 2.328 | Epoch 1 Train 1.762  → 保存 best.pth

# Neural ODE 阶段冒烟（torchdiffeq）
Epoch 0 [ode] Train 5.861 | 保存 best.pth

# 评估冒烟（case 0）
0_ED: LV=0.751 RV=1.112 | 0_ES: LV=0.757 RV=1.129
导出 0_ED/0_ES 的 input/gt/pred PLY + metrics.json
```

---

## 6. 正式训练 / 评估命令

```bash
conda activate ndm
cd /home/nay/paper/NeuralDeformableModels

# 正式训练（两表面，单卡 3090 推荐）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_simple.py \
  --data-root /home/nay/github/DeepSDF/3d_data \
  --epochs 100 --stage auto --stage-epochs 5,5,10,20 \
  --batch-size 4 --ode-batch-size 1 --lr 1e-4 \
  --num-workers 4 --prefetch-factor 2 --save-freq 5 \
  --output-dir output/two_surface_4090_fresh

# 断点续训（只能续“当前修复后代码生成且配置一致”的 checkpoint）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_simple.py \
  --data-root /home/nay/github/DeepSDF/3d_data \
  --epochs 100 --stage auto --stage-epochs 5,5,10,20 \
  --batch-size 4 --ode-batch-size 1 --lr 1e-4 \
  --num-workers 4 --prefetch-factor 2 --save-freq 5 \
  --output-dir output/two_surface_4090_fresh \
  --resume output/two_surface_4090_fresh/checkpoints/latest.pth

# 评估（导出预测 mesh + 指标）
python evaluate_two_surface.py \
  --checkpoint output/two_surface_4090_fresh/checkpoints/best.pth \
  --data-root /home/nay/github/DeepSDF/3d_data \
  --output-dir output/two_surface_4090_fresh_evaluation \
  --case-ids 0 16 --batch-size 1
```

快速冒烟（不占太多时间）：
```bash
python train_simple.py --data-root /home/nay/github/DeepSDF/3d_data \
  --epochs 2 --stage translation --batch-size 1 --num-workers 0 \
  --max-train-samples 4 --max-val-samples 2 --save-freq 1 \
  --alignment-check-cases 0 --output-dir output/smoke
```

---

## 7. 常见问题

1. **对齐门禁失败**（输入点与 GT 方向夹角 >20°）：说明输入点采集/GT 坐标系不一致，需重新生成点云或检查 pose/归一化；诊断用 `test_input_gt_alignment.py`，仅调试时可 `--skip-alignment-check`。
2. **显存不足（OOM）**：`--batch-size` 降到 2~4、`--ode-batch-size 1`，并设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
3. **不要续训旧 checkpoint**：官方指南明确，改动过采样点数（5000→5600）/GT 坐标的旧权重不可直接续训；正式实验从 epoch 0 重新训练。
4. **官方 train_ndm.py / 官方 metrics**：需要额外依赖（pytorch3d、pykeops、open3d、SimpleITK、comet_ml 等），可用 `setup_ndm_env.sh --with-official-deps`；其中 `mesh_intersection`（torch-mesh-isect）不在 PyPI，需按官方仓库自行编译。
5. **你的 MNMS2 与 3d_data 的关系**：3d_data 有 1331 例带 label 的网格；MNMS2 是 512 例的轮廓点/无 label 网格。若必须严格用 MNMS2，请按第 3 节做转换并先做对齐检查。

---

## 8. 路径速查

| 用途 | 路径 |
|---|---|
| 本项目（官方代码 + 已同步心脏管线） | `/home/nay/paper/NeuralDeformableModels` |
| 旧复现/适配版（含 checkpoints、分析文档） | `/home/nay/github/NeuralDeformableModels` |
| 旧复现训练/评估指南 | `/home/nay/github/NeuralDeformableModels/TRAIN_EVALUATION_GUIDE.md` |
| 论文分析 | `/home/nay/github/NeuralDeformableModels/NDM_Analysis.md` |
| 可直接训练的数据（1331 例） | `/home/nay/github/DeepSDF/3d_data` |
| 你的原始数据 | `/home/nay/datasets/MNMS2` |
| 一键环境脚本 | `/home/nay/paper/setup_ndm_env.sh` |
| conda 环境 | `ndm`（python3.9 / torch2.2.0 / cu118） |
