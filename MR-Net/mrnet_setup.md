# MR-Net 环境与心脏数据操作说明（mrnet_setup.md）

> 论文：Shape Registration with Learned Deformations for 3D Shape Reconstruction from Sparse and Incomplete Point Clouds (Medical Image Analysis 2021)
> 官方仓库：`/home/nay/paper/MR-Net`（本目录，原版 TensorFlow 1.x 代码 + 已同步的 PyTorch 迁移版）
> 旧复现/迁移版：`/home/nay/github/MR-Net`（含 `pytorch_mrnet/`、`MR_Net_Analysis.md`、output 结果）
> 本机：conda 环境 **`mrnet`**（python 3.9 + torch 2.2.0 + cu118）｜RTX 3090 Ti 24GB

---

## 1. 结论：能否跑通？

分两层说清楚：

| 版本 | 能否跑 | 说明 |
|---|---|---|
| 原版（`train.py`/`test.py`/`mrnet/`，Python 2.7 + TensorFlow 1.7 + TensorLayer） | ❌ 现代机器不可跑 | py2.7/TF1.7 无可用 wheel，且老 CUDA 不支持 RTX 30/40；只能作代码参考 |
| PyTorch 迁移版（`pytorch_mrnet/`，本仓库已同步） | ✅ 已实测跑通 | 完整的数据加载 → 训练 → 验证 → 测试 → 可视化闭环（见第 5 节） |

> ⚠️ 定位：这是 **MR-Net 架构可行性迁移**，不是论文数值结果的完整复现。
> 原因：缺少论文的共享拓扑 GT 网格、官方模板和 UKBB 训练数据（细节见
> `/home/nay/github/MR-Net/MR_Net_Analysis.md` 第七部分）。
> 当前实现用 Chamfer/边长/Laplacian 监督，本身不要求顶点对应关系。

---

## 2. 数据说明

### 2.1 原版 MR-Net 需要的数据格式
```
Data/
├── Manual/  samplexx.vtk   # 输入：2D 切片轮廓点云（堆叠）
└── Shapes/  samplexx.obj   # GT：模板变形得到的 3D 网格（共享拓扑）
```
（仓库自带 `Data/heart/`：`heart_face1/2/3.obj` 模板 + `test.vtk` demo 轮廓。）

### 2.2 PyTorch 迁移版使用的数据（已验证）
使用 DeepSDF 风格的心脏数据根（`--data-root`）：

```
<root>/mesh/<case>/ED.ply, ES.ply     # 带 per-vertex label：1=LV, 2=RV
<root>/points/<case>/ED.ply, ES.ply   # 稀疏输入点云（重采样到 3000 点）
<root>/norm/<case>/{pose.npy, unit_sphere.npz, aabb.npz}  # 坐标归一化
```
程序内部把输入/目标都变换到 canonical 空间，并构造两部件固定拓扑模板（LV/RV）。

### 2.3 你的 `/home/nay/datasets/MNMS2`
| 内容 | 能否直接用 |
|---|---|
| `raw/points/*.ply`（2D 切片轮廓顶点，属性含 label/view/layer） | ❌ 需清洗/重采样成 3000 点输入并做归一化 |
| `raw/mesh/*.ply`（无 per-vertex LV/RV label 的 RGB 网格） | ❌ 需先生成带 label 的网格（或改用 Chamfer 监督） |
| `preprocessed/deepsdf/*.npz`（pcd/query_sdf/center/scale） | ❌ 与迁移版所需布局不同，需转换 |

**推荐做法**：
- 快速跑通/训练：直接用已转换好的 `/home/nay/github/DeepSDF/3d_data`（1331 例，含 label 网格 + 输入点 + 归一化），本仓库的 `pytorch_mrnet` 已验证可跑。
- 严格使用 MNMS2：需要转换出 `mesh/<id>/{ED,ES}.ply(带 label 1/2)`、`points/<id>/{ED,ES}.ply`、`norm/<id>/{pose.npy,unit_sphere.npz}`；转换脚本可参考 DeepSDF/NDM 的同类工具（`NeuralDeformableModel/dataset/script_generate_input_sparse_point_cloud_from_seg_masks_sax_and_lax_*.py`、`tools/convert_deepsdf_3d_data_to_ndm.py`）。

---

## 3. 环境：一键配置（环境名 `mrnet`）

脚本：`/home/nay/paper/setup_mrnet_env.sh`

```bash
bash setup_mrnet_env.sh              # 任意路径自动定位仓库；默认 env=mrnet
bash setup_mrnet_env.sh --recreate   # 删除重建
bash setup_mrnet_env.sh --dry-run    # 只预览
bash setup_mrnet_env.sh --no-sync    # 不同步 pytorch_mrnet
```
覆盖变量：`MRNET_ENV_NAME`（默认 mrnet）、`MRNET_PYTHON`（3.9）、`MRNET_TORCH_VERSION`（2.2.0）、`MRNET_CUDA`（118）、`MRNET_SOURCE`（默认 `/home/nay/github/MR-Net`）。

脚本做的事：
1. 建/复用 `mrnet` 环境（python3.9 + pytorch2.2.0 + torchvision0.17.0 + pytorch-cuda11.8）；
2. 安装依赖：numpy/scipy/trimesh/Pillow/matplotlib/tqdm/psutil/PyYAML；
3. 从旧复现仓库同步 `pytorch_mrnet/`、`MR_Net_Analysis.md`、`Data/heart`；
4. 自检：torch CUDA + `MinimalMRNet` 合成前向（icosphere 模板 + 3000 随机轮廓点）。

> 注：`shape` 环境是此前 3DShape2VecSet 用的；MR-Net 请用本次新建的独立 `mrnet` 环境。

---

## 4. 运行命令

```bash
conda activate mrnet
cd /home/nay/paper/MR-Net

# 1) 数据/对齐检查
python -m pytorch_mrnet.check_dataset  --data-root /home/nay/github/DeepSDF/3d_data --cases 0 1 2
python -m pytorch_mrnet.check_alignment --data-root /home/nay/github/DeepSDF/3d_data --cases 0 1

# 2) 单例过拟合可行性（默认 500 步；500 步时 feasibility_pass=true）
python -m pytorch_mrnet.train_feasibility \
  --data-root /home/nay/github/DeepSDF/3d_data --mode single --steps 500 \
  --output output/pytorch_mrnet/single --device cuda

# 3) 5 例（ED/ES）训练
python -m pytorch_mrnet.train_feasibility \
  --data-root /home/nay/github/DeepSDF/3d_data --mode five-cases --steps 500 \
  --output output/pytorch_mrnet/five_cases --device cuda

# 4) case-disjoint 完整评测协议（默认 train 0-79 / val 80-89 / test 90-99，3 个随机种子）
python -m pytorch_mrnet.baseline \
  --data-root /home/nay/github/DeepSDF/3d_data \
  --output output/pytorch_mrnet/baseline \
  --train-cases 80 --validation-cases 10 --test-cases 10 \
  --epochs 12 --batch-size 8 --seeds 1024 2048 4096 --device cuda
```
输出：`summary.json`（Chamfer/改进率/唯一性/ED-ES 体积方向/mesh 质量）、`history.json`、
`checkpoint.pt`/`best_checkpoint.pt`、预测 `.ply`、正交投影 PNG（`visualizations/`）。

> 冒烟提示：小配置（如 `--train-cases 4 --validation-cases 2 --test-cases 2 --epochs 2 --seeds 1024`）
> 能跑完整流程，但 `baseline_pass` 会是 false，属正常现象——判据需要完整协议与足够训练。

---

## 5. 本机验证记录（mrnet 环境）

```
torch: 2.2.0 | cuda: True | gpu: NVIDIA GeForce RTX 3090 Ti
MR-Net synthetic forward OK | output mesh: (1, 162, 3)

# check_dataset（case 0）
input_points: (2,3000,3) | target_points: (2,2048,3) | input range -0.817..0.815

# train_feasibility --mode single --steps 500
initial_loss 0.02483 → final_loss 0.00866（ratio 0.349）
mean_sample_chamfer 0.00792 | feasibility_pass = true

# baseline（4/2/2 cases, 2 epochs, seed 1024）
流程完整：导出 splits.json / summary.json / test_predictions/*.ply / visualizations/*.png
（极小配置 baseline_pass=false，属预期）
```

---

## 6. 常见问题

1. **`baseline_pass=false`？** 小配置/短训练不会过判据；用完整协议（80/10/10、12 epochs、3 seeds）再看 `summary.json`。
2. **为什么不能做 per-vertex L1 监督？** 现有心脏 GT 网格拓扑不共享；论文的逐顶点监督需要先把所有目标配准到共享拓扑。迁移版因此使用 Chamfer 类损失。
3. **想跑原版 TF 代码？** 需要单独的 Python 2.7 + TensorFlow 1.7 环境（老 CUDA），RTX 30/40 基本不可行；建议以 PyTorch 迁移版为准。
4. **显存**：baseline 默认 batch 8；24GB 卡上如 OOM 降到 4。
5. **你的 MNMS2**：先按第 2.3 节转换成 DeepSDF 风格数据根；无 label 的 `raw/mesh` 无法直接生成 LV/RV 两部分模板。

---

## 7. 路径速查

| 用途 | 路径 |
|---|---|
| 本项目（原版 + 已同步 PyTorch 迁移版） | `/home/nay/paper/MR-Net` |
| 旧复现/迁移版（含分析文档与结果） | `/home/nay/github/MR-Net` |
| 迁移版代码 | `/home/nay/paper/MR-Net/pytorch_mrnet/` |
| 论文/迁移分析（含第 7 部分差距说明） | `/home/nay/github/MR-Net/MR_Net_Analysis.md` |
| 可直接训练的数据（1331 例） | `/home/nay/github/DeepSDF/3d_data` |
| 你的原始数据 | `/home/nay/datasets/MNMS2` |
| 一键环境脚本 | `/home/nay/paper/setup_mrnet_env.sh` |
| conda 环境 | `mrnet`（python3.9 / torch2.2.0 / cu118） |
