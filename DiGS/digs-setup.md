完成 ✅ `/home/nay/paper/setup_digs_env.sh` 已创建，并用 `--recreate` **从零完整验证通过**（连微型 DiGS 训练都端到端跑通了）。

## 一、你的 scikit 安装问题（根因 + 解决）

`pip install -r requirements.txt` 报错不是 scikit-image 的问题，而是 **open3d==0.11.2 的元数据把废弃的 PyPI 占位包 `sklearn` 声明为依赖**。这个占位包现在拒绝安装，报：
```
The 'sklearn' PyPI package is deprecated, use 'scikit-learn' ...
```
解决：安装时设置 `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` 让占位包装过去，**再补装真正的 scikit-learn**——因为 open3d 0.11.2 导入时会 `from sklearn.neighbors import KDTree`（DiGS 代码本身不 import sklearn）。

## 二、digs 环境正确配置（梳理结论）

| 项 | 版本/做法 | 原因 |
|---|---|---|
| python | **3.7.9**（conda-forge） | DiGS 官方测试版本；依赖 pin 均为 py3.7 wheel |
| pip | `pip<24`（23.3.2） | pip 24 起不支持 py3.7 |
| requirements.txt | 原样安装（matplotlib 3.3.4 / scipy 1.6.2 / numpy 1.19.2 / open3d 0.11.2 / scikit_image 0.18.1 / trimesh 3.9.14 / kaleido 0.2.1 等） | 官方 pin |
| sklearn 占位 | `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` | 绕过 open3d 的坏依赖 |
| scikit-learn | **0.24.2**（补装） | open3d.ml 导入需要真正的 sklearn 模块 |
| protobuf | **3.20.3**（补装/降级） | tensorboardX 2.3 的老 pb2 与 protobuf 4.x 不兼容 |
| plotly | **5.9.0**（必装） | `utils/visualizations.py` 顶层 `import plotly`，训练脚本都会加载 |
| torch | **1.13.1+cu117**（替代 README 的 1.8/CUDA10.2） | py3.7 能装的最高 torch；1.8 没有 Ampere/RTX 30 系(sm_86) kernel，1.13.1 支持 3090 且代码兼容（已实测训练跑通） |
| plotly-orca | 可选（`--with-viz`） | 仅静态图导出用 |

## 三、一键脚本用法（任意 clone 路径）

```bash
# 放在 DiGS 仓库里 / 任意父目录 / 显式给路径，均可自动定位：
bash setup_digs_env.sh                    # 默认找 DiGS、建 digs 环境
bash setup_digs_env.sh /path/to/DiGS      # 指定仓库
bash setup_digs_env.sh --recreate         # 删掉重来
bash setup_digs_env.sh --dry-run          # 只预览
bash setup_digs_env.sh --with-viz         # 附加 plotly-orca（可选）
```

可覆盖项：`DIGS_ENV_NAME`、`DIGS_PYTHON`、`DIGS_TORCH_VERSION`、`DIGS_TORCH_CUDA`（默认 `117`；老卡可改 `113/102`，无 GPU 用 `cpu`）。

## 四、验证结果

- `--recreate` 从零一键成功：python 3.7.9 + 全部依赖 + torch 1.13.1+cu117，CUDA 可用（RTX 3090 Ti，真实算子运行通过）。
- 微型 DiGS 训练冒烟测试通过（circle 2D，loss 3458→28，模型正常保存），证明 torch 1.13 与代码兼容。

注意：如果换到更新的 GPU（如 H100/sm_90），py3.7 没有可用 torch，需要整体升级到 py3.8+ / torch 2.x 并重调 open3d 等 pin——那种情况再单独说。