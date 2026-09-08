完成 ✅ 已创建 `/home/nay/paper/setup_bipt_env.sh`，并且用 `--recreate` 从零完整跑通验证（模拟新服务器一键场景，最终 `bipt` 环境 python 3.10.21 / torch 2.4.1 CUDA 可用 / vtk 9.7.0 / bipt 导入 OK）。

## 用法（任意 clone 路径都行）

```bash
# ① 脚本放在 Bi-PT 仓库根目录里（推荐，随仓库走）——直接：
bash setup_bipt_env.sh

# ② 脚本放在别处、clone 在别的路径——显式传仓库路径：
bash setup_bipt_env.sh /path/to/Bi-PT

# ③ 脚本放在某个父目录，下面正好有 Bi-PT clone——也能自动找到（找含 "name: bipt" 的 environment.yml）
```

它自动 `cd` 到仓库根目录，执行官方安装步骤：

```
conda env create -f environment.yml
conda activate bipt
pip install -e .
pip install -r requirements-preprocess.txt
pip install -r requirements-eval.txt
```

## 附加选项

```bash
bash setup_bipt_env.sh --dry-run        # 只打印计划，不改任何东西
bash setup_bipt_env.sh --recreate       # 删掉已有 bipt 环境后完全重建
bash setup_bipt_env.sh --repo /path/to/Bi-PT
BIPT_ENV_NAME=myenv bash setup_bipt_env.sh   # 自定义环境名
```

## 脚本自动处理的两个仓库本身的问题

1. **vtk 版本冲突**：`environment.yml` 里 conda 可能解析出 vtk 9.2.x，但 `requirements-preprocess.txt` 要求 **vtk>=9.3**，而 pip 无法覆盖 conda(distutils) 装的包（会直接报 `Cannot uninstall ... distutils installed project`）。脚本检测到 conda 版 vtk<9.3 时先 `conda remove vtk`，再让 pip 装 vtk 9.7+；若已是 ≥9.3 或 pip 管理则自动跳过。
2. **conda activate 与 `set -u` 冲突**：conda 的 cuda-nvcc 激活钩子引用未定义变量，所以脚本用 `set -eo pipefail`（不用 `-u`）。

## 注意事项

- `environment.yml` 写死 `pytorch-cuda=12.1`：换机器时若 CUDA 驱动/工具链不同，先按文件头注释改这一行再跑。
- `requirements-eval.txt` 提到的 `torch-mesh-isect`（SI 指标用的 CUDA 扩展，不在 PyPI）脚本不自动装——它只在 `--compute-si` 时才需要，缺省时该指标显示 NaN，不影响其它指标。需要的话手动按它 README 装。

以后新服务器上只要 `git clone` 完 Bi-PT，把脚本放进仓库根目录（或直接拷到任意位置加仓库路径参数）执行一次即可。