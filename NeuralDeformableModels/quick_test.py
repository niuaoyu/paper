"""
快速测试训练流程 - 只用1个病例
"""
import subprocess
import sys
from pathlib import Path

cmd = [
    sys.executable,
    'train_simple.py',
    '--epochs', '5',
    '--batch-size', '2',
    '--max-train-samples', '1',
    '--max-val-samples', '1',
    '--stage', 'auto',
    '--stage-epochs', '1,1,1,1',
    '--ode-time', '0.05',
    '--ode-tol', '0.01',
    '--save-freq', '2',
    '--output-dir', './output/test_run',
    '--num-workers', '0'  # 避免多进程问题
]

repo_root = Path(__file__).resolve().parent

print("运行快速测试...", flush=True)
print("命令:", ' '.join(cmd), flush=True)
print("="*60, flush=True)

# 让依赖缺失、训练异常等错误传播给调用方/CI。
subprocess.run(cmd, cwd=repo_root, check=True)
