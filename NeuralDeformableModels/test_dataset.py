"""
测试 HeartDataset 是否能正常加载数据
"""
import sys
import torch
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from NeuralDeformableModel.dataset.heart_dataset import HeartDataset, get_train_val_split


def test_dataset():
    print("="*60)
    print("测试 HeartDataset")
    print("="*60)
    
    # 使用病例0进行测试
    dataset = HeartDataset(
        root='/home/nay/github/DeepSDF/3d_data',
        case_ids=[0],
        phases=['ED', 'ES'],
        num_input_points=5600,
        num_gt_points=3000
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 加载第一个样本
    print("\n加载第一个样本...")
    sample = dataset[0]
    
    print(f"\n样本信息:")
    print(f"  病例ID: {sample['case_id']}")
    print(f"  时相: {sample['phase']}")
    print(f"  输入点云 shape: {sample['input'].shape}")
    print(f"  LV GT shape: {sample['lv_gt'].shape}")
    print(f"  RV GT shape: {sample['rv_gt'].shape}")
    
    print(f"\n坐标范围:")
    print(f"  输入点云: [{sample['input'].min():.4f}, {sample['input'].max():.4f}]")
    print(f"  LV GT: [{sample['lv_gt'].min():.4f}, {sample['lv_gt'].max():.4f}]")
    print(f"  RV GT: [{sample['rv_gt'].min():.4f}, {sample['rv_gt'].max():.4f}]")
    
    # 测试batch
    print("\n测试 DataLoader...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    
    print(f"  Batch input shape: {batch['input'].shape}")
    print(f"  Batch LV GT shape: {batch['lv_gt'].shape}")
    print(f"  Batch RV GT shape: {batch['rv_gt'].shape}")
    
    print("\n✓ Dataset 测试通过!")
    return True


def test_split():
    print("\n" + "="*60)
    print("测试数据划分")
    print("="*60)
    
    train_ids, val_ids = get_train_val_split(train_ratio=0.8, seed=42)
    
    print(f"\n训练集病例数: {len(train_ids)}")
    print(f"验证集病例数: {len(val_ids)}")
    print(f"前5个训练病例: {train_ids[:5]}")
    print(f"前5个验证病例: {val_ids[:5]}")
    
    print("\n✓ Split 测试通过!")
    return True


if __name__ == '__main__':
    try:
        test_dataset()
        test_split()
        print("\n" + "="*60)
        print("所有测试通过! 🎉")
        print("="*60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
