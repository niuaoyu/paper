"""简化版 NDM 训练脚本 - LV/RV 两表面监督模式。"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from NeuralDeformableModel.dataset.heart_dataset import HeartDataset, get_train_val_split
from NeuralDeformableModel.model.model import NeuralDeformableModel
from chamfer_loss import chamfer_distance
from test_input_gt_alignment import audit_sample


def setup_logging(output_dir):
    """设置日志"""
    log_file = output_dir / f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def quaternion_loss(p1, p2):
    """参数一致性损失"""
    return torch.mean((p1 - p2) ** 2)


def smoothing_loss(params):
    """参数平滑损失"""
    if params.shape[1] <= 1:
        return torch.tensor(0.0, device=params.device)
    diff = params[:, 1:, :] - params[:, :-1, :]
    return torch.mean(diff ** 2)


def displacement_smoothing_loss(original, deformed):
    """位移平滑损失"""
    displacement = deformed - original
    if displacement.shape[1] <= 1:
        return torch.tensor(0.0, device=displacement.device)
    diff = displacement[:, 1:, :] - displacement[:, :-1, :]
    return torch.mean(diff ** 2)


def quaternion_to_rotation_matrix(quaternion):
    """将 (w, x, y, z) 四元数安全地转换为批量旋转矩阵。"""
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=1, keepdim=True
    ).clamp_min(1e-8)
    qw, qx, qy, qz = quaternion.unbind(dim=1)

    row0 = torch.stack([
        1 - 2 * (qy.square() + qz.square()),
        2 * (qx * qy - qz * qw),
        2 * (qx * qz + qy * qw),
    ], dim=1)
    row1 = torch.stack([
        2 * (qx * qy + qz * qw),
        1 - 2 * (qx.square() + qz.square()),
        2 * (qy * qz - qx * qw),
    ], dim=1)
    row2 = torch.stack([
        2 * (qx * qz - qy * qw),
        2 * (qy * qz + qx * qw),
        1 - 2 * (qx.square() + qy.square()),
    ], dim=1)
    return torch.stack([row0, row1, row2], dim=1), quaternion


def expand_lv_profile(profile):
    """把 50 个纬度参数扩展到带 5 圈顶盖的 55x100 LV 模板。"""
    profile = torch.cat([profile, profile[:, -1:, :].expand(-1, 5, -1)], dim=1)
    return profile.expand(-1, -1, 100).reshape(profile.shape[0], 5500, 1)


def expand_rv_profile(profile):
    """把 50 个纬度参数扩展到 50x100 RV 模板。"""
    return profile.expand(-1, -1, 100).reshape(profile.shape[0], 5000, 1)


def expand_rv_x_profile(outer_profile, inner_profile):
    """RV 外壁和内壁分别占每圈的 60 与 40 个模板点。"""
    outer = outer_profile.expand(-1, -1, 60)
    inner = inner_profile.expand(-1, -1, 40)
    return torch.cat([outer, inner], dim=2).reshape(outer_profile.shape[0], 5000, 1)


class SimpleTrainer:
    """简化的训练器"""
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 创建模型 - 两表面模式
        self.logger.info("创建模型...")
        self.model = NeuralDeformableModel(
            zdim=512,
            time=args.ode_time,
            tol=args.ode_tol
        ).to(self.device)
        
        # 模板点数 – 两表面训练用 LV endo（椭球有方向）+ RV
        # LV epi = 5500 (球体, sph[0:5500]), LV endo = 5000 (sph[5500:10500])
        self.n_lv = 5000   # LV endo template points（椭球, z轴拉长）
        self.n_endo = 5000  # 保留兼容
        self.n_rv = 5000    # RV template points

        self.stage_order = ['translation', 'scale', 'rotation', 'shape', 'ode']
        self.stage_epochs = self._parse_stage_epochs(args.stage_epochs)
        
        # 创建数据集
        self.logger.info("加载数据集...")
        train_ids, val_ids = get_train_val_split(
            root=args.data_root,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        
        # 限制训练样本数用于快速测试
        if args.max_train_samples > 0:
            train_ids = train_ids[:min(args.max_train_samples, len(train_ids))]
        if args.max_val_samples > 0:
            val_ids = val_ids[:min(args.max_val_samples, len(val_ids))]
        
        self.train_dataset = HeartDataset(
            root=args.data_root,
            case_ids=train_ids,
            phases=['ED', 'ES'],
            num_input_points=5600,
            num_gt_points=3000,
        )
        
        self.val_dataset = HeartDataset(
            root=args.data_root,
            case_ids=val_ids,
            phases=['ED', 'ES'],
            num_input_points=5600,
            num_gt_points=3000,
            deterministic=True,
            seed=args.seed,
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )

        self.ode_train_loader = None
        self.ode_val_loader = None
        if args.ode_batch_size != args.batch_size:
            self.ode_train_loader = DataLoader(
                self.train_dataset,
                batch_size=args.ode_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            )
            self.ode_val_loader = DataLoader(
                self.val_dataset,
                batch_size=args.ode_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            )
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.lr_step,
            gamma=args.lr_gamma
        )
        
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.last_grad_norm = 0.0
        
        # 加载checkpoint
        if args.resume:
            self.load_checkpoint(args.resume)

    def _parse_stage_epochs(self, value):
        durations = [int(item.strip()) for item in value.split(',')]
        if len(durations) != 4 or any(item < 0 for item in durations):
            raise ValueError('--stage-epochs 必须包含 4 个非负整数，例如 5,5,10,20')
        return durations

    def get_stage(self, epoch):
        """返回当前训练阶段；auto 按平移、缩放、旋转、形状、ODE 推进。"""
        if self.args.stage != 'auto':
            return self.args.stage
        boundary = 0
        for stage, duration in zip(self.stage_order[:-1], self.stage_epochs):
            boundary += duration
            if epoch < boundary:
                return stage
        return 'ode'

    def _apply_shape_parameters(self, lv_pc, rv_pc, params):
        """LV endo 模板 (50×100=5000, 无顶盖) 与 RV 模板同结构。"""
        lv_scale = torch.cat([
            expand_rv_profile(params['a1']),
            expand_rv_profile(params['a2']),
            expand_rv_profile(params['a3']),
        ], dim=2)
        lv_offset = torch.cat([
            expand_rv_profile(params['e1']),
            expand_rv_profile(params['e2']),
            torch.zeros_like(expand_rv_profile(params['e1'])),
        ], dim=2)

        rv_scale = torch.cat([
            expand_rv_x_profile(params['a13'], params['a14']),
            expand_rv_profile(params['a23']),
            expand_rv_profile(params['a33']),
        ], dim=2)
        rv_offset = torch.cat([
            expand_rv_profile(params['e13']),
            expand_rv_profile(params['e23']),
            torch.zeros_like(expand_rv_profile(params['e13'])),
        ], dim=2)
        return (
            lv_scale * lv_pc,
            rv_scale * rv_pc,
            lv_offset,
            rv_offset,
        )

    def predict_surfaces(self, points, epoch=0):
        """按指定阶段生成 LV/RV，并返回阶段正则项。"""
        outputs = self.model(points)
        code1, _, code3 = outputs[:3]
        params = {
            'trans': outputs[3], 'quaternion': outputs[4], 'scale': outputs[5],
            'a1': outputs[6], 'a2': outputs[7], 'a3': outputs[8],
            'e1': outputs[9], 'e2': outputs[10],
            'trans3': outputs[19], 'quaternion3': outputs[20], 'scale3': outputs[21],
            'a13': outputs[22], 'a23': outputs[23], 'a33': outputs[24],
            'e13': outputs[25], 'e23': outputs[26], 'a14': outputs[27],
        }
        sph = outputs[28]
        lv_pc = sph[:, 5500:10500, :]          # LV endo (椭球, 有方向)
        rv_pc = sph[:, 10500:, :]              # RV
        stage = self.get_stage(epoch)
        stage_index = self.stage_order.index(stage)
        regularization = points.new_zeros(())

        if stage_index >= self.stage_order.index('shape'):
            lv_pc, rv_pc, lv_offset, rv_offset = self._apply_shape_parameters(
                lv_pc, rv_pc, params
            )
            lv_pc = params['scale'] * lv_pc + lv_offset
            rv_pc = params['scale3'] * rv_pc + rv_offset
            profile_names = [
                'a1', 'a2', 'a3', 'e1', 'e2',
                'a13', 'a14', 'a23', 'a33', 'e13', 'e23',
            ]
            regularization = regularization + sum(
                smoothing_loss(params[name]) for name in profile_names
            )
        elif stage_index >= self.stage_order.index('scale'):
            lv_pc = params['scale'] * lv_pc
            rv_pc = params['scale3'] * rv_pc

        if stage_index >= self.stage_order.index('rotation'):
            lv_rotation, lv_quaternion = quaternion_to_rotation_matrix(params['quaternion'])
            rv_rotation, rv_quaternion = quaternion_to_rotation_matrix(params['quaternion3'])
            lv_pc = torch.bmm(lv_pc, lv_rotation.transpose(1, 2))
            rv_pc = torch.bmm(rv_pc, rv_rotation.transpose(1, 2))
            regularization = regularization + quaternion_loss(lv_quaternion, rv_quaternion)

        lv_pc = lv_pc + params['trans']
        rv_pc = rv_pc + params['trans3']

        if stage == 'ode':
            if self.args.ode_detach_base:
                # 课程训练最后阶段只优化 ODE，避免形状正则继续更新基础分支。
                regularization = points.new_zeros(())
            ode_code1 = code1.detach() if self.args.ode_detach_base else code1
            ode_code3 = code3.detach() if self.args.ode_detach_base else code3
            ode_lv = lv_pc.detach() if self.args.ode_detach_base else lv_pc
            ode_rv = rv_pc.detach() if self.args.ode_detach_base else rv_pc
            lv_pred, _ = self.model.neural_mesh_forward1(ode_code1, ode_lv, None)
            rv_pred, _ = self.model.neural_mesh_forward3(ode_code3, ode_rv, None)
            regularization = regularization + self.args.ode_displacement_weight * (
                torch.mean((lv_pred - ode_lv).square()) +
                torch.mean((rv_pred - ode_rv).square())
            )
            regularization = regularization + self.args.ode_smooth_weight * (
                displacement_smoothing_loss(ode_lv, lv_pred) +
                displacement_smoothing_loss(ode_rv, rv_pred)
            )
            lv_pc, rv_pc = lv_pred, rv_pred

        return lv_pc, rv_pc, regularization, stage
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_lv_loss = 0
        total_rv_loss = 0
        
        train_loader = (
            self.ode_train_loader
            if self.get_stage(epoch) == 'ode' and self.ode_train_loader is not None
            else self.train_loader
        )
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            points = batch['input'].to(self.device, non_blocking=True)
            lv_gt = batch['lv_gt'].to(self.device, non_blocking=True)
            rv_gt = batch['rv_gt'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            lv_pred, rv_pred, regularization, stage = self.predict_surfaces(points, epoch)
            
            # 计算Chamfer损失
            lv_loss, _ = chamfer_distance(lv_pred, lv_gt)
            rv_loss, _ = chamfer_distance(rv_pred, rv_gt)

            # Chamfer 在模板与目标相距较远时可能把平移头推入 Tanh 饱和区。
            # 显式中心约束为两个平移分支提供稳定、方向明确的梯度。
            centroid_loss = (
                torch.mean((lv_pred.mean(dim=1) - lv_gt.mean(dim=1)).square()) +
                torch.mean((rv_pred.mean(dim=1) - rv_gt.mean(dim=1)).square())
            )
            
            # 总损失
            loss = (
                lv_loss + rv_loss +
                self.args.centroid_weight * centroid_loss +
                self.args.reg_weight * regularization
            )
            
            # 反向传播
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip
            )
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    f'Epoch {epoch} batch {batch_idx}: 检测到非有限梯度，请降低学习率'
                )
            self.last_grad_norm = float(grad_norm)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_lv_loss += lv_loss.item()
            total_rv_loss += rv_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lv': f'{lv_loss.item():.4f}',
                'rv': f'{rv_loss.item():.4f}',
                'grad': f'{self.last_grad_norm:.2f}',
                'stage': stage,
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_lv = total_lv_loss / len(train_loader)
        avg_rv = total_rv_loss / len(train_loader)
        
        return avg_loss, avg_lv, avg_rv
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_lv_loss = 0
        total_rv_loss = 0
        
        val_loader = (
            self.ode_val_loader
            if self.get_stage(epoch) == 'ode' and self.ode_val_loader is not None
            else self.val_loader
        )
        for batch in tqdm(val_loader, desc='Validating'):
            points = batch['input'].to(self.device, non_blocking=True)
            lv_gt = batch['lv_gt'].to(self.device, non_blocking=True)
            rv_gt = batch['rv_gt'].to(self.device, non_blocking=True)
            
            lv_pred, rv_pred, regularization, _ = self.predict_surfaces(points, epoch)
            
            # 计算损失
            lv_loss, _ = chamfer_distance(lv_pred, lv_gt)
            rv_loss, _ = chamfer_distance(rv_pred, rv_gt)
            centroid_loss = (
                torch.mean((lv_pred.mean(dim=1) - lv_gt.mean(dim=1)).square()) +
                torch.mean((rv_pred.mean(dim=1) - rv_gt.mean(dim=1)).square())
            )
            loss = (
                lv_loss + rv_loss +
                self.args.centroid_weight * centroid_loss +
                self.args.reg_weight * regularization
            )
            
            total_loss += loss.item()
            total_lv_loss += lv_loss.item()
            total_rv_loss += rv_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_lv = total_lv_loss / len(val_loader)
        avg_rv = total_rv_loss / len(val_loader)
        
        return avg_loss, avg_lv, avg_rv
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'stage': self.get_stage(epoch),
            'args': vars(self.args).copy(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }
        
        # 保存最新的
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存周期性checkpoint
        if (epoch + 1) % self.args.save_freq == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch:04d}.pth'
            torch.save(checkpoint, epoch_path)
        
        # 保存最佳
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f'✓ 保存最佳模型: val_loss={val_loss:.6f}')
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        self.logger.info(f'加载checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not self.args.reset_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            self.logger.info('仅恢复模型权重；优化器和学习率调度器已重置')
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = (
            float('inf') if self.args.reset_optimizer
            else checkpoint.get('best_val_loss', float('inf'))
        )
        
        self.logger.info(f'从epoch {self.start_epoch} 继续训练')
    
    def train(self):
        """主训练循环"""
        self.logger.info("开始训练...")
        self.logger.info(f"训练样本: {len(self.train_dataset)}")
        self.logger.info(f"验证样本: {len(self.val_dataset)}")
        self.logger.info(f"设备: {self.device}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练
            train_loss, train_lv, train_rv = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_lv, val_rv = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 日志
            self.logger.info(
                f'Epoch {epoch:04d} [{self.get_stage(epoch)}] | '
                f'Train: {train_loss:.6f} (LV:{train_lv:.6f} RV:{train_rv:.6f}) | '
                f'Val: {val_loss:.6f} (LV:{val_lv:.6f} RV:{val_rv:.6f}) | '
                f'LR: {self.scheduler.get_last_lr()[0]:.2e}'
            )
            
            # 保存checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        self.logger.info("训练完成!")
        self.logger.info(f"最佳验证损失: {self.best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='NDM 简化训练脚本')
    
    # 数据
    parser.add_argument('--data-root', type=str, 
                       default='/media/nay/f6b53612-1834-4e4e-aed5-19b00ed6cfc3/nay/3d-data',
                       help='数据根目录')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 训练
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='translation/scale/rotation/shape 阶段批大小')
    parser.add_argument('--ode-batch-size', type=int, default=1,
                       help='ODE 阶段批大小；24GB 显存建议从 1 开始')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='全局梯度范数裁剪阈值，防止编码器/平移头突然发散')
    parser.add_argument('--centroid-weight', type=float, default=1.0,
                       help='LV/RV预测与GT中心对齐损失权重')
    parser.add_argument('--lr-step', type=int, default=20,
                       help='学习率衰减步长')
    parser.add_argument('--lr-gamma', type=float, default=0.5,
                       help='学习率衰减系数')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                       help='每个 DataLoader worker 预取的 batch 数')
    
    # 模型
    parser.add_argument('--ode-time', type=float, default=1.0,
                       help='ODE时间')
    parser.add_argument('--ode-tol', type=float, default=0.001,
                       help='ODE容差')
    parser.add_argument('--stage', type=str, default='auto',
                       choices=['auto', 'translation', 'scale', 'rotation', 'shape', 'ode'],
                       help='固定训练阶段，或使用auto课程训练')
    parser.add_argument('--stage-epochs', type=str, default='5,5,10,20',
                       help='auto模式中平移、缩放、旋转、形状阶段的epoch数')
    parser.add_argument('--reg-weight', type=float, default=0.01,
                       help='形状参数和旋转一致性正则权重')
    parser.add_argument('--ode-displacement-weight', type=float, default=0.1,
                       help='ODE位移幅度正则权重')
    parser.add_argument('--ode-smooth-weight', type=float, default=0.05,
                       help='ODE位移平滑正则权重')
    parser.add_argument('--ode-detach-base', action=argparse.BooleanOptionalAction,
                       default=True,
                       help='ODE阶段冻结基础仿射/形状分支，与原论文训练方式一致')
    
    # 输出
    parser.add_argument('--output-dir', type=str, 
                       default='./output/simple_train',
                       help='输出目录')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='保存频率(epoch)')
    parser.add_argument('--resume', type=str, default='',
                       help='恢复训练的checkpoint路径')
    parser.add_argument('--reset-optimizer', action='store_true',
                       help='恢复模型权重但重置优化器和调度器，用于从发散前的最佳模型恢复')
    
    # 快速测试
    parser.add_argument('--max-train-samples', type=int, default=-1,
                       help='最大训练样本数(-1表示全部)')
    parser.add_argument('--max-val-samples', type=int, default=-1,
                       help='最大验证样本数(-1表示全部)')
    parser.add_argument('--alignment-check-cases', type=int, nargs='+', default=[16],
                       help='训练前检查输入/GT方向的代表病例')
    parser.add_argument('--max-alignment-angle', type=float, default=20.0,
                       help='允许的输入与GT LV-RV XY方向夹角（度）')
    parser.add_argument('--skip-alignment-check', action='store_true',
                       help='仅供诊断使用：跳过采集对齐门禁')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 创建输出目录和日志
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    # 打印配置
    logger.info("="*60)
    logger.info("NDM 简化训练")
    logger.info("="*60)
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("="*60)

    if not args.skip_alignment_check:
        alignment_dataset = HeartDataset(
            root=args.data_root,
            case_ids=args.alignment_check_cases,
            phases=['ED', 'ES'],
            deterministic=True,
            seed=args.seed,
        )
        alignment_reports = [
            audit_sample(
                alignment_dataset, case_id, phase, args.max_alignment_angle
            )
            for case_id, phase in alignment_dataset.samples
        ]
        failures = [
            report for report in alignment_reports if report['failures']
        ]
        for report in alignment_reports:
            logger.info(
                '采集对齐 %s/%s: XY方向夹角 %.2f°',
                report['case_id'], report['phase'],
                report['direction_angle_xy_degrees'],
            )
        if failures:
            details = '; '.join(
                f"{item['case_id']}/{item['phase']}="
                f"{item['direction_angle_xy_degrees']:.2f}°"
                for item in failures
            )
            raise RuntimeError(
                '输入点采集与GT方向门禁失败，停止训练。请先重新生成点数据并运行 '
                f'test_input_gt_alignment.py。失败样本: {details}'
            )
    
    # 开始训练
    trainer = SimpleTrainer(args, logger)
    trainer.train()


if __name__ == '__main__':
    main()
