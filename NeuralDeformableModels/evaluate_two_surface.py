"""Evaluate and export the LV/RV two-surface NDM model.

The exported PLY files are point clouds and can be opened directly in MeshLab.
Metrics are reported in the normalized coordinate system used by training.
"""
import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chamfer_loss import chamfer_distance
from NeuralDeformableModel.dataset.heart_dataset import HeartDataset, get_train_val_split
from NeuralDeformableModel.model.model import NeuralDeformableModel


def load_ply_header_and_writer(path, points, colors=None):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'points must have shape [N, 3], got {points.shape}')
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape != points.shape:
            raise ValueError(f'colors must have shape {points.shape}, got {colors.shape}')

    with open(path, 'w', encoding='ascii') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        if colors is not None:
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        if colors is None:
            for x, y, z in points:
                f.write(f'{x:.7g} {y:.7g} {z:.7g}\n')
        else:
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f'{x:.7g} {y:.7g} {z:.7g} {r} {g} {b}\n')


def export_cloud(path, points, color):
    colors = np.repeat(np.asarray(color, dtype=np.uint8)[None, :], len(points), axis=0)
    load_ply_header_and_writer(path, points, colors)


def directed_mean_squared_distance(source, target, chunk_size=1024):
    """Mean squared distance from each source point to its nearest target point."""
    values = []
    for chunk in source.split(chunk_size, dim=1):
        values.append(torch.cdist(chunk, target).square().min(dim=2).values)
    return torch.cat(values, dim=1).mean()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def make_model(device, args):
    model = NeuralDeformableModel(
        zdim=512, time=args.ode_time, tol=args.ode_tol
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def resolve_stage(requested_stage, checkpoint):
    if requested_stage != 'auto':
        return requested_stage
    checkpoint_stage = checkpoint.get('stage')
    if checkpoint_stage:
        return checkpoint_stage
    raise ValueError(
        '该旧 checkpoint 未保存训练阶段。请显式传入 --stage '
        'translation/scale/rotation/shape/ode；不要默认按 ODE 验证。'
    )


def predict_two_surfaces(model, points, device, args):
    outputs = model(points.to(device))
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
    lv = sph[:, 5500:10500, :]  # LV endo (椭球)
    rv = sph[:, 10500:, :]      # RV

    # Match the two-surface training path in train_simple.py.
    from train_simple import quaternion_to_rotation_matrix, expand_lv_profile
    from train_simple import expand_rv_profile, expand_rv_x_profile

    stage = args.resolved_stage
    stage_index = {'translation': 0, 'scale': 1, 'rotation': 2, 'shape': 3, 'ode': 4}[stage]

    if stage_index >= 3:  # shape / ode
        lv_scale = torch.cat([
            expand_rv_profile(params['a1']), expand_rv_profile(params['a2']),
            expand_rv_profile(params['a3'])], dim=2)
        lv_offset = torch.cat([
            expand_rv_profile(params['e1']), expand_rv_profile(params['e2']),
            torch.zeros_like(expand_rv_profile(params['e1']))], dim=2)
        rv_scale = torch.cat([
            expand_rv_x_profile(params['a13'], params['a14']),
            expand_rv_profile(params['a23']), expand_rv_profile(params['a33'])], dim=2)
        rv_offset = torch.cat([
            expand_rv_profile(params['e13']), expand_rv_profile(params['e23']),
            torch.zeros_like(expand_rv_profile(params['e13']))], dim=2)
        lv = params['scale'] * lv_scale * lv + lv_offset
        rv = params['scale3'] * rv_scale * rv + rv_offset
    elif stage_index >= 1:  # scale / rotation
        lv = params['scale'] * lv
        rv = params['scale3'] * rv

    if stage_index >= 2:  # rotation / shape / ode
        lv_r, _ = quaternion_to_rotation_matrix(params['quaternion'])
        rv_r, _ = quaternion_to_rotation_matrix(params['quaternion3'])
        lv = torch.bmm(lv, lv_r.transpose(1, 2))
        rv = torch.bmm(rv, rv_r.transpose(1, 2))
    lv = lv + params['trans']
    rv = rv + params['trans3']

    if stage == 'ode':
        lv, _ = model.neural_mesh_forward1(code1, lv, None)
        rv, _ = model.neural_mesh_forward3(code3, rv, None)
    return lv, rv


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='/media/nay/f6b53612-1834-4e4e-aed5-19b00ed6cfc3/nay/3d-data')
    parser.add_argument('--output-dir', default='./output/evaluation')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--case-ids', nargs='*', type=int, default=None,
        help='仅评估指定病例，例如 --case-ids 16 1260；默认使用完整验证集',
    )
    parser.add_argument('--max-samples', type=int, default=-1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--ode-time', type=float, default=1.0)
    parser.add_argument('--ode-tol', type=float, default=0.001)
    parser.add_argument(
        '--stage',
        choices=['auto', 'translation', 'scale', 'rotation', 'shape', 'ode'],
        default='auto',
        help='默认读取新 checkpoint 保存的阶段；旧 checkpoint 必须显式指定',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.case_ids:
        val_ids = args.case_ids
    else:
        _, val_ids = get_train_val_split(args.data_root, args.train_ratio, args.seed)
    if args.max_samples > 0:
        val_ids = val_ids[:args.max_samples]
    dataset = HeartDataset(
        args.data_root, val_ids, ['ED', 'ES'], 5600, 3000,
        deterministic=True, seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    model, checkpoint = make_model(device, args)
    args.resolved_stage = resolve_stage(args.stage, checkpoint)
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': sha256_file(args.checkpoint),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'checkpoint_stage': checkpoint.get('stage'),
        'resolved_stage': args.resolved_stage,
        'checkpoint_val_loss': checkpoint.get('val_loss'),
        'data_root': str(Path(args.data_root).resolve()),
        'case_ids': [int(value) for value in val_ids],
        'phases': ['ED', 'ES'],
        'seed': args.seed,
        'sampling': {
            'input_points': 5600,
            'input_upsampling': 'preserve-all-original-points-and-append-duplicates',
            'gt_points_per_surface': 3000,
            'deterministic_per_case_phase': True,
        },
    }
    write_json(output_dir / 'manifest.json', manifest)
    logging.info(
        'checkpoint epoch=%s, stage=%s, val_loss=%s, device=%s, samples=%d',
        checkpoint.get('epoch'), args.resolved_stage, checkpoint.get('val_loss'),
        device, len(dataset),
    )
    if args.resolved_stage not in ('shape', 'ode'):
        logging.warning(
            '警告：当前 checkpoint 仅完成 %s 阶段。预测仍是平移/缩放/旋转后的固定基元，'
            '尚未学习逐纬度形状参数或 ODE 局部形变，不能作为最终重建结果。',
            args.resolved_stage,
        )

    records = []
    for batch in loader:
        points = batch['input'].to(device)
        lv_pred, rv_pred = predict_two_surfaces(model, points, device, args)
        lv_gt = batch['lv_gt'].to(device)
        rv_gt = batch['rv_gt'].to(device)
        for item in range(points.shape[0]):
            lv_loss, _ = chamfer_distance(
                lv_pred[item:item + 1], lv_gt[item:item + 1]
            )
            rv_loss, _ = chamfer_distance(
                rv_pred[item:item + 1], rv_gt[item:item + 1]
            )
            input_item = points[item:item + 1]
            union_gt = torch.cat([
                lv_gt[item:item + 1], rv_gt[item:item + 1]
            ], dim=1)
            input_to_gt = directed_mean_squared_distance(input_item, union_gt)
            case_id = int(batch['case_id'][item])
            phase = batch['phase'][item]
            stem = f'{case_id}_{phase}'
            export_cloud(output_dir / f'{stem}_input.ply', batch['input'][item].numpy(), [255, 255, 255])
            export_cloud(output_dir / f'{stem}_lv_gt.ply', batch['lv_gt'][item].numpy(), [0, 255, 0])
            export_cloud(output_dir / f'{stem}_rv_gt.ply', batch['rv_gt'][item].numpy(), [0, 128, 255])
            export_cloud(output_dir / f'{stem}_lv_pred.ply', lv_pred[item].cpu().numpy(), [255, 80, 80])
            export_cloud(output_dir / f'{stem}_rv_pred.ply', rv_pred[item].cpu().numpy(), [255, 180, 0])
            records.append({
                'case_id': case_id, 'phase': phase,
                'lv_chamfer_normalized_squared': float(lv_loss),
                'rv_chamfer_normalized_squared': float(rv_loss),
                'input_to_union_gt_directed_squared': float(input_to_gt),
            })
            logging.info(
                '%s: LV=%.6f RV=%.6f input->GT=%.6f', stem,
                records[-1]['lv_chamfer_normalized_squared'],
                records[-1]['rv_chamfer_normalized_squared'],
                records[-1]['input_to_union_gt_directed_squared'],
            )
            write_json(output_dir / 'records.partial.json', records)

    summary = {
        'checkpoint': str(args.checkpoint), 'stage': args.resolved_stage,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'checkpoint_val_loss': checkpoint.get('val_loss'),
        'records': records,
        'mean_lv_chamfer_normalized_squared': float(np.mean([r['lv_chamfer_normalized_squared'] for r in records])),
        'mean_rv_chamfer_normalized_squared': float(np.mean([r['rv_chamfer_normalized_squared'] for r in records])),
        'mean_input_to_union_gt_directed_squared': float(np.mean([r['input_to_union_gt_directed_squared'] for r in records])),
    }
    write_json(output_dir / 'metrics.json', summary)
    manifest['status'] = 'complete'
    manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['num_records'] = len(records)
    write_json(output_dir / 'manifest.json', manifest)
    logging.info('结果已写入 %s', output_dir)


if __name__ == '__main__':
    main()