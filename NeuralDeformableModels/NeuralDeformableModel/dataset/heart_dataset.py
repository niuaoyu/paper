"""
Heart Dataset Adapter for NDM
适配 DeepSDF 格式数据到 NDM 训练
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class HeartDataset(Dataset):
    """
    加载 /home/nay/github/DeepSDF/3d_data 格式数据
    """
    
    def __init__(self, 
                 root='/home/nay/github/DeepSDF/3d_data',
                 case_ids=None,
                 phases=['ED', 'ES'],
                 num_input_points=5600,
                 num_gt_points=3000,
                 use_normalization='unit_sphere',
                 deterministic=False,
                 seed=42):
        """
        Args:
            root: 数据根目录
            case_ids: 病例ID列表，None则自动扫描
            phases: 要使用的时相
            num_input_points: 输入点云点数
            num_gt_points: GT采样点数
            use_normalization: 'unit_sphere' 或 'aabb'
            deterministic: 是否为每个病例/时相使用固定采样（验证/评估应开启）
            seed: 固定采样的基础随机种子
        """
        self.root = Path(root)
        self.phases = phases
        self.num_input_points = num_input_points
        self.num_gt_points = num_gt_points
        self.use_normalization = use_normalization
        self.deterministic = deterministic
        self.seed = seed
        
        # 扫描所有可用病例
        if case_ids is None:
            mesh_dir = self.root / 'mesh'
            case_ids = sorted([int(d.name) for d in mesh_dir.iterdir() if d.is_dir()])
        
        self.samples = []
        for case_id in case_ids:
            for phase in phases:
                # 检查必要文件是否存在
                mesh_file = self.root / 'mesh' / str(case_id) / f'{phase}.ply'
                point_file = self.root / 'points' / str(case_id) / f'{phase}.ply'
                if mesh_file.exists() and point_file.exists():
                    self.samples.append((case_id, phase))
        
        print(f'HeartDataset: {len(self.samples)} samples from {len(case_ids)} cases')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        case_id, phase = self.samples[idx]
        rng = self._rng_for_sample(case_id, phase)
        
        # 加载输入点云
        point_file = self.root / 'points' / str(case_id) / f'{phase}.ply'
        input_pts = self._load_ply_vertices(point_file)
        
        # 加载归一化参数
        norm_params = self._load_normalization(case_id)
        
        # 应用归一化
        input_pts = self._apply_normalization(input_pts, norm_params)
        
        # 重采样到指定点数
        if len(input_pts) != self.num_input_points:
            input_pts = self._resample_points(input_pts, self.num_input_points, rng)
        
        lv_gt, rv_gt = self._load_ground_truth(case_id, phase, rng)
        
        return {
            'input': torch.from_numpy(input_pts).float(),
            'lv_gt': torch.from_numpy(lv_gt).float(),
            'rv_gt': torch.from_numpy(rv_gt).float(),
            'case_id': case_id,
            'phase': phase
        }

    def _rng_for_sample(self, case_id, phase):
        if not self.deterministic:
            return np.random
        phase_offset = sum(ord(char) for char in phase)
        sample_seed = (self.seed * 1000003 + int(case_id) * 97 + phase_offset) % (2 ** 32)
        return np.random.RandomState(sample_seed)

    def _load_ground_truth(self, case_id, phase, rng):
        """加载两表面 GT：从 mesh PLY 取 label==1 (LV) 和 label==2 (RV)。

        mesh 与 points 位于同一原始坐标系，需要应用相同的 pose + unit_sphere
        归一化才能与 canonical 空间中的 input 对齐。
        """
        mesh_file = self.root / 'mesh' / str(case_id) / f'{phase}.ply'
        mesh_verts, mesh_labels = self._load_ply_with_labels(mesh_file)

        # 对 mesh GT 做与 input 相同的归一化
        norm_params = self._load_normalization(case_id)
        mesh_verts = self._apply_normalization(mesh_verts, norm_params)

        lv_verts = mesh_verts[mesh_labels == 1]
        rv_verts = mesh_verts[mesh_labels == 2]
        lv_gt = self._sample_surface(lv_verts, self.num_gt_points, rng)
        rv_gt = self._sample_surface(rv_verts, self.num_gt_points, rng)
        return lv_gt, rv_gt
    
    def _load_ply_vertices(self, path):
        """加载PLY顶点坐标"""
        with open(path, 'rb') as f:
            counts = {}
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('element '):
                    _, name, count = line.split()
                    counts[name] = int(count)
                elif line == 'end_header':
                    break
            
            dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                            ('label', 'u1'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            verts = np.fromfile(f, dtype=dtype, count=counts['vertex'])
        
        return np.column_stack([verts['x'], verts['y'], verts['z']]).astype(np.float64)
    
    def _load_ply_with_labels(self, path):
        """加载PLY顶点和标签"""
        with open(path, 'rb') as f:
            counts = {}
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('element '):
                    _, name, count = line.split()
                    counts[name] = int(count)
                elif line == 'end_header':
                    break
            
            dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                            ('label', 'u1'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            verts = np.fromfile(f, dtype=dtype, count=counts['vertex'])
        
        coords = np.column_stack([verts['x'], verts['y'], verts['z']]).astype(np.float64)
        labels = verts['label']
        return coords, labels
    
    def _load_normalization(self, case_id):
        """加载归一化参数"""
        norm_dir = self.root / 'norm' / str(case_id)
        
        # 加载pose
        pose = np.load(norm_dir / 'pose.npy')
        
        # 加载scale和offset
        if self.use_normalization == 'unit_sphere':
            npz = np.load(norm_dir / 'unit_sphere.npz')
        else:
            npz = np.load(norm_dir / 'aabb.npz')
        
        offset = npz['offset']
        scale = float(npz['scale'])
        
        return {'pose': pose, 'offset': offset, 'scale': scale}
    
    def _apply_normalization(self, points, norm_params):
        """应用归一化变换"""
        pose = norm_params['pose']
        offset = norm_params['offset']
        scale = norm_params['scale']
        
        # 齐次坐标
        pts_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        
        # 应用pose变换
        pts_transformed = (pose @ pts_h.T).T[:, :3]
        
        # 应用scale和offset
        pts_normalized = scale * (pts_transformed + offset)
        
        return pts_normalized
    
    def _resample_points(self, points, target_num, rng=np.random):
        """重采样；上采样保留全部原始点，仅追加所需重复点。"""
        if len(points) == target_num:
            return points
        
        if len(points) < target_num:
            extra_indices = rng.choice(
                len(points), target_num - len(points), replace=True
            )
            return np.concatenate([points, points[extra_indices]], axis=0)
        
        # FPS下采样
        indices = self._farthest_point_sample(points, target_num, rng)
        return points[indices]
    
    def _farthest_point_sample(self, points, n_samples, rng=np.random):
        """最远点采样"""
        n_points = len(points)
        selected = np.zeros(n_samples, dtype=np.int32)
        distances = np.ones(n_points) * 1e10
        
        # 随机选择第一个点
        farthest = rng.randint(0, n_points)
        
        for i in range(n_samples):
            selected[i] = farthest
            centroid = points[farthest]
            dist = np.sum((points - centroid) ** 2, axis=1)
            mask = dist < distances
            distances[mask] = dist[mask]
            farthest = np.argmax(distances)
        
        return selected
    
    def _sample_surface(self, vertices, n_samples, rng=np.random):
        """从表面顶点均匀采样"""
        if len(vertices) <= n_samples:
            # 不足时重复
            indices = rng.choice(len(vertices), n_samples, replace=True)
        else:
            # 随机采样
            indices = rng.choice(len(vertices), n_samples, replace=False)
        
        return vertices[indices]


def get_train_val_split(root='/home/nay/github/DeepSDF/3d_data', 
                         train_ratio=0.8, 
                         seed=42):
    """
    生成训练/验证划分
    """
    mesh_dir = Path(root) / 'mesh'
    case_ids = sorted([int(d.name) for d in mesh_dir.iterdir() if d.is_dir()])
    
    np.random.seed(seed)
    np.random.shuffle(case_ids)
    
    split_idx = int(len(case_ids) * train_ratio)
    train_ids = case_ids[:split_idx]
    val_ids = case_ids[split_idx:]
    
    return train_ids, val_ids
