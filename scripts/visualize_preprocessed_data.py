#!/usr/bin/env python3
"""
可视化预处理的UDF/SDF数据
显示点云和距离场值的分布
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_and_visualize(npz_path, title_prefix=""):
    """加载并可视化一个npz文件"""
    data = np.load(npz_path)

    # 检查数据键
    print(f"\n文件: {npz_path.name}")
    print(f"数据键: {list(data.keys())}")

    points = None
    values = None

    # 尝试不同的键名
    if 'poisson_grid_points' in data:
        points = data['poisson_grid_points']
    elif 'points' in data:
        points = data['points']

    if 'sdf_values' in data:
        values = data['sdf_values']
        value_type = "SDF"
    elif 'udf_values' in data:
        values = data['udf_values']
        value_type = "UDF"
    elif 'values' in data:
        values = data['values']
        value_type = "Distance"

    if points is None or values is None:
        print(f"警告: 无法找到点或值数据")
        return None

    print(f"点云形状: {points.shape}")
    print(f"{value_type}值形状: {values.shape}")
    print(f"{value_type}值范围: [{values.min():.4f}, {values.max():.4f}]")
    print(f"{value_type}值均值: {values.mean():.4f}, 标准差: {values.std():.4f}")

    return {
        'points': points,
        'values': values,
        'value_type': value_type,
        'filename': npz_path.name
    }

def visualize_multiple_samples(npz_dir, output_dir, num_samples=3):
    """可视化多个样本"""
    npz_dir = Path(npz_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有npz文件
    npz_files = sorted(npz_dir.glob('*.npz'))[:num_samples]

    if not npz_files:
        print(f"错误: 在 {npz_dir} 中没有找到npz文件")
        return

    print(f"\n找到 {len(npz_files)} 个文件")

    # 为每个文件创建可视化
    for npz_file in npz_files:
        data_dict = load_and_visualize(npz_file)
        if data_dict is None:
            continue

        points = data_dict['points']
        values = data_dict['values']
        value_type = data_dict['value_type']
        filename = data_dict['filename']

        # 创建图形
        fig = plt.figure(figsize=(16, 10))

        # 1. 3D点云（按距离值着色）
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=values, cmap='RdYlBu_r', s=1, alpha=0.6)
        ax1.set_title(f'{value_type} Values (3D View)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        plt.colorbar(scatter, ax=ax1, label=f'{value_type} Value')

        # 2. XY平面投影
        ax2 = fig.add_subplot(2, 3, 2)
        scatter2 = ax2.scatter(points[:, 0], points[:, 1],
                              c=values, cmap='RdYlBu_r', s=1, alpha=0.3)
        ax2.set_title('XY Plane Projection')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_aspect('equal')
        plt.colorbar(scatter2, ax=ax2, label=f'{value_type} Value')

        # 3. XZ平面投影
        ax3 = fig.add_subplot(2, 3, 3)
        scatter3 = ax3.scatter(points[:, 0], points[:, 2],
                              c=values, cmap='RdYlBu_r', s=1, alpha=0.3)
        ax3.set_title('XZ Plane Projection')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_aspect('equal')
        plt.colorbar(scatter3, ax=ax3, label=f'{value_type} Value')

        # 4. 距离值直方图
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.hist(values, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_title(f'{value_type} Value Distribution')
        ax4.set_xlabel(f'{value_type} Value')
        ax4.set_ylabel('Frequency')
        ax4.axvline(0, color='r', linestyle='--', label='Zero Level')
        ax4.legend()

        # 5. 距离值统计
        ax5 = fig.add_subplot(2, 3, 5)
        stats_text = f"""
        文件: {filename}
        类型: {value_type}

        点数: {len(points):,}

        {value_type}范围:
        Min: {values.min():.4f}
        Max: {values.max():.4f}
        Mean: {values.mean():.4f}
        Std: {values.std():.4f}
        Median: {np.median(values):.4f}

        点云范围:
        X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]
        Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]
        Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]
        """
        ax5.text(0.1, 0.5, stats_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='center', family='monospace')
        ax5.axis('off')

        # 6. 表面点分布（接近零的点，仅对SDF有意义）
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        if value_type == "SDF":
            # 提取接近表面的点（|SDF| < 阈值）
            threshold = np.percentile(np.abs(values), 10)  # 前10%最接近表面
            surface_mask = np.abs(values) < threshold
            surface_points = points[surface_mask]

            if len(surface_points) > 0:
                ax6.scatter(surface_points[:, 0], surface_points[:, 1],
                           surface_points[:, 2], c='red', s=2, alpha=0.6)
                ax6.set_title(f'Near-Surface Points (|SDF| < {threshold:.3f})')
            else:
                ax6.set_title('No Near-Surface Points Found')
        else:
            # 对于UDF，显示最小值点（最接近表面）
            threshold = np.percentile(values, 10)
            near_surface = points[values < threshold]
            if len(near_surface) > 0:
                ax6.scatter(near_surface[:, 0], near_surface[:, 1],
                           near_surface[:, 2], c='red', s=2, alpha=0.6)
                ax6.set_title(f'Closest Points (UDF < {threshold:.3f})')
            else:
                ax6.set_title('No Close Points Found')

        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')

        plt.suptitle(f'Preprocessed Data Visualization: {filename}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 保存图像
        output_path = output_dir / f"{filename.replace('.npz', '')}_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"保存可视化: {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='可视化预处理的UDF/SDF数据')
    parser.add_argument('--sdf_dir', type=str, default='data/npz-resample',
                       help='SDF数据目录')
    parser.add_argument('--udf_dir', type=str, default='data/npz-udf',
                       help='UDF数据目录')
    parser.add_argument('--output_dir', type=str, default='data/results/preprocessed_vis',
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='可视化的样本数量')

    args = parser.parse_args()

    print("=" * 60)
    print("可视化预处理数据")
    print("=" * 60)

    # 可视化SDF数据
    if Path(args.sdf_dir).exists():
        print(f"\n处理SDF数据从: {args.sdf_dir}")
        sdf_output = Path(args.output_dir) / "sdf"
        visualize_multiple_samples(args.sdf_dir, sdf_output, args.num_samples)

    # 可视化UDF数据
    if Path(args.udf_dir).exists():
        print(f"\n处理UDF数据从: {args.udf_dir}")
        udf_output = Path(args.output_dir) / "udf"
        visualize_multiple_samples(args.udf_dir, udf_output, args.num_samples)

    print("\n" + "=" * 60)
    print("完成！可视化结果保存在:", args.output_dir)
    print("=" * 60)

if __name__ == '__main__':
    main()
