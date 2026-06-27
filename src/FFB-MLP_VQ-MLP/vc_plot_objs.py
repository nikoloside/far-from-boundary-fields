import imagej
import os
import numpy as np
import nibabel as nib
import glob
import vedo as vd
import yaml
import shutil

# 全局 ImageJ 实例
_ij = None

def get_imagej():
    global _ij
    if _ij is None:
        # 读取配置文件
        from config_utils import get_config
        
        config = get_config()
        foundation_path = config['foundation_path']
        sourcePath = config['fiji_path']
        print(f"Fiji path: {sourcePath}")
        
        try:
            _ij = imagej.init(sourcePath, mode="headless", add_legacy=False)
            print(f"ImageJ version: {_ij.getVersion()}")
        except Exception as e:
            print(f"Error initializing ImageJ: {e}")
            print("Trying alternative initialization...")
            _ij = imagej.init('sc.fiji:fiji', mode="headless", add_legacy=True)
            print(f"ImageJ version: {_ij.getVersion()}")
    return _ij

def get_from_nib(file_name):
    img = nib.load(file_name)
    data = img.get_fdata()
    return data

def save_as_nib(file_name, variable):
    ni_img = nib.Nifti1Image(variable, affine=np.eye(4))
    nib.save(ni_img, file_name)

def getMaskForSdf(data):
    mask = data.copy()
    mask[data >= 0] = 1  # 内部領域
    mask[data < 0] = 0   # 外部
    return mask

def getNormForSdf(data):
    sdf = data.copy()
    print(sdf.min(), sdf.max())
    sdf[sdf < 0] *= -1  # 内部領域
    sdf = 255 - (sdf + 1) / 2 * 255
    print(sdf.min(), sdf.max())
    return sdf

def processSDFWithImageJ(data_ori, work_path, isolevel=0.03, resolution=128):
    # 获取 ImageJ 实例
    ij = get_imagej()

    # 基本参数设置
    shift = resolution / 2
    data = data_ori.copy()
    data = data + isolevel

    # 获取mask和归一化数据
    mask = getMaskForSdf(data)
    discrete = getNormForSdf(data)

    save_as_nib(os.path.join(work_path, "discrete.nii"), discrete)

    # 首先进行直接重建
    print("Performing direct reconstruction...")
    vol_direct = vd.Volume(data_ori).isosurface(isolevel).smooth()
    scale = 1/resolution*2
    vol_direct = vol_direct.shift(-shift, -shift, -shift).scale(scale, scale, scale).rotate_x(180).rotate_y(-90).rotate_z(90)
    vol_direct.write(os.path.join(work_path, "reconstructed_direct.obj"))
    print("Direct reconstruction completed.")

    # 转换为 ImageJ 格式
    try:
        dataset = ij.py.to_java(discrete)
    except Exception as e:
        print(f"Error converting to Java: {e}")
        print("Trying alternative conversion...")
        dataset = ij.py.to_java(discrete.astype(np.float32))

    # ImageJ 分割处理
    print("Start SegmentationProcess")
    script = """
    // @ImagePlus(label="Input image", description="Image to segment") imp
    // @OUTPUT ImagePlus resultImage

    import ij.IJ;
    import ij.ImagePlus;
    import ij.ImageStack;
    import inra.ijpb.binary.BinaryImages;
    import inra.ijpb.morphology.MinimaAndMaxima3D;
    import inra.ijpb.morphology.Morphology;
    import inra.ijpb.morphology.Strel3D;
    import inra.ijpb.watershed.Watershed;

    radius = 2;
    tolerance = 10;
    strConn = "6";
    dams = true;

    conn = Integer.parseInt(strConn);

    image = imp.getImageStack().duplicate();
    regionalMinima = MinimaAndMaxima3D.extendedMinima(image, tolerance, conn);
    imposedMinima = MinimaAndMaxima3D.imposeMinima(image, regionalMinima, conn);
    labeledMinima = BinaryImages.componentsLabeling(regionalMinima, conn, 32);
    resultStack = Watershed.computeWatershed(imposedMinima, labeledMinima, conn, dams);
    
    resultImage = new ImagePlus("watershed", resultStack);
    resultImage.setCalibration(imp.getCalibration());
    """
    try:
        imp = ij.py.to_imageplus(dataset)
        args = {"imp": imp}
        result = ij.py.run_script("BeanShell", script, args)
        resultImp = result.getOutput("resultImage")

        # 转换回 numpy 数组
        xr = ij.py.from_java(resultImp)
        xr = np.array(xr)
        xr[mask == 0] = -1
        # 设置边界为-1
        xr[0, :, :] = -1
        xr[-1, :, :] = -1
        xr[:, 0, :] = -1
        xr[:, -1, :] = -1
        xr[:, :, 0] = -1
        xr[:, :, -1] = -1

        # 过滤小区域
        filtNoisy = 50
        unique, counts = np.unique(xr, return_counts=True)
        for value in unique[counts < filtNoisy]:
            xr[xr == int(value)] = -1

        

        # 保存分割结果
        save_as_nib(os.path.join(work_path, "imj.nii"), xr)

        # 使用 vedo 进行重建
        print("Performing ImageJ-based reconstruction...")
        vol = vd.Volume(xr).isosurface(isolevel).smooth()
        scale = 1/resolution*2
        vol = vol.shift(-shift, -shift, -shift).scale(scale, scale, scale).rotate_x(180).rotate_y(-90).rotate_z(90)

        # 保存结果
        os.makedirs(work_path, exist_ok=True)
        vol.write(os.path.join(work_path, "reconstructed.obj"))
        print("ImageJ-based reconstruction completed.")
    except Exception as e:
        print(f"Error during ImageJ processing: {e}")
        print("ImageJ-based reconstruction failed, but direct reconstruction is still available.")

def main():
    # 设置工作目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nii_dir = os.path.join(base_dir, "ffb-df")
    output_dir = os.path.join(base_dir, "recon_output")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有nii文件
    nii_files = glob.glob(os.path.join(nii_dir, "*-ffb-df.nii"))
    
    for nii_file in nii_files:
        print(f"Processing {nii_file}")
        # 创建对应的输出目录
        file_name = os.path.basename(nii_file).replace(".nii", "")
        work_path = os.path.join(output_dir, file_name)
        
        # 如果目录已存在，先清理
        if os.path.exists(work_path):
            shutil.rmtree(work_path)
        os.makedirs(work_path)
        
        # 读取数据并处理
        data = get_from_nib(nii_file)[...,0]
        print(data.shape)
        processSDFWithImageJ(data, work_path)

import os
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# Define input and output paths
# input_pattern = "../result-exp/final-result/*/*.nii"
input_pattern = "../result-exp/final-result/squirrel-feature-tsdf-50/*.nii"
nii_files = glob.glob(input_pattern)

output_base = "./objs"

# nii_files = [
#     "../result-exp/final-result/ffbdfwm/test_epoch_4800_ind_1-vq-poc-1-md.nii",
#     "../result-exp/final-result/ffbdfwm/test_epoch_4800_ind_4-vq-poc-4-md.nii",
#     # "../result-exp/final-result/ffbdf-1/test_epoch_4800_ind_1-vq-poc-1-md.nii",
#     # "../result-exp/final-result/ffbdf-1/test_epoch_4800_ind_4-vq-poc-4-md.nii",
#     "../result-exp/final-result/ffbsigmoid-wm/test_epoch_4800_ind_1-vq-poc-1-md.nii",
#     "../result-exp/final-result/ffbsigmoid-wm/test_epoch_4800_ind_4-vq-poc-4-md.nii",
#     # "../result-exp/final-result/ffbsigmoid-1/test_epoch_4800_ind_1-vq-poc-1-md.nii",
#     # "../result-exp/final-result/ffbsigmoid-1/test_epoch_4800_ind_4-vq-poc-4-md.nii",
#     "../result-exp/final-result/gssdf.020/test_epoch_4800_ind_1-vq-poc-1-md.nii",
#     "../result-exp/final-result/gssdf.020/test_epoch_4800_ind_4-vq-poc-4-md.nii",
#     "../result-exp/final-result/tsdf-1-1/test_epoch_4800_ind_1-vq-poc-1-md.nii",
#     "../result-exp/final-result/tsdf-1-1/test_epoch_4800_ind_4-vq-poc-4-md.nii",
#              ]

# fit_nums = [
#     0.03,
#     0.03,
#     0.03, 
#     0.03,
#     0.035,
#     0.035,
#     0.05,
#     0.05,
# ]
# for nii_file, fit_num in zip(nii_files, fit_nums):
fit_num = 0.03
for nii_file in nii_files:
    # Load NIfTI file
    img = nib.load(nii_file)
    data = img.get_fdata()
    
    # Rotate the slice [90, 0, 90] degrees
    if 'gssdf' in nii_file:
        data = np.transpose(data, (1, 0, 2))  # 交换 x 和 y 轴

    data = np.transpose(data, (1, 0, 2))

    if 'gssdf' in  nii_file:
        data = np.rot90(data, 1, axes=(0,1))
        data = np.rot90(data, 1, axes=(0,2))
        data = np.rot90(data, 1, axes=(1,2))
        fit_num = 0.035
    elif '2025-06' in nii_file:
        data = np.rot90(data, 1, axes=(0,1))
        data = np.rot90(data, 1, axes=(0,2))
        data = np.rot90(data, 2, axes=(1,2))
        # continue
    else:
        data = np.rot90(data, 1, axes=(0,2))
        # continue

    if 'tsdf' in nii_file:
        fit_num = 0.5
    #     data[data < 0] *= 5

    # # Convert tsdf-0-1 data to [-1,1] range
    # if 'tsdf-0-1' in nii_file:
    #     data = data - 1

    # if 'tsdf-1-1' in nii_file:
    #     data = data / 2

    if '2025-06' in nii_file:
        data = np.transpose(data, (1, 0, 2))  # 交换 x 和 y 轴

    # Create output directory
    rel_path = os.path.relpath(nii_file, "../../ffb-df/result-exp/final-result")
    output_dir = os.path.join(output_base, os.path.dirname(rel_path))
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # break
    # Create output filename
    output_file = os.path.join(output_dir, 
                              os.path.splitext(os.path.basename(nii_file))[0])

    
    if not os.path.exists(output_file):
        os.makedirs(output_file) 
    # else:
    #     continue

    # 读取数据并处理
    processSDFWithImageJ(data, output_file, isolevel=fit_num)
    # processSDFWithImageJ(data, output_file)

print("Processing completed!")
