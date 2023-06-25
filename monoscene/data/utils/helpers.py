import numpy as np
import monoscene.data.utils.fusion as fusion
import torch


def compute_CP_mega_matrix(target, is_binary=False):
    """
    Parameters
    ---------
    target: (H, W, D)
        contains voxels semantic labels

    is_binary: bool
        if True, return binary voxels relations else return 4-way relations
    """
    label = target.reshape(-1)
    label_row = label
    N = label.shape[0]
    super_voxel_size = [i//2 for i in target.shape]
    if is_binary:
        matrix = np.zeros((2, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)
    else:
        matrix = np.zeros((4, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)

    for xx in range(super_voxel_size[0]):
        for yy in range(super_voxel_size[1]):
            for zz in range(super_voxel_size[2]):
                col_idx = xx * (super_voxel_size[1] * super_voxel_size[2]) + yy * super_voxel_size[2] + zz
                label_col_megas = np.array([
                    target[xx * 2,     yy * 2,     zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2],
                    target[xx * 2,     yy * 2 + 1, zz * 2],
                    target[xx * 2,     yy * 2,     zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2 + 1],
                    target[xx * 2,     yy * 2 + 1, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2 + 1],
                ])
                label_col_megas = label_col_megas[label_col_megas != 255]
                for label_col_mega in label_col_megas:
                    label_col = np.ones(N)  * label_col_mega
                    if not is_binary:
                        matrix[0, (label_row != 255) & (label_col == label_row) & (label_col != 0), col_idx] = 1.0 # non non same
                        matrix[1, (label_row != 255) & (label_col != label_row) & (label_col != 0) & (label_row != 0), col_idx] = 1.0 # non non diff
                        matrix[2, (label_row != 255) & (label_row == label_col) & (label_col == 0), col_idx] = 1.0 # empty empty
                        matrix[3, (label_row != 255) & (label_row != label_col) & ((label_row == 0) | (label_col == 0)), col_idx] = 1.0 # nonempty empty
                    else:
                        matrix[0, (label_row != 255) & (label_col != label_row), col_idx] = 1.0 # diff
                        matrix[1, (label_row != 255) & (label_col == label_row), col_idx] = 1.0 # same
    return matrix


def vox2pix(cam_E, cam_k, 
            vox_origin, voxel_size, 
            img_W, img_H, 
            scene_size):
    """计算体素中心到 2D 的投影
    
    Args:
        cam_E: 4x4
           = NYUv2 数据集中的相机位姿
           = SemKITTI 数据集中从相机坐标系到激光雷达坐标系的变换 (相机相对于激光雷达的位姿) 
        cam_k: 3x3
            相机内参
        vox_origin: (3,)
            索引为 (0, 0, 0) 的体素的世界(NYU) / 激光雷达(SemKITTI) 坐标
        img_W: int
            图像宽度
        img_H: int
            图像高度
        scene_size: (3,)
            场景尺寸 (米) 
            = SemKITTI 为 (51.2, 51.2, 6.4)
            = NYUv2 为 (4.8, 4.8, 2.88)
    
    Returns:
        projected_pix: (N, 2)
            体素投影到 2D 的位置
        fov_mask: (N,)
            体素的模板, 指引了在图像的 FOV 之内的体素 
        pix_z: (N,)
            体素到传感器的距离 (米)
    """
    # 计算场景的 x, y, z 边界 (米)
    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin  # 场景边界 min 为体素原点所在的世界坐标
    vol_bnds[:,1] = vox_origin + np.array(scene_size)  # 场景边界 max 为原点世界坐标 + 场景尺寸

    # 计算激光雷达坐标系下的体素中心
    vol_dim = np.ceil((vol_bnds[:,1] - vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)  # 各维度格子数量
    xv, yv, zv = np.meshgrid(
            range(vol_dim[0]),
            range(vol_dim[1]),
            range(vol_dim[2]),
            indexing='ij'
          )
    # 得到每一个体素的索引
    vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
          ], axis=0).astype(int).T

    # 将体素中心从激光雷达坐标系投影到相机坐标系
    cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = fusion.rigid_transform(cam_pts, cam_E)

    # 从相机坐标系投影到像素位置
    projected_pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # 消除在视锥之外的像素
    pix_z = cam_pts[:, 2]  # 保留体素到相机的深度信息
    fov_mask = np.logical_and(pix_x >= 0,  # 得到视锥之内的体素模板
                np.logical_and(pix_x < img_W,
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y < img_H,
                pix_z > 0))))


    return projected_pix, fov_mask, pix_z


def compute_local_frustum(pix_x, pix_y, min_x, max_x, min_y, max_y, pix_z):
    valid_pix = np.logical_and(pix_x >= min_x,
                np.logical_and(pix_x < max_x,
                np.logical_and(pix_y >= min_y,
                np.logical_and(pix_y < max_y,
                pix_z > 0))))
    return valid_pix

def compute_local_frustums(projected_pix, pix_z, target, img_W, img_H, dataset, n_classes, size=4):
    """
    Compute the local frustums mask and their class frequencies
    
    Parameters:
    ----------
    projected_pix: (N, 2)
        2D projected pix of all voxels
    pix_z: (N,)
        Distance of the camera sensor to voxels
    target: (H, W, D)
        Voxelized sematic labels
    img_W: int
        Image width
    img_H: int
        Image height
    dataset: str
        ="NYU" or "kitti" (for both SemKITTI and KITTI-360)
    n_classes: int
        Number of classes (12 for NYU and 20 for SemKITTI)
    size: int
        determine the number of local frustums i.e. size * size
    
    Returns
    -------
    frustums_masks: (n_frustums, N)
        List of frustums_masks, each indicates the belonging voxels  
    frustums_class_dists: (n_frustums, n_classes)
        Contains the class frequencies in each frustum
    """
    H, W, D = target.shape
    ranges = [(i * 1.0/size, (i * 1.0 + 1)/size) for i in range(size)]
    local_frustum_masks = []
    local_frustum_class_dists = []
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
    for y in ranges:
        for x in ranges:
            start_x = x[0] * img_W
            end_x = x[1] * img_W
            start_y = y[0] * img_H
            end_y = y[1] * img_H
            local_frustum = compute_local_frustum(pix_x, pix_y, start_x, end_x, start_y, end_y, pix_z)
            if dataset == "NYU":
                mask = (target != 255) & np.moveaxis(local_frustum.reshape(60, 60, 36), [0, 1, 2], [0, 2, 1])
            elif dataset == "kitti":
                mask = (target != 255) & local_frustum.reshape(H, W, D)

            local_frustum_masks.append(mask)
            classes, cnts = np.unique(target[mask], return_counts=True)
            class_counts = np.zeros(n_classes)
            class_counts[classes.astype(int)] = cnts
            local_frustum_class_dists.append(class_counts)
    frustums_masks, frustums_class_dists = np.array(local_frustum_masks), np.array(local_frustum_class_dists)
    return frustums_masks, frustums_class_dists
