import open3d as o3d
import numpy as np
import re
pcd = o3d.geometry.PointCloud()
import re
from concurrent import futures as futures
import json
import numpy as np
import os
from utils import *
from tqdm import tqdm

path_root = '/remote-home/share/MotionDetectionSegemant'
anno_file_dir = path_root + '/hcl_anno/'
anno_files = os.listdir(path_root + '/hcl_anno/')

min_k = 3
numberofworkers = 4
voxel_size = 0.2   #downsample the gt ground and others to speed up




def topk_partition(matrix, k, top_max=True):
    '''
    输出matrix矩阵的最大/小前k个值的index,无序
    所对应的top k的值:matrix[index[i][0], index[i][1]]
                        matrix[index[i]]
    :param matrix: 二维或一维
    :param k:
    :param top_max: True:最大值的前k个值,无序,False:最小值的前k个值,无序
    :return:
    '''
    flag_2d = False
    if len(matrix.shape) == 2:
        flag_2d = True
        matrix1 = matrix.reshape((matrix.shape[0] * matrix.shape[1]))
    else:
        matrix1 = matrix.copy()
    if top_max: # matrix前k个最大值 所对应的index
        index = np.argpartition(matrix1, -k)[-k:len(matrix1)]
    else:   # matrix前k个最小值 所对应的index
        index = np.argpartition(matrix1, k-1)[0:k]

    # 解析index
    if flag_2d:
        index_2d = []
        for i in index:
            row = i // matrix.shape[1]
            col = i % matrix.shape[1]
            index_2d.append([row,col])
        index = index_2d

    return index


info = []
for anno_file in tqdm(anno_files):
    anno_file = anno_file_dir + anno_file
    with open(anno_file) as f:
        gt_file = json.load(f)
    frame_data = gt_file["frames"][0]
    gt_file["frames"].pop(0)

    gt_file['data'] = frame_data['pc_name'].split('/')[-3]
    # if gt_file['data'] in dir_replace.keys():
    #     gt_file['data'] = dir_replace[gt_file['data']]

    bin_path = '/'.join([path_root,'hcl_seg',gt_file['data'],'bin',frame_data['pc_name'].split('/')[-1]])

    l = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)[:, :3]
    pc_label = np.fromfile(bin_path.replace('bin','label'), dtype=np.int32).reshape(-1, 1) & 0xFFFF

    gt_other = None
    gt_ground = None
    gt_data = l
    gt_ground = gt_data[np.where([pc_label==20])[1]]
    gt_other = gt_data[np.where([pc_label==0])[1]]
    min_g,max_g = np.min(gt_ground,0)[-1],np.max(gt_ground,0)[-1]

    pcd.points = o3d.utility.Vector3dVector(gt_ground)
    downpcd = pcd
    downpcd=pcd.voxel_down_sample(voxel_size=0.2)
    gt_ground = np.asarray(downpcd.points)

    pcd.points = o3d.utility.Vector3dVector(gt_other)
    downpcd = pcd
    downpcd=pcd.voxel_down_sample(voxel_size=0.2)
    gt_other = np.asarray(downpcd.points)

    gt_ground = np.c_[gt_ground, np.ones((gt_ground.shape[0], 1))]
    gt_other = np.c_[gt_other, np.zeros((gt_other.shape[0], 1))]
    gt_data = np.concatenate([gt_other, gt_ground], axis=0)




    for frame_data in tqdm(gt_file["frames"]): 
        
        ground = []
        other = []
        gt_point_list = []
        
        for instant in frame_data['instance']:
            gt_point_list+=instant['seg_points']

        next_bin_path = '/'.join([path_root,'hcl_seg',gt_file['data'],'bin',frame_data['pc_name'].split('/')[-1]])
        pc_label = np.fromfile(next_bin_path.replace('bin','label'), dtype=np.int32).reshape(-1, 1) 
        semantic_label = pc_label & 0xFFFF
        
        
        new_pc = np.fromfile(next_bin_path, dtype=np.float32).reshape(-1, 5)[:, :3]
        new_pc = np.c_[new_pc, np.arange(new_pc.shape[0])]
        mask_pz = ((new_pc[:,2])>=min_g-0.2) & (new_pc[:,2]<=max_g+0.2)
        mask = np.array([i not in gt_point_list for i in range(new_pc.shape[0])])
        crop_pcd = new_pc[mask&mask_pz]
        # print(crop_pcd.shape)
        crop_pcd = np.c_[crop_pcd, np.arange(crop_pcd.shape[0])]
        
        def working_k(i):
            point  = crop_pcd[i,:3]
            distance = np.sqrt(np.sum((point-gt_data[:,:3])**2,axis=1))
            index = topk_partition(distance, min_k, top_max=False)

            # score = np.average(gt_data[index,3])
            score = np.sum(gt_data[index,3]/(distance[index]+1e-6))/np.sum(1/(distance[index]+1e-6))
            if  score>= 0.5:
                ground.append(int(crop_pcd[i,3]))
            else:
                other.append(int(crop_pcd[i,3]))
           
        with futures.ThreadPoolExecutor(max_workers=8) as executor:
            infos = executor.map(working_k, [i for i in range(crop_pcd.shape[0])])
            
        label_index = np.where([semantic_label != 0])[1]
        ground_index = [i for i in list(set(ground)) if i not in label_index]
        
        pc_label[ground_index] = 20
        pc_label.tofile(next_bin_path.replace('bin','label'))

        info.append({
            'ground_index': ground_index,
            'label_path':next_bin_path.replace('bin','label')
        })

import pickle
with open(f"{path_root}/ground_process.pkl",'wb') as f:
    result = pickle.dump(info,f)