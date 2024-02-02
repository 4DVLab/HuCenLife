from tqdm import tqdm
import json
import numpy as np
import os
from utils import *
from tqdm import tqdm
# import action_merge as action
from config import *
import os
import json
import argparse



def options():
    parser = argparse.ArgumentParser(description='HCL segmentation converting ...')
    parser.add_argument('--path_root',type=str,default='/remote-home/share/MotionDetectionSegemant')
    parser.add_argument('--split_file',type=str,default='./HCL_split.json')   # the split file 
    parser.add_argument('--split',type=str,default='train')                   # train / test 
    args = parser.parse_args()
    return args

args = options()
path_root = args.path_root
split = args.split
N = 262144
pkl_info = []
anno_file_dir = path_root + '/hcl_anno/'
anno_files = os.listdir(anno_file_dir)
with open('./HCL_split.json') as f:
    json_anno = json.load(f)
train_json = json_anno['train']
test_json = json_anno['test']
anno_files_train = [i  for i in anno_files if i in train_json]
anno_files_test = [i  for i in anno_files if i in test_json]

with open('./label_remap2.json') as f:
    label = json.load(f)
ground_info = np.load('/'.join([path_root,'ground_process.pkl']),allow_pickle=True)

count = 0
if split == 'test':
    count = 100000

if not os.path.exists('/'.join([path_root,'hcl_seg'])):
    os.mkdir('/'.join([path_root,'hcl_seg']))

if args.split == 'train':
    anno_files_traintest = anno_files_train
else:
    anno_files_traintest = anno_files_test


for anno_file in tqdm(anno_files_traintest):
    anno_file = anno_file_dir + anno_file
    with open(anno_file) as f:
        data = json.load(f)
    data_file = anno_file.split('.')[-2].split('/')[-1]
    for i in range(data['frames_number']):
        label_instance = np.zeros((N,1),dtype = np.int32)
        action_list = []
        for j in range(len(data['frames'][i]['instance'])): # for instance loop
            instance = data['frames'][i]['instance'][j]
            # try:
            category = instance['category']
            if category in list(class_map.keys()):
                category = class_map[category]
            label_instance[instance['seg_points']] = np.asarray(label[category] + ((j+1 << 16) & 0xFFFF0000),dtype="int32").reshape(-1,1)


            action = instance['action']

        if not os.path.exists('/'.join([path_root,'hcl_seg',data['data']])):
            os.mkdir('/'.join([path_root,'hcl_seg',data['data']]))
        label_dir = '/'.join([path_root,'hcl_seg',data['data'],'label'])
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
        dir_ = '/'.join([path_root,'hcl_seg',data['data'],'bin'])
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        
        fname = label_dir + "/"+data['frames'][i]['pc_name'].split('/')[-1].replace('.bin','.label')
        pcd_path = '/'.join([path_root,'HCL_Full',data_file,'bin',data['frames'][i]['pc_name'].split('/')[-1]])
        # label_instance.tofile(fname)
        info = {
            'scene': data_file,
            'pcd': dir_ + "/"+data['frames'][i]['pc_name'].split('/')[-1],
            'label': fname,
            'image_name':'/'.join([path_root,'HCL_Full',data_file,'img_blur/cam1',data['frames'][i]['pc_name'].split('/')[-1]]),
            'action':action_list
        }
        # print(info)
        count += 1
        
        pc = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)
        crop_range = [[12.5,0,0,25,50,6]]
        index = crop_pc_utils(pc,crop_range,0)
        pc_new = pc[index]
        label_new = label_instance[index]
        assert pc_new.shape[0] == label_new.shape[0]
        pc_new.tofile(info['pcd'])
        
        label_key = '/'.join(fname.split('/')[-3:])
        ground_label = ground_info[label_key]
        label_new[ground_label] = 20
        label_new.tofile(fname)
        
        pkl_info.append(info)

import pickle
with open(f"/remote-home/share/MotionDetectionSegemant/{split}.pkl",'wb') as f:
    result = pickle.dump(pkl_info,f)
