import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
import glob
from .base_video_dataset import BaseVideoDataset
import cv2 as cv

class Whispers(BaseVideoDataset):
    def __init__(self, path, type) -> None:
        self.base_path = path
        self.type = type
        self.nir_his_list = None
        self.nir_false_list = None
        self.vis_his_list = None
        self.vis_false_list = None
        self.red_nir_his_list = None
        self.red_nir_false_list = None
        self.get_list()
        
        # ['car39', 'car40']
        
        
        if self.type == 'HSI-NIR':
            self.no_train_list = ['car39', 'car40']
            self.nir_his_list, self.nir_false_list = self.del_list(self.nir_his_list, self.nir_false_list)
            self.sequence_list = self.nir_false_list
            
        elif self.type == 'HSI-VIS':
            self.no_train_list = ['car4', 'car6', 'car7', 'car8', 'car10', 
                               'automobile2', 'automobile5', 'automobile13', 'automobile14', 'automobile6', 'automobile7', 'automobile8',
                               ]
            self.vis_his_list, self.vis_false_list = self.del_list(self.vis_his_list, self.vis_false_list)
            self.sequence_list = self.vis_false_list
        else:
            self.no_train_list = []
            self.red_nir_his_list, self.red_nir_false_list = self.del_list(self.red_nir_his_list, self.red_nir_false_list)
            self.sequence_list = self.red_nir_false_list
        print(len(self.sequence_list))
        
    def del_list(self, list1, list2):
        arr1 = []
        arr2 = []
        for index , item in enumerate(list1):
            name = item.split('/')[-1]
            if name not in self.no_train_list:
                arr1.append(item)
        for index , item in enumerate(list2):
            name = item.split('/')[-1]
            if name not in self.no_train_list:
                arr2.append(item)
        return arr1, arr2
        
    def X2Cube(self, img,B=[4, 4],skip = [4, 4],bandNumber=16):
        t = img
        img = cv.imread(img, -1)
        # Parameters
        try:
            M, N = img.shape
        except:
            print(t)
        col_extent = N - B[1] + 1
        row_extent = M - B[0] + 1
        # Get Starting block indices
        start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
        # Generate Depth indeces
        didx = M * N * np.arange(1)
        start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
        # Get offsetted indices across the height and width of input array
        offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
        # Get all actual indices & index into input array for final output
        out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
        if bandNumber == 15:
            out = out[:-1]
        out = np.transpose(out)
        DataCube = out.reshape(M//skip[0], N//skip[1],bandNumber )
        return DataCube
        
    def _get_sequence_path(self, seq_id):
        if self.type == 'HSI-NIR':
            return self.nir_false_list[seq_id]
        elif self.type == 'HSI-VIS':
            return self.vis_false_list[seq_id]
        else:
            return self.red_nir_false_list[seq_id]
        
    def _read_bb_anno(self, seq_id):
        if self.type == 'HSI-VIS':
            bb_anno_file = self.vis_false_list[seq_id] + '/groundtruth_rect.txt'
        elif self.type == 'HSI-NIR':
            bb_anno_file = self.nir_false_list[seq_id] + '/groundtruth_rect.txt'
        else:
            bb_anno_file = self.red_nir_false_list[seq_id] + '/groundtruth_rect.txt'
        gt = np.loadtxt(bb_anno_file, dtype=np.float32)
        return torch.tensor(gt)
    
    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0).byte()

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio
        
    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_id)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = torch.ones(len(bbox), dtype=torch.uint8)
        visible_ratio = torch.ones(len(bbox), dtype=torch.uint8)
        # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        # visible, visible_ratio = [True] * len(valid), 1.0
        # visible = visible & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}
    
    def __len__(self):
        if self.type == 'HSI-NIR':
            return len(self.nir_false_list)
        elif self.type == 'HSI-VIS':
            return len(self.vis_false_list)
        else:
            return len(self.rednir_false_list)
        
    def _get_frame_path(self, seq_path, frame_id, hsi=False):
        if not hsi:
            return os.path.join(seq_path, '{:04}.jpg'.format(frame_id+1))
        else:
            return os.path.join(seq_path, '{:04}.png'.format(frame_id+1))
        
    def _get_frame(self, seq_path, frame_id):
        return cv.imread(self._get_frame_path(seq_path, frame_id))
    
    def hsi_get_frame(self, seq_path, frame_id):
        if self.type == 'HSI-NIR':
            return self.X2Cube(img = self._get_frame_path(seq_path, frame_id, True), B=[5, 5], skip=[5, 5], bandNumber=25).astype(np.float32)
        elif self.type == 'HSI-VIS':
            return self.X2Cube(img = self._get_frame_path(seq_path, frame_id, True), B=[4, 4], skip=[4, 4], bandNumber=16).astype(np.float32)
        else:
            return self.X2Cube(img = self._get_frame_path(seq_path, frame_id, True), B=[4, 4], skip=[4, 4], bandNumber=15).astype(np.float32)
    
    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        t = seq_path.split('/')
        self.name = t[-1]
        t[-2] = self.type
        hsi_path = ('/').join(t)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        hsi_fram_list = [self.hsi_get_frame(hsi_path, f_id) for f_id in frame_ids]
        obj_meta = None
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        return frame_list, anno_frames, obj_meta, hsi_fram_list
    
    
    def get_name(self):
        return self.name
        
    def get_list(self):
        nir_base_path = os.path.join(self.base_path, 'training')
        self.nir_his_list = glob.glob(nir_base_path + '/HSI-NIR' + '/*')
        self.nir_false_list = glob.glob(nir_base_path + '/HSI-NIR-FalseColor' + '/*')
        self.vis_his_list = glob.glob(nir_base_path + '/HSI-VIS' + '/*')
        self.vis_false_list = glob.glob(nir_base_path + '/HSI-VIS-FalseColor' + '/*')
        self.red_nir_his_list = glob.glob(nir_base_path + '/HSI-RedNIR' + '/*')
        self.red_nir_false_list = glob.glob(nir_base_path + '/HSI-RedNIR-FalseColor' + '/*')
        # print(111)