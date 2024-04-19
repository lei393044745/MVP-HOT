import cv2 as cv
import os
import sys
sys.path.append('/media/fox/6E3464883464555D2/xsx/VideoX-master/SeqTrack')
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
import glob
import numpy as np 
from torchvision.ops.boxes import box_area
import torch
def bbox_style():
    return [
        {'color': (0,255,0)}, #绿色
        {'color': (0,0,255)}, #红色
        {'color': (255,0,0)}, #蓝色
        {'color': (255,255,0)},#青色
        {'color': (255,0,255)},#紫色
        {'color': (0,255,255)},#黄色
        {'color': (128,128,128)},#灰色
        {'color': (128, 128, 0)},
        {'color': (128, 0, 128)}
    ]

import numpy as np

def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1) # (N,)
    area2 = box_area(boxes2) # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def main(result_dir, dataset_dir, tracker_names):
    all_bbox = {}
    for name in tracker_names:
        all_bbox[name] = sorted(glob.glob(os.path.join(result_dir,name) + '/*.txt'))
    viedo_length = len(all_bbox[name])
    # print(11)
    style = bbox_style()
    best_path = '/media/fox/6E3464883464555D2/xsx/VideoX-master/SeqTrack/lib/test/tracker/best_view/'
    for i in range(viedo_length):
        bbox = {}
        for name in tracker_names:
            try:
                bbox[name] = np.loadtxt(all_bbox[name][i], delimiter=',')
            except:
                bbox[name] = np.loadtxt(all_bbox[name][i])
        type = all_bbox[name][i].split('/')[-1][:3]
        base = dataset_dir
        viedo_name = all_bbox[name][i].split('-')[-1][:-4]
        if type == 'nir':
            base += '/HSI-NIR-FalseColor'
        elif type == 'red':
            base += '/HSI-RedNIR-FalseColor'
        else:
            base += '/HSI-VIS-FalseColor'
        anno = base + '/' + viedo_name + '/groundtruth_rect.txt'
        bbox['gt'] = np.loadtxt(anno)
        images = sorted(glob.glob(f'{base}/{viedo_name}/*.jpg'))
        length = len(bbox['gt'])
        for i in range(length):
            false_image = cv.imread(images[i])
            # gtbox = bbox
            h, w, _ = false_image.shape
            # lable_height = 50
            # lable_img = np.ones((lable_height, w, 3), dtype=np.uint8) * 255
            # gtbox = list(map(int, gtbox))
            gt = None
            p_seq_iou = 0

            arr = []
            for index,name in enumerate(['gt'] + tracker_names):
                pred = bbox[name][i]
                pred_bbox = list(map(int, pred))
                if name == 'gt' or index == 0:
                    gt = pred_bbox
                else:
                    iou = box_iou(box_xywh_to_xyxy(torch.tensor([pred_bbox])), box_xywh_to_xyxy(torch.tensor([gt])))[0].item()
                    if name == 'VP-HOT':
                        p_seq_iou = iou
                    else:
                        arr.append(iou)
                    # print(iou.item())
                cv.rectangle(
                    false_image, (pred_bbox[0], pred_bbox[1]), (pred_bbox[0] + pred_bbox[2], pred_bbox[1]+pred_bbox[3]),
                    style[index]['color'], 2
                )
                # cv.putText(lable_img,f'{name}', (index * 60 + 10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, style[index]['color'], 1)
            # result = np.vstack((false_image, lable_img))
            cv.putText(false_image, f'{i}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,255), 2,cv.LINE_AA)
            if p_seq_iou - max(arr) >= 0.15:
                cv.imwrite(best_path + f'{type}-{viedo_name}-{i}.jpg', false_image)
            cv.imshow('', false_image)
            cv.waitKey(1)

if __name__ == '__main__':
    result_dir = '/media/fox/6E3464883464555D2/xsx/VideoX-master/SeqTrack/lib/test/tracker/result'
    dataset_dir = '/media/fox/15080939085/whisper2023/validation'
    tracker_names = ['SiamGAT','TransT','SiamCAR','SeqTrack','VP-HOT','SiamBAN','MHT','STARK']
    
    main(result_dir, dataset_dir, tracker_names)