import os
import sys
sys.path.append('/media/fox/6E3464883464555D2/xsx/VideoX-master/SeqTrack')
# from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.seqtrack_utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.models.seqtrack import build_seqtrack
from lib.test.tracker.seqtrack_utils import Preprocessor
from lib.utils.box_ops import clip_box
import numpy as np
import glob
import numpy as np  
import importlib
from lib.test.tracker.report import ExperimentOTB
import argparse
from eval import eval
parser = argparse.ArgumentParser(description='transt tracking')
parser.add_argument('--epoch', default='', type=str,
        help='name of results')
args = parser.parse_args()
size = 256
class SEQTRACK():
    def __init__(self, cfg, weight_path):
        super(SEQTRACK, self).__init__()
        network = build_seqtrack(cfg)
        network.load_state_dict(torch.load(weight_path, map_location='cpu')['net'], strict=True)
        network.encoder.body.lamdas = np.ones(network.encoder.body.prompt_depth)
        self.cfg = cfg
        self.seq_format = self.cfg.DATA.SEQ_FORMAT
        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.bins = self.cfg.MODEL.BINS
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.hanning = torch.tensor(np.hanning(self.bins)).unsqueeze(0).cuda()
            self.hanning = self.hanning
        else:
            self.hanning = None
        self.start = self.bins + 1 # start token
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.debug = False
        self.frame_id = 0

        # # online update settings
        # DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, 'VOT22'):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS['VOT22']
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)
        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, 'VOT22'):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD['VOT22']
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        print("Update threshold is: ", self.update_threshold)



    def initialize(self, image, info: dict, hsi_image):

        # get the initial templates
        z_patch_arr, _ = sample_target(image, info['init_bbox'], 4.0,
                                       output_sz=size)
        hsi_z_patch_arr, _ = sample_target(hsi_image, info['init_bbox'], 4.0,
                                       output_sz=size)
        hsi_z_patch_arr = hsi_z_patch_arr.astype(np.float32)
        template = self.preprocessor.process(z_patch_arr)
        hsi_template = self.preprocessor.process(hsi_z_patch_arr)
        self.template_list = [template] * self.num_template
        self.hsi_template_list = [hsi_template] * self.num_template
        # get the initial sequence i.e., [start]
        batch = template.shape[0]
        self.init_seq = (torch.ones([batch, 1]).to(template) * self.start).type(dtype=torch.int64)

        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, hsi_image):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, 4.0,
                                                   output_sz=size)  # (x1, y1, w, h)
        hsi_x_patch_arr, _ = sample_target(hsi_image, self.state, 4.0,
                                                   output_sz=size)
        hsi_x_patch_arr = hsi_x_patch_arr.astype(np.float32)
        search = self.preprocessor.process(x_patch_arr)
        hsi_search = self.preprocessor.process(hsi_x_patch_arr)
        images_list = self.template_list + [search]
        hsi_image_list = self.hsi_template_list + [hsi_search]
        # run the encoder
        with torch.no_grad():
            xz = self.network.forward_encoder(images_list, hsi_image_list)

        # run the decoder
        with torch.no_grad():
            out_dict = self.network.inference_decoder(xz=xz,
                                                      sequence=self.init_seq,
                                                      window=self.hanning,
                                                      seq_format=self.seq_format)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)

        # if use other formats of sequence
        if self.seq_format == 'corner':
            pred_boxes = box_xyxy_to_cxcywh(pred_boxes)
        if self.seq_format == 'whxy':
            pred_boxes = pred_boxes[:, [2, 3, 0, 1]]

        pred_boxes = pred_boxes / (self.bins-1)
        pred_box = (pred_boxes.mean(dim=0) * size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # update the template
        if self.num_template > 1:
            conf_score = out_dict['confidence'].sum().item() * 10 # the confidence score
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, _ = sample_target(image, self.state, 4.0,
                                               output_sz=size)
                hsi_z_patch_arr, _ = sample_target(hsi_image, self.state, 4.0,
                                                   output_sz=size)
                hsi_z_patch_arr = hsi_z_patch_arr.astype(np.float32)
                template = self.preprocessor.process(z_patch_arr)
                hsi_template = self.preprocessor.process(hsi_z_patch_arr)
                self.template_list.append(template)
                self.hsi_template_list.append(hsi_template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)
                    self.hsi_template_list.pop(1)

        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

def X2Cube(img,B=[4, 4],skip = [4, 4],bandNumber=16):
        t = img
        img = cv2.imread(img, -1)
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


dataset_path = '/media/fox/15080939085/whisper2023/validation'
data_type = 'vis-'
# weight_path = '/media/fox/6E3464883464555D/xsx/VideoX-master/SeqTrack/checkpoints/train/seqtrack/%s/SEQTRACK_ep%04d.pth.tar' % (data_type[:-1] , int(args.epoch))
weight_path = '/media/fox/6E3464883464555D2/xsx/VideoX-master/SeqTrack/checkpoints/train/seqtrack/seqtrack_b256/SEQTRACK_ep%04d.pth.tar' % int(args.epoch)
# weight_path = '/media/fox/6E3464883464555D2/xsx/VideoX-master/SeqTrack/checkpoints/train/seqtrack/fix_up/0.0005/red/SEQTRACK_ep0010.pth.tar'
# weight_path = '/media/fox/6E3464883464555D/xsx/VideoX-master/SeqTrack/checkpoints/train/seqtrack/fix_up/0.0001/rednir/SEQTRACK_ep0010.pth.tar'
# weight_path = '/media/fox/6E3464883464555D/xsx/VideoX-master/SeqTrack/checkpoints/train/seqtrack/fix_up/0.0001/nir/SEQTRACK_ep0010.pth.tar'
save_path = '/media/fox/6E3464883464555D2/xsx/VideoX-master/SeqTrack/lib/test/tracker/result/prompt_seqtrack'
# hard = ['cards16','cards19','rainystreet10','rainystreet16','toy1', 'fruit', 
#         'campus', 'pedestrian2','card','coin', 'pool10','rider2','playground','worker', 'paper', 'basketball']
# vis_hsi_list = []
# vis_false_list = []
# for item in hard:
#     vis_false_list.append(dataset_path + '/HSI-VIS-FalseColor/' + item)
#     vis_hsi_list.append(dataset_path + '/HSI-VIS/' + item)

if data_type == 'nir-':
    s1 = '/HSI-NIR'
    s2 = '/HSI-NIR-FalseColor'
elif data_type == 'rednir-':
    s1 = '/HSI-RedNIR'
    s2 = '/HSI-RedNIR-FalseColor'
else:
    s1 = '/HSI-VIS'
    s2 = '/HSI-VIS-FalseColor'

vis_hsi_list = sorted(glob.glob(dataset_path + s1 + '/*'))
vis_false_list = sorted(glob.glob(dataset_path + s2 + '/*'))
viedo_length = len(vis_hsi_list)
config_module = importlib.import_module("lib.config.seqtrack.config")
cfg = config_module.cfg # generate cfg from lib.config
config_module.update_config_from_file('/media/fox/6E3464883464555D2/xsx/VideoX-master/SeqTrack/experiments/seqtrack/seqtrack_b256.yaml') 
# e = ExperimentOTB(
#                     root_dir=None,
#                     result_dir = '/media/fox/6E3464883464555D/xsx/VideoX-master/SeqTrack/lib/test/tracker/result',
#                     report_dir= '/media/fox/6E3464883464555D/xsx/VideoX-master/SeqTrack/lib/test/tracker/report',
#                     OTB = vis_false_list
# )
tracker = SEQTRACK(cfg, weight_path)
vis = True
for idx, (hsi_path, false_path) in enumerate(zip(vis_hsi_list, vis_false_list)):
    pred_boxex = []
    name = hsi_path.split('/')[-1]
    toc = 0
    hsi_images_path = sorted(glob.glob(hsi_path + '/*.png'))
    false_images_path = sorted(glob.glob(false_path + '/*.jpg'))
    length = len(hsi_images_path)
    bb_anno_file = false_path + '/groundtruth_rect.txt'
    boxes = np.loadtxt(bb_anno_file, dtype=np.float32)
    for i, (hsi_path, false_path, box) in enumerate(zip(hsi_images_path, false_images_path, boxes)):
        tic = cv2.getTickCount()
        false_image = cv2.cvtColor(cv2.imread(false_path), cv2.COLOR_BGR2RGB)
        if data_type == 'nir-':
            hsi_image = X2Cube(hsi_path, [5,5],[5,5],25)
        elif data_type == 'vis-':
            hsi_image = X2Cube(hsi_path, [4,4],[4,4],16)
        else:
            hsi_image = X2Cube(hsi_path, [4,4],[4,4],15)
        if i == 0:
            info = {"init_bbox": box}
            tracker.initialize(false_image, info, hsi_image)
            pred_boxex.append(list(box))
        else:
            pred = tracker.track(false_image, hsi_image)['target_bbox']
            pred_boxex.append(pred)
            if vis:
                gtbox = box
                gtbox = list(map(int, gtbox))
                img = false_image.copy()
                pred_bbox = list(map(int, pred))
                cv2.rectangle(
                    img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[0] + pred_bbox[2], pred_bbox[1]+pred_bbox[3]),
                    (0,255,255), 3
                )
                cv2.rectangle(
                    img, (gtbox[0], gtbox[1]), (gtbox[0] + gtbox[2], gtbox[1]+gtbox[3]),
                    (0,255,0), 3
                )
                cv2.imshow('', img)
                cv2.waitKey(1)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    result_path = os.path.join(save_path, data_type + '{}.txt'.format(name))
    # print('({:4d})viedo: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
    #     idx,name, toc, length / toc
    # ))
    with open(result_path, 'w') as f:
        for x in pred_boxex:
            f.write(','.join([str(i) for i in x]) + '\n')
eval()
# e.report(tracker_names=['prompt_seqtrack'])