import numpy as np
import pandas as pd
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib
import json
import matplotlib.font_manager as fm
from collections import defaultdict
# 导入Time New Roman字体
from matplotlib import font_manager
font_path = r'D:\py\VideoX-master\SeqTrack\lib\test\tracker\Times New Roman.ttf'  # 替换为Times New Roman字体文件的路径
prop = font_manager.FontProperties(fname=font_path)

# 设置全局字体
matplotlib.rcParams['font.family'] = prop.get_name()
def region_to_bbox(region, center=True):
    n = len(region)
    assert n == 4 or n == 8, ('GT region format is invalid, should have 4 or 8 entries.')

    if n == 4:
        return _rect(region, center)
    else:
        return _poly(region, center)
nbins_iou = 50
nbins_ce = 51
# we assume the grountruth bounding boxes are saved with 0-indexing
def _rect(region, center):
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
        return np.array([cx, cy, w, h])
    else:
        return region


def _poly(region, center):
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return np.array([cx, cy, w, h])
    else:
        return np.array([cx - w / 2, cy - h / 2, w, h])

def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist

def overlap_ratio(rect1, rect2):
    '''Compute overlap ratio between two rects
    Args
        rect:2d array of N x [x,y,w,h]
    Return:
        iou
    '''
    # if rect1.ndim==1:
    #     rect1 = rect1[np.newaxis, :]
    # if rect2.ndim==1:
    #     rect2 = rect2[np.newaxis, :]
    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou

def success_overlap(gt_bb, result_bb, n_frame):
    thresholds_overlap = np.arange(0.02, 1.02, 0.02)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    mask = np.sum(gt_bb > 0, axis=1) == 4
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(iou >= thresholds_overlap[i]) / float(n_frame)
    return success
def compile_results(gt, bboxes):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    n_thresholds = 50
    ops = np.zeros(n_thresholds)
    distance_thresholds = np.linspace(1,50,50)
    dp_20 = np.zeros(50)
    precision=dp_20
    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
    for i in range(50):
        precision[i] =np.float64(sum(new_distances < distance_thresholds[i])) / np.size(new_distances)
    dp_20 = precision[19]
    average_center_location_error =new_distances.mean()
    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)

    success=success_overlap(gt4, bboxes, l)
    # integrate over the thresholds
    auc = np.mean(success)
    return precision,success,average_center_location_error, auc,dp_20

def _intersection(rects1, rects2):
    r"""Rectangle intersection.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T

def rect_iou(rects1, rects2, bound=None):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape
    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious

def center_error(rects1, rects2):
    r"""Center error.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2
    centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2
    errors = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=-1))
    return errors

def _calc_metrics(boxes, anno):
        # can be modified by children classes
        ious = rect_iou(boxes, anno)
        center_errors = center_error(boxes, anno)
        return ious, center_errors

def _calc_curves(ious, center_errors):
    ious = np.asarray(ious, float)[:, np.newaxis]
    center_errors = np.asarray(center_errors, float)[:, np.newaxis]

    thr_iou = np.linspace(0, 1, nbins_iou)[np.newaxis, :]
    thr_ce = np.arange(0, nbins_ce)[np.newaxis, :]

    bin_iou = np.greater(ious, thr_iou)
    bin_ce = np.less_equal(center_errors, thr_ce)

    succ_curve = np.mean(bin_iou, axis=0)
    prec_curve = np.mean(bin_ce, axis=0)

    return succ_curve, prec_curve


def plot_curves(tracker_names):
        report_dir = './report'
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(report_dir)
        assert os.path.exists(report_dir), \
            'No reports found. Run "report" first' \
            'before plotting curves.'
        report_file = os.path.join(report_dir, 'performance.json')
        assert os.path.exists(report_file), \
            'No reports found. Run "report" first' \
            'before plotting curves.'

        # load pre-computed performance
        with open(report_file) as f:
            performance = json.load(f)

        succ_file = os.path.join(report_dir, 'success_plots.png')
        prec_file = os.path.join(report_dir, 'precision_plots.png')
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # sort trackers by success score
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower left',fancybox=False,edgecolor='black',framealpha=0.8)

        matplotlib.rcParams.update({'font.size': 9})
        # ax.set_xlabel(fontproperties=prop)
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1.0), ylim=(0, 1),
               title='Success plots of OPE')
        ax.xaxis.set_ticks(np.arange(0,1.1,0.1))
        ax.yaxis.set_ticks(np.arange(0,1.1,0.1))
        ax.grid(True)
        fig.tight_layout()
        
        #print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

        # sort trackers by precision score
        tracker_names = list(performance.keys())
        prec = [t[key]['precision_score'] for t in performance.values()]
        inds = np.argsort(prec)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot precision curves
        thr_ce = np.arange(0, nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower right',edgecolor='black',framealpha=0.8)
        
        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Precision plots of OPE')
        ax.xaxis.set_ticks(np.arange(0,thr_ce.max()+1,5))
        ax.yaxis.set_ticks(np.arange(0,1.1,0.1))
        ax.grid(True)
        fig.tight_layout()
        
        #print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, dpi=300)

def bbox_style():
    return [
        {'color': (0,255/ 255,0)}, #绿色
        {'color': (0,0,255/255)}, #红色
        {'color': (255/255,0,0)}, #蓝色
        {'color': (255/255,255/255,0)},#青色
        {'color': (255/255,0,255/255)},#紫色
        {'color': (0,255/255,255/255)},#黄色
        {'color': (128/255,128/255,128/255)},#灰色
        {'color': (128/255, 128/255, 0)},
        {'color': (128/255, 0, 128/255)}
    ]

def plt_radar(res,dp, path, max_value, yticks, title):
    plt.style.use('ggplot')
    ours_auc    =    list(res['MVP-HOT'].values())
    ours_dp = list(dp['MVP-HOT'].values())
    feature =    list(res['MVP-HOT'].keys())
    colors = bbox_style()
    angles = np.linspace(0, 2 * np.pi, len(feature), endpoint=False)
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, polar=True)
    
    lable_angles = angles * 180 / np.pi
    lable_distances = np.ones(len(feature))
    mark_arr = ['x-', 'o-']
    for index, key in enumerate(res):
        val = list(res[key].values())
        ax.plot(np.concatenate((angles, [angles[0]])), np.concatenate((val, [val[0]])), mark_arr[index % 2], linewidth=2, label=key,color=colors[index]['color'])
    logo=[str(x)+'\n'+'('+str(f"{ours_auc[i]:.3f}") + ',' + str(f"{ours_dp[i]:.3f}") +')' for i,x in enumerate(feature)]
    ax.set_thetagrids(angles * 180 / np.pi, [])
    
    ax.set_ylim(0, max_value)
    ax.set_yticks(yticks)
    # plt.title(title, fontsize=12)
    
    for i, lable in enumerate(ax.get_xticklabels()):
        angle = lable_angles[i]
        distance = lable_distances[i]
        x = np.deg2rad(angle)
        y = distance - 0.05
        plt.text(x, y, logo[i], ha='center', va='center')
    
    
    plt.legend(loc='lower center', prop={'size':10},ncol=4, bbox_to_anchor=[0.5, -0.25]) # 控制图标的居中和往下的距离
    ax.grid(True)
    # plt.show()
    plt.savefig(path,dpi=300, bbox_inches='tight')

def eval():
    # gt = pd.read_table(,header=None)
    # gt=gt.dropna(axis=1, how='all').to_numpy()
    tracker_names = ['SiamGAT','TransT','SiamCAR','SeqTrack','MVP-HOT','SiamBAN','MHT', 'STARK']
    # s_types = ['base', 'prompt']
    performance = {}
    report_dir = './report/'
    report_file = os.path.join(report_dir, 'performance.json')
    
    performance = {}
    tracker_video_type_auc = defaultdict(lambda: defaultdict(float))
    tracker_video_type_dp = defaultdict(lambda: defaultdict(float))
    for name in tracker_names:
        result_base_path = './result'
        res = os.path.join(result_base_path, name)
    
        all_list = sorted(glob.glob(res + '/*.txt'))
        gt_path = 'F:/whisper2023/validation'
    
        gt_nir_path = os.path.join(gt_path, 'HSI-NIR-FalseColor')
        gt_vis_path = os.path.join(gt_path, 'HSI-VIS-FalseColor')
        gt_red_path = os.path.join(gt_path, 'HSI-RedNIR-FalseColor')
    
        # viedo_length = len(glob.glob(gt_nir_path + '/*'))
        viedo_length = len(glob.glob(gt_nir_path + '/*') + glob.glob(gt_red_path + '/*') + glob.glob(gt_vis_path + '/*'))
        succ_curve = np.zeros((viedo_length, nbins_iou))#IOU阈值21 成功率
        prec_curve = np.zeros((viedo_length, nbins_ce))
    
        performance.update({name: {'overall': {},'seq_wise': {}}})
        count = 0
        viedo_type_dict_auc = defaultdict(list)
        viedo_type_dict_dp = defaultdict(list)
        for s, res_path in enumerate(all_list):
            seq_name = res_path.split('\\')[-1]
            type, t = seq_name.split('-')
            # if type != 'nir': continue
            gt_path = ''
            if type == 'vis':
                gt_path = gt_vis_path
            elif type == 'rednir':
                gt_path = gt_red_path
            else:
                gt_path = gt_nir_path 
            gt_path += '/' + t[:-4]
            gt = np.loadtxt(gt_path + '/groundtruth_rect.txt')
            type_arr = []
            with open(gt_path + '/description.txt', 'r') as file:
                type_arr = [line.strip() for line in file.readlines() if line.strip()]
                if ('Low resolution' in type_arr):
                    print(111)
            try:
                res = np.loadtxt(res_path ,delimiter=',')
            except:
                res = np.loadtxt(res_path)
            dp, op, cle, auc, dp_20 = compile_results(gt, res)
            ious, center_errors = _calc_metrics(gt, res)
            succ_curve[count], prec_curve[count] = _calc_curves(ious, center_errors)
            succ_curve[count] = op
            for video_type in type_arr:
                viedo_type_dict_auc[video_type].append(np.mean(op))
                viedo_type_dict_dp[video_type].append(np.mean(prec_curve[count]))
            performance[name]['seq_wise'].update({type + seq_name: {
                    'success_curve': succ_curve[count].tolist(),
                    'precision_curve': prec_curve[count].tolist(),
                    'success_score': np.mean(succ_curve[count]),
                    'precision_score': prec_curve[count][20],
                    'success_rate': succ_curve[count][nbins_iou // 2],
                    # 'speed_fps': speeds[s] if speeds[s] > 0 else -1
                    }})
            count += 1
            # s = 'auc:%.3f, dp_20:%.3f' % (float(auc), float(dp_20))
            # print(type +'-' + seq_name+ ':' + s)
        succ_curve = np.mean(succ_curve, axis=0)
        prec_curve = np.mean(prec_curve, axis=0)
        succ_score = np.mean(succ_curve)
        prec_score = prec_curve[20]
        succ_rate = succ_curve[nbins_iou // 2]
        print(name, succ_score, prec_score)
        for key in viedo_type_dict_auc:
            auc_res = round(sum(viedo_type_dict_auc[key]) / len(viedo_type_dict_auc[key]), 3)
            dp_res = round(sum(viedo_type_dict_dp[key]) / len(viedo_type_dict_dp[key]), 3)
            tracker_video_type_auc[name][key] = auc_res
            tracker_video_type_dp[name][key] = dp_res

        performance[name]['overall'].update({
                'success_curve': succ_curve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'success_score': succ_score,
                'precision_score': prec_score,
                'success_rate': succ_rate,
                # 'speed_fps': avg_speed
                })
    
    with open(report_file, 'w') as f:
        json.dump(performance, f, indent=4)
            
    plot_curves(tracker_names)
    plt_radar(tracker_video_type_auc,tracker_video_type_dp, './report/auc_radar.png', 0.8, [0, 0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7 ], 'auc')
    
if __name__ == '__main__':
    eval()

