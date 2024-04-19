import torch
import math
import numpy as np
import cv2 as cv
import torch.nn.functional as F
from lib.utils.misc import NestedTensor


class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

        
        self.hsi_vis_mean = torch.tensor([0.09264746, 0.08855829, 0.08247536, 0.08143572, 0.07748746, 0.0890622,
 0.10415976, 0.10092119, 0.10402559, 0.11850767, 0.08558685, 0.08982185,
 0.07610605, 0.09552862, 0.10765684, 0.13679735]).view((1, 16, 1, 1)).cuda()
        self.hsi_vis_std = torch.tensor([0.09521574, 0.08823087, 0.08089152, 0.07771741, 0.0729734,  0.08347688,
 0.09561309, 0.08946099, 0.09642888, 0.10841756, 0.07613651, 0.07741433,
 0.07978947, 0.08000497, 0.08878143, 0.11464765]).view((1, 16, 1, 1)).cuda()
        
        self.hsi_nir_mean = torch.tensor([0.22888592, 0.2012517, 0.21936508, 0.23368526, 0.26842279, 0.27764923,
 0.28093583, 0.29102584, 0.28420158, 0.29318393, 0.27188997, 0.26692739,
 0.2757229,  0.28376873, 0.28079491, 0.28172783, 0.28375229, 0.28455873,
 0.24056335, 0.2366674,  0.24797409, 0.23813221, 0.24698492, 0.22444439,
 0.25095202] ).view((1, 25, 1, 1)).cuda()
        self.hsi_nir_std = torch.tensor([0.02372335, 0.01984755, 0.01946667, 0.01978092, 0.0245898,  0.02334601,
 0.02592662, 0.02704679, 0.02594258, 0.02427856, 0.02230078, 0.02248841,
 0.02181426, 0.02199267, 0.02187123, 0.02220285, 0.02096013, 0.01911544,
 0.01753863, 0.01754301, 0.01814249, 0.01730974, 0.01840824, 0.01723943,
 0.0180452 ]).view((1, 25, 1, 1)).cuda()
        self.hsi_red_mean = torch.tensor([0.34968978, 0.26896745, 0.20409033, 0.14713213, 0.10182434, 0.07621241,
 0.0589028,  0.04882661, 0.04681665, 0.0406662,  0.03926101, 0.04065134,
 0.03377544, 0.03037895, 0.03295941]).view((1, 15, 1, 1)).cuda()
        
        self.hsi_red_std = torch.tensor([0.24439706, 0.19852264, 0.15069625, 0.1110164,  0.07371341, 0.05517538,
 0.04201854, 0.03733097, 0.0363201,  0.03250302, 0.03137494, 0.03268342,
 0.0289234,  0.0257265,  0.02840968]).view((1, 15, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        if img_arr.shape[-1] == 16:
            img_tensor_norm = ((img_tensor / img_arr.max()) - self.hsi_vis_mean) / self.hsi_vis_std
        elif img_arr.shape[-1] == 25:
            img_tensor_norm = ((img_tensor / img_arr.max()) - self.hsi_nir_mean) / self.hsi_nir_std
        elif img_arr.shape[-1] == 15:
            img_tensor_norm = ((img_tensor / img_arr.max()) - self.hsi_red_mean) / self.hsi_red_std
        else:
            img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        return img_tensor_norm


def sample_target(im, target_bb, search_area_factor, output_sz=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))

        return im_crop_padded, resize_factor

    else:
        return im_crop_padded, 1.0

def resize_sample_target(im, output_sz=None):
    """ Resize the image

    args:
        im - cv image
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

    # Resize image
    # deal with attention mask
    H, W, _ = im.shape
    if output_sz is not None:
        resize_factor = (output_sz / W, output_sz / H)  # (w,h) rather than (h,w)
        im_resized = cv.resize(im, (output_sz, output_sz))
        return im_resized, resize_factor
    else:
        return im, 1.0

def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / (crop_sz[0]-1)
    else:
        return box_out