import cv2
import numpy as np
import torch
import lpips
import torch.nn.functional as F
import logging
logging.getLogger('lpips').setLevel(logging.WARNING)

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.utils.img_util import img2tensor
from torchvision.transforms.functional import normalize

@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)
        mean = [0.5]
        std = [0.5]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    # img = img.astype(np.float64)
    # img2 = img2.astype(np.float64)

    loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).cuda()  # RGB, normalized to [-1,1]

    img = img.astype(np.float32) / 255.
    img2 = img2.astype(np.float32) / 255.

    img, img2 = img2tensor([img, img2], bgr2rgb=True, float32=True)

    normalize(img, mean, std, inplace=True)
    normalize(img2, mean, std, inplace=True)

    # calculate lpips
    lpips_val = loss_fn_vgg(img.unsqueeze(0).cuda(), img2.unsqueeze(0).cuda())

    lpips_val = np.float64(round(lpips_val.item(), 6))

    return lpips_val