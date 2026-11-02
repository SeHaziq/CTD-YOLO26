# Adapted from dmMaze/comic-text-detector, commit 440b978563c71b758e31aaa315d100faba1efa2f.
# GPL-3.0; see LICENSES/GPL-3.0.txt. Namespaced and limited to inference for library integration.
import cv2
import numpy as np
import torch
from typing import Union
from .basemodel import TextDetBase
from .utils.imgproc_utils import letterbox
from .utils.yolov5_utils import non_max_suppression
from .utils.db_utils import SegDetectorRepresenter
def preprocess_img(img, input_size=(1024, 1024), device='cpu', bgr2rgb=True, half=False, to_tensor=True):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
    if to_tensor:
        img_in = img_in.transpose((2, 0, 1))[::-1]
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
        if to_tensor:
            img_in = torch.from_numpy(img_in).to(device)
            if half:
                img_in = img_in.half()
    return (img_in, ratio, int(dw), int(dh))

def postprocess_mask(img: Union[torch.Tensor, np.ndarray], thresh=None):
    if isinstance(img, torch.Tensor):
        img = img.squeeze_()
        if img.device != 'cpu':
            img = img.detach_().cpu()
        img = img.numpy()
    else:
        img = img.squeeze()
    if thresh is not None:
        img = img > thresh
    img = img * 255
    return img.astype(np.uint8)

def postprocess_yolo(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
    det = non_max_suppression(det, conf_thresh, nms_thresh)[0]
    if det.device != 'cpu':
        det = det.detach_().cpu().numpy()
    det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
    det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
    if sort_func is not None:
        det = sort_func(det)
    blines = det[..., 0:4].astype(np.int32)
    confs = np.round(det[..., 4], 3)
    cls = det[..., 5].astype(np.int32)
    return (blines, cls, confs)
class TextDetector:
    def __init__(self, model_path, input_size=1024, device="cpu"):
        self.net = TextDetBase(model_path, device=device)
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)
