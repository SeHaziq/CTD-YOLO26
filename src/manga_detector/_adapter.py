"""Optional YOLO block adapter around unchanged CTD line/mask/grouping code."""
from pathlib import Path
import cv2
import numpy as np
import torch
from ._ctd.inference import TextDetector,preprocess_img,postprocess_yolo,postprocess_mask
from ._ctd.utils.textblock import group_output
from ._ctd.utils.textmask import refine_mask,REFINEMASK_INPAINT


def adapt_boxes(boxes,scores,legacy,shape,transfer_iou=.5):
    """Convert original-coordinate one-class boxes; preserve scores/language provenance."""
    h,w=shape[:2]
    boxes=np.array(boxes,dtype=np.float64,copy=True).reshape(-1,4)
    scores=np.asarray(scores,dtype=np.float64).reshape(-1)
    if len(boxes)!=len(scores):raise ValueError('Box and score counts differ')
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():raise ValueError('Non-finite detector output')
    if np.any((scores<0)|(scores>1)):raise ValueError('Invalid confidence')
    boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w-1)
    boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h-1)
    boxes=boxes.astype(np.int32)
    keep=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
    boxes,scores=boxes[keep],scores[keep]
    classes=np.full(len(boxes),2,dtype=np.int32);sources=[]
    oldboxes,oldclasses,_=legacy
    oldboxes=np.asarray(oldboxes,dtype=float).reshape(-1,4)
    for i,b in enumerate(boxes):
        if len(oldboxes):
            inter=np.maximum(np.minimum(oldboxes[:,2:],b[2:])-np.maximum(oldboxes[:,:2],b[:2]),0).prod(axis=1)
            union=(oldboxes[:,2:]-oldboxes[:,:2]).clip(0).prod(axis=1)+(b[2:]-b[:2]).prod()-inter
            overlap=inter/np.maximum(union,1e-9);j=int(overlap.argmax())
            if overlap[j]>=transfer_iou:
                classes[i]=oldclasses[j];sources.append('overlapping_ctd_language');continue
        sources.append('unknown')
    return (boxes,classes,scores),sources


def preserve_ctd_boxes(adapted,language_sources,legacy,overlap_threshold=.5):
    """Keep CTD boxes and add YOLO boxes only outside heavily overlapping regions."""
    boxes,classes,scores=adapted
    oldboxes,oldclasses,oldscores=legacy
    oldboxes=np.asarray(oldboxes,dtype=np.int32).reshape(-1,4)
    accepted=[]
    for i,b in enumerate(boxes):
        if len(oldboxes):
            inter=np.maximum(np.minimum(oldboxes[:,2:],b[2:])-np.maximum(oldboxes[:,:2],b[:2]),0).prod(axis=1)
            denominator=np.minimum((oldboxes[:,2:]-oldboxes[:,:2]).clip(0).prod(axis=1),(b[2:]-b[:2]).prod())
            if np.max(inter/np.maximum(denominator,1))>=overlap_threshold:continue
        accepted.append(i)
    ids=np.array(accepted,dtype=np.int64)
    merged=(np.concatenate([oldboxes,boxes[ids]]),np.concatenate([np.asarray(oldclasses,dtype=np.int32),classes[ids]]),
            np.concatenate([np.asarray(oldscores,dtype=float),scores[ids]]))
    return merged,['ctd_language']*len(oldboxes)+[language_sources[i] for i in ids],['ctd']*len(oldboxes)+['yolo26']*len(ids)


class HybridDetector:
    def __init__(self,cfg,load_yolo=True):
        self.cfg=cfg
        if cfg.get('block_policy','replace') not in {'replace','preserve_ctd'}:raise ValueError('Unknown block policy')
        self.ctd=TextDetector(str(cfg['ctd_weights']),input_size=cfg['input_size'],device=cfg['device'])
        self.yolo=None
        if load_yolo:
            from ultralytics import YOLO
            self.yolo=YOLO(str(cfg['yolo_weights']))

    @torch.no_grad()
    def __call__(self,image,backend='yolo26'):
        if backend not in {'ctd','yolo26'}:raise ValueError('Unknown backend')
        if backend=='yolo26' and self.yolo is None:raise RuntimeError('YOLO26 was not loaded; use the CTD backend or enable the hybrid')
        cfg=self.cfg;n=cfg['input_size'];h,w=image.shape[:2]
        inp,_,dw,dh=preprocess_img(image,input_size=(n,n),device=cfg['device'])
        raw_blocks,raw_mask,db=self.ctd.net(inp)
        scale=(w/(n-dw),h/(n-dh))
        legacy=postprocess_yolo(raw_blocks,.4,.35,scale)
        mask=postprocess_mask(raw_mask)
        quads,scores=self.ctd.seg_rep((n,n),db)
        keep=np.where(scores[0]>.6);lines=quads[0][keep];line_scores=scores[0][keep]
        mask=cv2.resize(mask[:mask.shape[0]-dh,:mask.shape[1]-dw],(w,h),interpolation=cv2.INTER_LINEAR)
        if lines.size:
            lines=lines.astype(float);lines[...,0]*=scale[0];lines[...,1]*=scale[1];lines=lines.astype(np.int32)
        else:lines=np.empty((0,4,2),np.int32)
        blocks=legacy;language_sources=['ctd_language']*len(legacy[0]);detector_sources=['ctd']*len(legacy[0])
        if backend=='yolo26':
            result=self.yolo.predict(image,imgsz=n,device=0 if cfg['device']=='cuda' else 'cpu',
                     conf=cfg['yolo_confidence'],iou=cfg['yolo_nms_iou'],rect=False,verbose=False)[0]
            blocks,language_sources=adapt_boxes(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),legacy,image.shape,cfg['language_transfer_iou'])
            detector_sources=['yolo26']*len(blocks[0])
            if cfg.get('block_policy','replace')=='preserve_ctd':
                blocks,language_sources,detector_sources=preserve_ctd_boxes(blocks,language_sources,legacy,cfg.get('preserve_overlap_threshold',.5))
        # These routines can mutate line/block objects; each call owns fresh arrays.
        grouped=group_output(blocks,lines.copy(),w,h,mask)
        refined=refine_mask(image,mask,grouped,refine_mode=REFINEMASK_INPAINT)
        return {'raw_blocks':blocks[0],'raw_block_class':blocks[1],'raw_block_confidence':blocks[2],
                'raw_block_language_sources':language_sources,'raw_lines':lines,'raw_line_confidence':line_scores,
                'raw_block_detector_sources':detector_sources,'block_policy':cfg.get('block_policy','replace'),
                'mask':mask,'refined':refined,'grouped':grouped,'backend':backend,
                'note':'Grouped lines may include CTD synthetic fallback rectangles. Raw DB lines are unchanged by the block adapter.'}
