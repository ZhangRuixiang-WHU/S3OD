import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmdet.core.mask.structures import BitmapMasks
from mmrotate.core.bbox.transforms import poly2obb_le90, obb2poly_le90
from torch.nn import functional as F


import sklearn.mixture as skm

def resize_image(inputs, resize_ratio=0.5):
    down_inputs = F.interpolate(inputs, 
                                scale_factor=resize_ratio, 
                                mode='nearest')
    
    return down_inputs


def pop_elements(_list, count):
    for idx in range(count):
        _list.pop(idx)
    return _list

def pointobb2thetaobb(pointobb):
    return poly2obb_le90(pointobb)

def thetaobb2pointobb(thetaobb):
    return obb2poly_le90(thetaobb).reshape(
        -1, 2
    ) 

def bbox2points(box):
    min_x, min_y, max_x, max_y = torch.split(box[:, :4], [1, 1, 1, 1], dim=1)

    return torch.cat(
        [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dim=1
    ).reshape(
        -1, 2
    )  # n*4,2


def points2bbox(point, max_w, max_h):
    point = point.reshape(-1, 4, 2)
    if point.size()[0] > 0:
        min_xy = point.min(dim=1)[0]
        max_xy = point.max(dim=1)[0]
        xmin = min_xy[:, 0].clamp(min=0, max=max_w)
        ymin = min_xy[:, 1].clamp(min=0, max=max_h)
        xmax = max_xy[:, 0].clamp(min=0, max=max_w)
        ymax = max_xy[:, 1].clamp(min=0, max=max_h)
        min_xy = torch.stack([xmin, ymin], dim=1)
        max_xy = torch.stack([xmax, ymax], dim=1)
        return torch.cat([min_xy, max_xy], dim=1)  # n,4
    else:
        return point.new_zeros(0, 4)


def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor."""
    if not isinstance(obj, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(obj)))


def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    tr_mat = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: Tuple[int, int],
    dsize_dst: Tuple[int, int],
) -> torch.Tensor:
    check_is_tensor(dst_pix_trans_src_pix)

    if not (
        len(dst_pix_trans_src_pix.shape) == 3
        or dst_pix_trans_src_pix.shape[-2:] == (3, 3)
    ):
        raise ValueError(
            "Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}".format(
                dst_pix_trans_src_pix.shape
            )
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(src_h, src_w).to(
        dst_pix_trans_src_pix
    )
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix.float()).to(
        src_norm_trans_src_pix.dtype
    )
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(dst_h, dst_w).to(
        dst_pix_trans_src_pix
    )

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (
        dst_pix_trans_src_pix @ src_pix_trans_src_norm
    )
    return dst_norm_trans_src_norm


def warp_affine(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    if not isinstance(src, torch.Tensor):
        raise TypeError(
            "Input src type is not a torch.Tensor. Got {}".format(type(src))
        )

    if not isinstance(M, torch.Tensor):
        raise TypeError("Input M type is not a torch.Tensor. Got {}".format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}".format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}".format(M.shape))

    # TODO: remove the statement below in kornia v0.6
    if align_corners is None:
        message: str = (
            "The align_corners default value has been changed. By default now is set True "
            "in order to match cv2.warpAffine."
        )
        warnings.warn(message)
        # set default value for align corners
        align_corners = True

    B, C, H, W = src.size()

    # we generate a 3x3 transformation matrix from 2x3 affine

    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), dsize)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm.float())

    grid = F.affine_grid(
        src_norm_trans_dst_norm[:, :2, :],
        [B, C, dsize[0], dsize[1]],
        align_corners=align_corners,
    )

    return F.grid_sample(
        src.float(),
        grid,
        align_corners=align_corners,
        mode=mode,
        padding_mode=padding_mode,
    ).to(src.dtype)


class Transform2D:
    @staticmethod
    def transform_rbboxes(bbox, M, out_shape): ## change to rbbox
        if isinstance(bbox, Sequence):
            assert len(bbox) == len(M)
            return [
                Transform2D.transform_rbboxes(b, m, o)
                for b, m, o in zip(bbox, M, out_shape)
            ]
        else:
            if bbox.shape[0] == 0:
                return bbox
            score = None
            if bbox.shape[1] > 5:
                score = bbox[:, 5:]
            points = thetaobb2pointobb(bbox[:, :5])
            points = torch.cat(
                [points, points.new_ones(points.shape[0], 1)], dim=1
            )  # n,3
            points = torch.matmul(M, points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = pointobb2thetaobb(points)#, out_shape[1], out_shape[0])
            if score is not None:
                return torch.cat([bbox, score], dim=1)
            return bbox

    @staticmethod
    def transform_bboxes(bbox, M, out_shape):
        if isinstance(bbox, Sequence):
            assert len(bbox) == len(M)
            return [
                Transform2D.transform_bboxes(b, m, o)
                for b, m, o in zip(bbox, M, out_shape)
            ]
        else:
            if bbox.shape[0] == 0:
                return bbox
            score = None
            if bbox.shape[1] > 4:
                score = bbox[:, 4:]
            points = bbox2points(bbox[:, :4])
            points = torch.cat(
                [points, points.new_ones(points.shape[0], 1)], dim=1
            )  # n,3
            points = torch.matmul(M, points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = points2bbox(points, out_shape[1], out_shape[0])
            if score is not None:
                return torch.cat([bbox, score], dim=1)
            return bbox

    @staticmethod
    def transform_masks(
        mask: Union[BitmapMasks, List[BitmapMasks]],
        M: Union[torch.Tensor, List[torch.Tensor]],
        out_shape: Union[list, List[list]],
    ):
        if isinstance(mask, Sequence):
            assert len(mask) == len(M)
            return [
                Transform2D.transform_masks(b, m, o)
                for b, m, o in zip(mask, M, out_shape)
            ]
        else:
            if mask.masks.shape[0] == 0:
                return BitmapMasks(np.zeros((0, *out_shape)), *out_shape)
            mask_tensor = (
                torch.from_numpy(mask.masks[:, None, ...]).to(M.device).to(M.dtype)
            )
            return BitmapMasks(
                warp_affine(
                    mask_tensor,
                    M[None, ...].expand(mask.masks.shape[0], -1, -1),
                    out_shape,
                )
                .squeeze(1)
                .cpu()
                .numpy(),
                out_shape[0],
                out_shape[1],
            )

    @staticmethod
    def transform_image(img, M, out_shape):
        if isinstance(img, Sequence):
            assert len(img) == len(M)
            return [
                Transform2D.transform_image(b, m, shape)
                for b, m, shape in zip(img, M, out_shape)
            ]
        else:
            if img.dim() == 2:
                img = img[None, None, ...]
            elif img.dim() == 3:
                img = img[None, ...]

            return (
                warp_affine(img.float(), M[None, ...], out_shape, mode="nearest")
                .squeeze()
                .to(img.dtype)
            )

def thr_select_policy(scores, given_gt_thr=0.5, percent=35, vaild_len=100):
    """The policy of choosing pseudo label

    The previous GMM-B policy is used as default.
    1. Use the predicted bbox to fit a GMM with 2 center.
    2. Find the predicted bbox belonging to the positive
        cluster with highest GMM probability.
    3. Take the class score of the finded bbox as gt_thr.

    Args:
        scores (nd.array): The scores.

    Returns:
        float: Found gt_thr.

    """
    if len(scores) < vaild_len:
        return given_gt_thr
    
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(scores,list):
        scores = np.array(scores)
    if len(scores.shape) == 1:
        scores = scores[:, np.newaxis]
        
        
    if isinstance(percent,int):
        pos_thr = np.percentile(scores,percent)
        return pos_thr
    else:
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        pos_thr = (gmm.means_[0][0]+gmm.means_[1][0])/2
        
        if percent == 'gmm' or  percent == 'gmm-mean':
            return pos_thr
        else:
            print("The percent of gmm must be in ['gmm', 'gmm-mean']")



def get_adaptive_size1(bbox_list, label_list,vaild_len=10,percent=30,min=0.5,max=0.9):
    class_metrix = {'tiny':[], 'small':[], 'nomal':[]}
    cls_thr = {'tiny':0.9, 'small':0.9, 'nomal':0.9}

    for proposal, proposal_label in zip(bbox_list, label_list):
        if proposal.size(0) == 0:
            pass
        else:
            for box, cls in zip(proposal,proposal_label):
                if (box[2] * box [3]) >= 1024:
                    class_metrix['nomal'].append(box[5].item())
                elif 400< (box[2] * box [3]) < 1024:
                    class_metrix['small'].append(box[5].item())
                else:
                    class_metrix['tiny'].append(box[5].item())
    for key,value in class_metrix.items():
        if len(value) > vaild_len:
            # cls_thr[key] = np.clip(np.percentile(value,percent),min,max)
            thr = thr_select_policy(value, given_gt_thr=0.5, percent=percent, vaild_len= vaild_len)
            cls_thr[key] = np.clip(thr,min,max)
        else:
            cls_thr[key] = np.clip(cls_thr[key],min,max)
    return cls_thr

def filter_invalid_with_adaptive_thr_size1(bbox, label, score, cls_thr, min_size=0):
    if bbox.size(0) == 0:
        return bbox,label
    else:
        size_label = []
        for box in bbox:
            if (box[2] * box [3]) >= 1024:
                size_label.append('nomal')
            elif 400< (box[2] * box [3]) < 1024:
                size_label.append('small')
            else:
                size_label.append('tiny')
        valid = torch.tensor([sco > cls_thr[cls] for sco, cls in zip(score,size_label)])
        bbox = bbox[valid]
        label = label[valid]
        if min_size is not None:
            bw = bbox[:, 2]
            bh = bbox[:, 3]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
            label = label[valid]
        return bbox, label


def get_adaptive_size2(bbox_list, label_list,vaild_len=20,percent=30,min=0.8,max=0.96, small_size = 800):
    class_metrix = {'small':[], 'nomal':[]}
    cls_thr = {'small':0.9, 'nomal':0.9}

    for proposal, proposal_label in zip(bbox_list, label_list):
        if proposal.size(0) == 0:
            pass
        else:
            for box, cls in zip(proposal,proposal_label):
                if (box[2] * box [3]) >= small_size:
                    class_metrix['nomal'].append(box[5].item())
                else:
                    class_metrix['small'].append(box[5].item())
    for key,value in class_metrix.items():
        if len(value) > vaild_len:
            thr = thr_select_policy(value,given_gt_thr=0.5, percent=percent, vaild_len= vaild_len) #np.clip(np.percentile(value,percent),min,max)
            cls_thr[key] = np.clip(thr,min,max)
        else:
            cls_thr[key] = np.clip(cls_thr[key],min,max)
    return cls_thr

def get_adaptive_size(class_metrix, bbox_list, label_list,percent=30,min=0.8,max=0.96):
    # class_metrix = {'small':[], 'nomal':[]}
    cls_thr = {'small':0.9, 'nomal':0.9}

    for proposal, proposal_label in zip(bbox_list, label_list):
        if proposal.size(0) == 0:
            pass
        else:
            for box, cls in zip(proposal,proposal_label):
                if (box[2] * box [3]) >= 800:
                    if len(class_metrix['nomal'])>=200:
                        class_metrix['nomal'].pop(0)
                    class_metrix['nomal'].append(box[5].item())
                else:
                    if len(class_metrix['small'])>=200:
                        class_metrix['small'].pop(0)
                    class_metrix['small'].append(box[5].item())
    for key,value in class_metrix.items():
        if len(value) > 100:
            cls_thr[key] = np.clip(np.percentile(value,percent),min,max)
        else:
            cls_thr[key] = np.clip(cls_thr[key],min,max)
    return cls_thr


def filter_invalid_with_adaptive_thr_size2(bbox, label, score, cls_thr, min_size=0, small_size = 800):
    if bbox.size(0) == 0:
        return bbox,label
    else:
        size_label = []
        for box in bbox:
            if (box[2] * box [3]) >= small_size:
                size_label.append('nomal')
            else:
                size_label.append('small')
        valid = torch.tensor([sco > cls_thr[cls] for sco, cls in zip(score,size_label)])
        bbox = bbox[valid]
        label = label[valid]
        if min_size is not None:
            bw = bbox[:, 2]
            bh = bbox[:, 3]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
            label = label[valid]
        return bbox, label

def get_adaptive_thr(bbox_list, label_list,vaild_len=10,percent=30,min=0.5,max=0.9):
    class_metrix = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[]}
    cls_thr = {0:0.9, 1:0.9, 2:0.9, 3:0.9, 4:0.9, 5:0.9, 6:0.9, 7:0.9, 8:0.9, 9:0.9, 10:0.9, 11:0.9, 12:0.9, 13:0.9, 14:0.9, 15:0.9}

    for proposal, proposal_label in zip(bbox_list, label_list):
        for box, cls in zip(proposal,proposal_label):
            class_metrix[cls.item()].append(box[5].item())
    for key,value in class_metrix.items():
        if len(value) > vaild_len:
            cls_thr[key] = np.clip(np.percentile(value,percent),min,max)
        else:
            cls_thr[key] = np.clip(cls_thr[key],min,max)
    return cls_thr

def filter_invalid_with_adaptive_thr_cls(bbox, label, score, cls_thr, min_size=0):
    if bbox.size(0) == 0:
        return bbox,label
    else:
        valid = torch.tensor([sco > cls_thr[cls.item()] for sco, cls in zip(score,label)])
        bbox = bbox[valid]
        label = label[valid]
        if min_size is not None:
            bw = bbox[:, 2]
            bh = bbox[:, 3]
            valid = (bw > min_size) & (bh > min_size)
            bbox = bbox[valid]
            label = label[valid]
        return bbox, label

def filter_invalid(bbox, label=None, score=None, mask=None, thr=0.0, min_size=0):
    if score is not None:
        valid = score > thr
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    if min_size is not None:
        bw = bbox[:, 2]
        bh = bbox[:, 3]
        # bw = bbox[:, 2] - bbox[:, 0]
        # bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    return bbox, label, mask

def filter_invalid_scr(bbox, label=None, score=None, mask=None, thr=0.0, min_size=0):
    if score is not None:
        valid = score > thr
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
        score = score[valid]
    if min_size is not None:
        bw = bbox[:, 2]
        bh = bbox[:, 3]
        # bw = bbox[:, 2] - bbox[:, 0]
        # bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
        score = score[valid]
    score = 1.0 + score 
    return bbox, score, label, mask

def filter_invalid_scr2(bbox, label=None, score=None, mask=None, thr=0.0, min_size=0):
    if score is not None:
        valid = score > thr
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
        score = score[valid]
    if min_size is not None:
        bw = bbox[:, 2]
        bh = bbox[:, 3]
        # bw = bbox[:, 2] - bbox[:, 0]
        # bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
        score = score[valid]
    score = 1.0 - (score * score)
    return bbox, score, label, mask



def filter_invalid_scr_TNL_mode4(bbox, label=None, score=None, mask=None, thr=0.0, min_size=0):
    if score is not None:
        valid = score > thr
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
        score = score[valid]
    if min_size is not None:
        bw = bbox[:, 2]
        bh = bbox[:, 3]
        # bw = bbox[:, 2] - bbox[:, 0]
        # bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
        score = score[valid]
    score = 1.0 - (score)
    return bbox, score, label, mask



def filter_invalid_org(bbox, label=None, score=None, mask=None, thr=0.0, min_size=0):
    if score is not None:
        valid = score > thr
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    if min_size is not None:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)
        bbox = bbox[valid]
        if label is not None:
            label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    return bbox, label, mask