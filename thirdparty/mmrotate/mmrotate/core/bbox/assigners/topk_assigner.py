import torch
import json
import numpy

from ..builder import build_bbox_coder
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from ..builder import ROTATED_BBOX_ASSIGNERS
#from mmcv.utils import build_from_cfg




@ROTATED_BBOX_ASSIGNERS.register_module()
class TopKAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            `min_pos_iou` is set to avoid assigning bboxes that have extremely
            small iou with GT as positive samples. It brings about 0.3 mAP
            improvements in 1x schedule but does not affect the performance of
            3x schedule. More comparisons can be found in
            `PR #7464 <https://github.com/open-mmlab/mmdetection/pull/7464>`_.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 assign_metric='iou',
                 topk=3):
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None, ast_bboxes=None, cls_scores = None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        # overlaps = self.iou_calculator(gt_bboxes, bboxes, mode=self.assign_metric)
        overlaps = self.iou_calculator(gt_bboxes,  bboxes, mode=self.assign_metric, ast_bboxes=ast_bboxes)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        if cls_scores is not None:
            assign_result = self.assign_wrt_ranking_cls(overlaps, gt_labels, cls_scores)
        else:
            assign_result = self.assign_wrt_ranking(overlaps, gt_labels)

        # debug = False
        # if gt_labels is None: # means the assign process is in RPN stage, rather than R-CNN stage
        #     gt_sizes = gt_bboxes[:,2] * gt_bboxes[:,3]
        #     sizes_0_8 = gt_sizes <= 64
        #     sizes_0_8_id = (torch.where(sizes_0_8==True)[0]).add(1)
        #     num_0_8 = len(sizes_0_8_id)
        #     sizes_8_16 = (gt_sizes > 64) & (gt_sizes <= 256)
        #     sizes_8_16_id = (torch.where(sizes_8_16==True)[0]).add(1)
        #     num_8_16 = len(sizes_8_16_id)
        #     sizes_16_24 = (gt_sizes > 256) & (gt_sizes <= 796)
        #     sizes_16_24_id = (torch.where(sizes_16_24==True)[0]).add(1)
        #     num_16_24 = len(sizes_16_24_id)
        #     sizes_24_32 = (gt_sizes > 796) & (gt_sizes <= 1024)
        #     sizes_24_32_id = (torch.where(sizes_24_32==True)[0]).add(1)
        #     num_24_32 = len(sizes_24_32_id)
        #     sizes_32_48 = (gt_sizes > 1024) & (gt_sizes <= 2304)
        #     sizes_32_48_id = (torch.where(sizes_32_48==True)[0]).add(1)
        #     num_32_48 = len(sizes_32_48_id)
        #     sizes_48_64 = (gt_sizes > 2304) & (gt_sizes <= 4096)
        #     sizes_48_64_id = (torch.where(sizes_48_64==True)[0]).add(1)
        #     num_48_64 = len(sizes_48_64_id)
        #     sizes_64_96 = (gt_sizes > 4096) & (gt_sizes <= 9216)
        #     sizes_64_96_id = (torch.where(sizes_64_96==True)[0]).add(1)
        #     num_64_96 = len(sizes_64_96_id)
        #     sizes_96_ = (gt_sizes > 9216)
        #     sizes_96_id = (torch.where(sizes_96_==True)[0]).add(1)
        #     num_96_ = len(sizes_96_id)

        #     anchor_0_8, anchor_8_16, anchor_16_24, anchor_24_32, anchor_32_48, anchor_48_64, anchor_64_96, anchor_96_,= 0,0,0,0,0,0,0,0
        #     for s_idx in sizes_0_8_id:
        #         anchor_0_8 += (torch.where(assign_result.gt_inds==s_idx)[0]).size(0)
        #     for s_idx in sizes_8_16_id:
        #         anchor_8_16 += (torch.where(assign_result.gt_inds==s_idx)[0]).size(0)
        #     for s_idx in sizes_16_24_id:
        #         anchor_16_24 += (torch.where(assign_result.gt_inds==s_idx)[0]).size(0)
        #     for s_idx in sizes_24_32_id:
        #         anchor_24_32 += (torch.where(assign_result.gt_inds==s_idx)[0]).size(0)
        #     for s_idx in sizes_32_48_id:
        #         anchor_32_48 += (torch.where(assign_result.gt_inds==s_idx)[0]).size(0)
        #     for s_idx in sizes_48_64_id:
        #         anchor_48_64 += (torch.where(assign_result.gt_inds==s_idx)[0]).size(0)
        #     for s_idx in sizes_64_96_id:
        #         anchor_64_96 += (torch.where(assign_result.gt_inds==s_idx)[0]).size(0)
        #     for s_idx in sizes_96_id:
        #         anchor_96_ += (torch.where(assign_result.gt_inds==s_idx)[0]).size(0)
        #     if debug:
        #         FilePath = "/home/zrx/ssod/SoftTeacher/work_dirs/s3od_r1_sla_percent.txt"
        #         with open(FilePath, "a") as filewrite:   #”w"代表着每次运行都覆盖txt的内容
        #             filewrite.write(str(num_0_8) + " " + str(anchor_0_8) + " " + str(num_8_16) + " " + str(anchor_8_16) + " " + str(num_16_24) + " " + str(anchor_16_24) + " " + str(num_24_32) + " " + str(anchor_24_32) + " " + str(num_32_48) + " " + str(anchor_32_48) + " " + str(num_48_64) + " " + str(anchor_48_64) + " " + str(num_64_96) + " " + str(anchor_64_96) + " " + str(num_96_) + " " + str(anchor_96_) +"\n")
 

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_ranking(self,  overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0

        for j in range(self.topk):
            for i in range(num_gts):
                inj = self.topk-1-j
                assigned_gt_inds[gt_argmax_overlaps[i,inj]] = i + 1

        # for i in range(num_gts):
        #     for j in range(self.topk):
        #         max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
        #         assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def assign_wrt_ranking_cls(self,  overlaps, gt_labels=None, cls_label = None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        assigned_gt_inds[cls_label.reshape(-1)<-2] = 0

        for j in range(self.topk):
            for i in range(num_gts):
                inj = self.topk-1-j
                assigned_gt_inds[gt_argmax_overlaps[i,inj]] = i + 1

        # for i in range(num_gts):
        #     for j in range(self.topk):
        #         max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
        #         assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
