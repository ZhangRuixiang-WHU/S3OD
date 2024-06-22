# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_SAMPLERS
from .sampling_result import SamplingResult
from .base_sampler import BaseSampler


@BBOX_SAMPLERS.register_module()
class RandomSamplerPseudo(BaseSampler):
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_ub (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        from mmdet.core.bbox import demodata
        super(RandomSamplerPseudo, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))
        # self.hard_precent =use_hard_neg_precent

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        # This is a temporary fix. We can revert the following code
        # when PyTorch fixes the abnormal return of torch.randperm.
        # See: https://github.com/open-mmlab/mmdetection/pull/5014
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
    
    def _sample_neg_hard(self, assign_result, num_expected, hard_precent=0.5,**kwargs):
        """Randomly sample some negative samples."""
        neg_inds_hard = torch.nonzero(assign_result.gt_inds == -2, as_tuple=False)
        if neg_inds_hard.numel() != 0:
            neg_inds_hard = neg_inds_hard.squeeze(1)
        num_hard_expected = int(num_expected * hard_precent)
        if len(neg_inds_hard) <= num_hard_expected:
            neg_inds_hard_sample = neg_inds_hard
        else:
            neg_inds_hard_sample = self.random_choice(neg_inds_hard, num_hard_expected)

        neg_inds_simple = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds_simple.numel() != 0:
            neg_inds_simple = neg_inds_simple.squeeze(1)
        num_simple_expected = num_expected - len(neg_inds_hard_sample)
        if len(neg_inds_simple) <= num_simple_expected:
            neg_inds_simple_sample = neg_inds_simple
        else:
            neg_inds_simple_sample = self.random_choice(neg_inds_simple, num_simple_expected)
        
        if neg_inds_hard_sample.numel() != 0:
            neg_inds = torch.cat([neg_inds_hard_sample,neg_inds_simple_sample])
        else:
            neg_inds = neg_inds_simple_sample
        # try:
        #     assert len(neg_inds) == num_expected
        # except:
        #     print('################################################')
        #     print('neg_inds_hard_sample:',neg_inds_hard_sample)
        #     print('neg_inds_hard_sample.shape:',neg_inds_hard_sample.shape)
        #     print('num_simple_expected:',neg_inds_simple_sample)
        #     print('num_simple_expected.shape:',neg_inds_simple_sample.shape)
        #     print('neg_inds.len:',len(neg_inds))
        #     print('num_expected:',num_expected)
        #     print('################################################')
            
        return neg_inds

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               use_hard = False,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if use_hard:
            if len(bboxes.shape) < 2:
                bboxes = bboxes[None, :]

            bboxes = bboxes[:, :4]

            gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
            if self.add_gt_as_proposals and len(gt_bboxes) > 0:
                if gt_labels is None:
                    raise ValueError(
                        'gt_labels must be given when add_gt_as_proposals is True')
                bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
                assign_result.add_gt_(gt_labels)
                gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
                gt_flags = torch.cat([gt_ones, gt_flags])

            num_expected_pos = int(self.num * self.pos_fraction)
            pos_inds = self.pos_sampler._sample_pos(
                assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
            # We found that sampled indices have duplicated items occasionally.
            # (may be a bug of PyTorch)
            pos_inds = pos_inds.unique()
            num_sampled_pos = pos_inds.numel()
            num_expected_neg = self.num - num_sampled_pos
            if self.neg_pos_ub >= 0:
                _pos = max(1, num_sampled_pos)
                neg_upper_bound = int(self.neg_pos_ub * _pos)
                if num_expected_neg > neg_upper_bound:
                    num_expected_neg = neg_upper_bound
            neg_inds = self.neg_sampler._sample_neg_hard(
                assign_result, num_expected_neg, hard_precent=0.1, bboxes=bboxes, **kwargs)
            neg_inds = neg_inds.unique()

            sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                            assign_result, gt_flags)
        else:
            if len(bboxes.shape) < 2:
                bboxes = bboxes[None, :]

            bboxes = bboxes[:, :4]

            gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
            if self.add_gt_as_proposals and len(gt_bboxes) > 0:
                if gt_labels is None:
                    raise ValueError(
                        'gt_labels must be given when add_gt_as_proposals is True')
                bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
                assign_result.add_gt_(gt_labels)
                gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
                gt_flags = torch.cat([gt_ones, gt_flags])

            num_expected_pos = int(self.num * self.pos_fraction)
            pos_inds = self.pos_sampler._sample_pos(
                assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
            # We found that sampled indices have duplicated items occasionally.
            # (may be a bug of PyTorch)
            pos_inds = pos_inds.unique()
            num_sampled_pos = pos_inds.numel()
            num_expected_neg = self.num - num_sampled_pos
            if self.neg_pos_ub >= 0:
                _pos = max(1, num_sampled_pos)
                neg_upper_bound = int(self.neg_pos_ub * _pos)
                if num_expected_neg > neg_upper_bound:
                    num_expected_neg = neg_upper_bound
            neg_inds = self.neg_sampler._sample_neg(
                assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
            neg_inds = neg_inds.unique()

            sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                            assign_result, gt_flags)
        return sampling_result
