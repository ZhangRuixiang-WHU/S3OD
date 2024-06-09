import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
# from mmdet.models import DETECTORS, build_detector
# from mmdet.core import multi_apply
from mmrotate.core import rbbox2roi,obb2xyxy
from mmrotate import ROTATED_DETECTORS, build_detector

from s3od.utils.structure_utils import dict_split, weighted_loss
from s3od.utils import log_image_with_boxes, log_every_n,get_iter_num

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid,filter_invalid_scr2, filter_invalid_with_adaptive_thr_size2,get_adaptive_size2


import numpy as np
import math


@ROTATED_DETECTORS.register_module()
class S3OD(MultiSteamDetector):
    def __init__(self, teacher_model: dict, student_model: dict, train_cfg=None, test_cfg=None):
        super(S3OD, self).__init__(
            dict(teacher=build_detector(teacher_model), student=build_detector(student_model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.current_iter = 0
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight
        # self.size_thr = {}

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        self.current_iter += 1
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unsup_student" in data_groups and self.current_iter > 20000:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_rbbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]

        pseudo_bboxes_pre = self._transform_rbbox(
            teacher_info["det_bboxes_pre"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels_pre = teacher_info["det_labels_pre"]
        
        gt_neg_bboxes, get_neg_scores, _, _ = multi_apply(
            filter_invalid_scr2,
            [bbox[:, :5] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5] for bbox in pseudo_bboxes],
            thr=-self.train_cfg.cls_neg_gt_threshold,
        )
        ######### ---get the size-aware threshold---############
        if self.train_cfg.with_SAT:
            # print('**************  with SAT  ****************')
            if self.train_cfg.with_load:
                iter_num = get_iter_num() + 30000
            else:
                iter_num = get_iter_num()
            # cls_thr_percent = round(80 * math.exp(-iter_num/10000)+self.train_cfg.cls_thr_percent)
            # cls_thr_percent = np.clip(cls_thr_percent,30,95)
            cls_thr_percent = self.train_cfg.cls_thr_percent
            cls_thr = get_adaptive_size2(pseudo_bboxes_pre,pseudo_labels_pre,vaild_len=self.train_cfg.cls_thr_vaild_len,percent=cls_thr_percent,min=self.train_cfg.cls_thr_min,max=self.train_cfg.cls_thr_max,small_size=self.train_cfg.small_size)
            pseudo_bboxes, pseudo_labels = multi_apply(
                filter_invalid_with_adaptive_thr_size2,
                [bbox[:, :5] for bbox in pseudo_bboxes],
                pseudo_labels,
                [bbox[:, 5] for bbox in pseudo_bboxes],
                [cls_thr]*len(pseudo_labels),
            )
        ######### ---get the size-aware threshold---############
        loss = {}
        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        loss.update(
            self.unsup_rcnn_cls_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                gt_neg_bboxes,
                get_neg_scores,
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
            )
        )
        return loss

    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student.with_rpn:
            if not self.train_cfg.with_SAT:
                gt_bboxes = []
                for bbox in pseudo_bboxes:
                    bbox, _, _ = filter_invalid(
                        bbox[:, :5],
                        score=bbox[
                            :, 5
                        ],  # TODO: replace with foreground score, here is classification score,
                        thr=self.train_cfg.rpn_pseudo_threshold,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    gt_bboxes.append(bbox)
            else:
                gt_bboxes = pseudo_bboxes
            
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore,with_clsassign = self.train_cfg.with_SRA
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            # log_image_with_boxes(
            #     "rpn",
            #     student_info["img"][0],
            #     pseudo_bboxes[0][:, :5],
            #     bbox_tag="rpn_pseudo_label",
            #     scores=pseudo_bboxes[0][:, 5],
            #     interval=50,
            #     img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            # )
            return losses, proposal_list
        else:
            return {}, None

    def unsup_rcnn_cls_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        gt_neg_bboxes,
        get_neg_scores,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        if not self.train_cfg.with_SAT:
            gt_bboxes, gt_labels, _ = multi_apply(
                filter_invalid,
                [bbox[:, :5] for bbox in pseudo_bboxes],
                pseudo_labels,
                [bbox[:, 5] for bbox in pseudo_bboxes],
                thr=self.train_cfg.cls_pseudo_threshold,
            )
        else:
            gt_bboxes, gt_labels = pseudo_bboxes,pseudo_labels
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        
        gt_neg_bboxes = [obb2xyxy(pro[:,0:5], 'le90') for pro in gt_neg_bboxes]
        
        if self.train_cfg.with_FNS:
            aligned_proposals = self._transform_bbox(
                proposal_list,
                M,
                [meta["img_shape"] for meta in teacher_img_metas],
            )
            with torch.no_grad():
                _, _scores = self.teacher.roi_head.simple_test_bboxes(
                    teacher_feat,
                    teacher_img_metas,
                    aligned_proposals,
                    None,
                    rescale=False,
                )
                # bg_score = torch.cat([_score[:, -1] for _score in _scores])
                neg_scores = []
                for _score,gt_l,gt_neg in zip(_scores,gt_labels,get_neg_scores):
                    gt_weight = torch.ones_like(gt_l)
                    # proposals_weight = 1.3 - (_score[:, -1]*_score[:, -1])
                    # proposals_weight = torch.clip(proposals_weight,0.5,1.0).reshape(-1).to(gt_weight)
                    proposals_weight = torch.ones([_score.shape[0],]).to(gt_weight)
                    # proposals_weight = _score[:, -1].reshape(-1)
                    gt_neg_scr = gt_neg.reshape(-1)*2
                    bg_s = torch.cat([gt_weight,proposals_weight,gt_neg_scr])
                    neg_scores.append(bg_s)

            sampling_results = self.get_sampling_result_FNS(
                img_metas,
                proposal_list,
                gt_bboxes,
                gt_labels,
                _scores,
                gt_neg_bboxes
            )
            label_weight_list=[]
            for neg_srcore, sample_res in zip(neg_scores,sampling_results):
                pos_weight = torch.ones([len(sample_res.pos_inds),]).to(neg_srcore)
                neg_weight = neg_srcore[sample_res.neg_inds]
                # if neg_weight.sum().item() >0:
                #     ratio = (len(sample_res.neg_inds)/neg_weight.sum().item())
                #     print(ratio)
                #     neg_weight = ratio*neg_weight
                label_weight = torch.cat([pos_weight,neg_weight])
                label_weight_list.append(label_weight)
            label_weight_batch = torch.cat([weight for weight in label_weight_list])
        else:
            sampling_results = self.get_sampling_result(
                img_metas,
                proposal_list,
                gt_bboxes,
                gt_labels
            )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        if self.train_cfg.with_FNS:
            bbox_targets[1][:] = label_weight_batch

        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        # if len(gt_bboxes[0]) > 0:
        #     log_image_with_boxes(
        #         "rcnn_cls",
        #         student_info["img"][0],
        #         gt_bboxes[0],
        #         bbox_tag="pseudo_label",
        #         labels=gt_labels[0],
        #         class_names=self.CLASSES,
        #         interval=50,
        #         img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
        #     )
        return loss

    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            gt_hbboxes = obb2xyxy(gt_bboxes[i], 'le90')
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_hbboxes, gt_bboxes_ignore[i], gt_labels[i],
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_hbboxes,
                gt_labels[i],
            )
            if gt_bboxes[i].numel() == 0:
                sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                    (0, gt_bboxes[0].size(-1))).zero_()
            else:
                sampling_result.pos_gt_bboxes = \
                    gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

            sampling_results.append(sampling_result)
        return sampling_results

    def get_sampling_result_FNS(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        scores_list,
        gt_neg_bboxes,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            gt_hbboxes = obb2xyxy(gt_bboxes[i], 'le90')
            gt_neg_box = gt_neg_bboxes[i]
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_hbboxes, gt_bboxes_ignore[i], gt_labels[i], scores_list[i][:,-1],
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_hbboxes,
                gt_labels[i],
                gt_neg_box,
            )
            if gt_bboxes[i].numel() == 0:
                sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                    (0, gt_bboxes[0].size(-1))).zero_()
            else:
                sampling_result.pos_gt_bboxes = \
                    gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

            sampling_results.append(sampling_result)
        return sampling_results


    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes
    
    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_rbbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_rbboxes(bboxes, trans_mat, max_shape)
        return bboxes
    
    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        )

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 6) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )


        thrs = []
        for i, proposals in enumerate(proposal_list):
            dynamic_ratio = 1.0
            scores = proposals[:, 5].clone()
            scores = scores.sort(descending=True)[0]
            if len(scores) == 0:
                thrs.append(1)  # no kept pseudo boxes
            else:
                # num_gt = int(scores.sum() + 0.5)
                num_gt = int(scores.sum() * dynamic_ratio + 0.5)
                num_gt = min(num_gt, len(scores) - 1)
                thrs.append(scores[num_gt] - 1e-5)
        proposal_list_pre, proposal_label_list_pre, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label,thr in zip(
                        proposal_list, proposal_label_list, thrs
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        
        teacher_info["det_bboxes_pre"] = proposal_list_pre
        teacher_info["det_labels_pre"] = proposal_label_list_pre

        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
