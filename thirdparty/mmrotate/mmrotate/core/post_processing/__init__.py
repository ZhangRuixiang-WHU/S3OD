# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_nms_rotated import (aug_multiclass_nms_rotated,
                               multiclass_nms_rotated,multiclass_nms_rotated_sigma)

__all__ = ['multiclass_nms_rotated','multiclass_nms_rotated_sigma', 'aug_multiclass_nms_rotated']
