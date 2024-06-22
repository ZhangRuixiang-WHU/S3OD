# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_detector_by_patches, init_detector, inference_detector
from .train import train_detector

__all__ = ['inference_detector_by_patches', 'train_detector','init_detector','inference_detector']
