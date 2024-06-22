# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DotaDataset(CocoDataset):

    CLASSES = ('large-vehicle', 'small-vehicle', 'plane', 'bridge', 'ship', 'harbor', 'swimming-pool', 'roundabout', 'soccer-ball-field', 
               'helicopter', 'storage-tank', 'tennis-court', 'baseball-diamond', 'basketball-court', 'ground-track-field','container-crane')
