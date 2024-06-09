# from mmdet.datasets import DATASETS, ConcatDataset, build_dataset
from mmdet.datasets import ConcatDataset
from mmrotate.datasets import ROTATED_DATASETS, build_dataset


@ROTATED_DATASETS.register_module()
class SemiDataset(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup: dict, unsup: dict, **kwargs):
        super().__init__([build_dataset(sup), build_dataset(unsup)], **kwargs)

    @property
    def sup(self):
        return self.datasets[0]

    @property
    def unsup(self):
        return self.datasets[1]
