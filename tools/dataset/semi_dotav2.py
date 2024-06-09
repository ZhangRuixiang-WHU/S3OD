# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
"""Generate labeled and unlabeled dataset for coco train.

Example:
python tools/coco_semi.py
"""

import argparse
import numpy as np
import json
import os
import shutil

def prepare_coco_data(seed=1, percent=10.0, version=2017, seed_offset=0):

    np.random.seed(seed + seed_offset)
    
    image_dir = '/home/zrx/ssod/SoftTeacher/data/dota2/train_obb/split_images/images'
    label_dir = '/home/zrx/ssod/SoftTeacher/data/dota2/train_obb/split_images/annfiles'
    labeled_save_path = '/home/zrx/ssod/SoftTeacher/data/dota2/coco/semi_train/semi-{}@{}'.format(str(int(seed)),str(int(percent)))
    unlabeled_save_path = '/home/zrx/ssod/SoftTeacher/data/dota2/coco/semi_train/semi-{}@{}-unlabeled'.format(str(int(seed)),str(int(percent)))
    os.makedirs(labeled_save_path, exist_ok=True)
    os.makedirs(unlabeled_save_path, exist_ok=True)
    # labeled_image_save_path = os.path.join(labeled_save_path,'images')
    labeled_label_save_path = os.path.join(labeled_save_path,'labels')
    unlabeled_label_save_path = os.path.join(unlabeled_save_path,'labels')
    os.makedirs(labeled_label_save_path, exist_ok=True)
    os.makedirs(unlabeled_label_save_path, exist_ok=True)

    image_list = os.listdir(image_dir)
    labeled_tot = int(percent / 100.0 * len(image_list))
    labeled_ind = np.random.choice(
        range(len(image_list)), size=labeled_tot, replace=False
    )
    labeled_images = []
    unlabeled_images = []
    labeled_ind = set(labeled_ind)
    for i in range(len(image_list)):
        if i in labeled_ind:
            labeled_images.append(image_list[i].split('.png')[0])
        else:
            unlabeled_images.append(image_list[i].split('.png')[0])
    
    for idx in labeled_images:
        src_label = os.path.join(label_dir,'{}.txt'.format(idx))
        shutil.copy(src_label,labeled_label_save_path)
    print('copy {} samples for labeled of {} percent {} fold'.format(str(len(labeled_images)),str(int(percent)),str(int(seed))))
    for idx in unlabeled_images:
        src_label = os.path.join(label_dir,'{}.txt'.format(idx))
        shutil.copy(src_label,unlabeled_label_save_path)
    print('copy {} samples for unlabeled of {} percent {} fold'.format(str(len(unlabeled_images)),str(int(percent)),str(int(seed))))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data-dir", type=str)
    parser.add_argument("--percent", type=float, default=10)
    parser.add_argument("--version", type=int, default=2017)
    parser.add_argument("--seed", type=int, help="seed", default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    print(args)
    # DATA_DIR = args.data_dir
    prepare_coco_data(args.seed, args.percent, args.version, args.seed_offset)
