import os
import cv2
import numpy as np
# from tqdm import trange
# import xml.etree.ElementTree as ET
# # import shutil


# src_label_dir = '/data/zrx/Rotated/DOTA-v1.5/coco/semi_train/semi-2@1/labels_hbbxml/'


# label_list = os.listdir(src_label_dir)

# num = 0

# img_num = 0

# label_info = {}

# for idx in trange(len(label_list)):
#     img_num += 1
#     label_file_name = label_list[idx]
#     label_file_path = os.path.join(src_label_dir, label_file_name)


#     tree=ET.parse(label_file_path)
#     root=tree.getroot()
#     for single_object in root.findall('object'):
#         num = num + 1
#         classname = single_object.find('name').text
#         if classname in label_info.keys():
#             label_info[classname] += 1
#         else:
#             label_info[classname] = 1
    

# print('img_num:', img_num)
# print('ship_num:', num)
# print('label_info:', label_info)



import os
# import cv2
# import numpy as np
from tqdm import trange
import xml.etree.ElementTree as ET
# import shutil


src_label_dir = '/data/zrx/Rotated/SSDA_dotav2/val_obb/split_images/annfiles'#labeled_target_obb


label_list = os.listdir(src_label_dir)

num = 0

img_num = 0

label_info = {}

for idx in trange(len(label_list)):
    img_num += 1
    label_file_name = label_list[idx]
    label_file_path = os.path.join(src_label_dir, label_file_name)


    tree=open(label_file_path)
    root=tree.readlines()
    for single_object in root:
        num = num + 1
        classname = single_object.split(' ')[8]
        if classname in label_info.keys():
            label_info[classname] += 1
        else:
            label_info[classname] = 1
    

print('img_num:', img_num)
print('ship_num:', num)
print('label_info:', label_info)
print('num of cls:', len(label_info))