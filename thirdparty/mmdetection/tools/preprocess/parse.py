import os
# import cv2
# import numpy as np
from tqdm import trange
import xml.etree.ElementTree as ET
# import shutil


src_label_dir = '/data/zrx/VisDrone/train/trainlabelr_hbb/'


label_list = os.listdir(src_label_dir)

num = 0

img_num = 0

label_info = {}

for idx in trange(len(label_list)):
    img_num += 1
    label_file_name = label_list[idx]
    label_file_path = os.path.join(src_label_dir, label_file_name)


    tree=ET.parse(label_file_path)
    root=tree.getroot()
    for single_object in root.findall('object'):
        num = num + 1
        classname = single_object.find('name').text
        if classname in label_info.keys():
            label_info[classname] += 1
        else:
            label_info[classname] = 1
    

print('img_num:', img_num)
print('ship_num:', num)
print('label_info:', label_info)