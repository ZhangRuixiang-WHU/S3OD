# import os
# # import cv2
# # import numpy as np
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
import cv2
import numpy as np
from tqdm import trange
import xml.etree.ElementTree as ET
# import shutil
def pointobb2thetaobb(pointobb):
    """convert pointobb to thetaobb
    Input:
        pointobb (list[1x8]): [x1, y1, x2, y2, x3, y3, x4, y4]
    Output:
        thetaobb (list[1x5])
    """
    pointobb = np.int0(np.array(pointobb))
    pointobb.resize(4, 2)
    rect = cv2.minAreaRect(pointobb)
    x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
    theta = theta / 180.0 * np.pi
    thetaobb = [x, y, w, h, theta]
    
    return thetaobb



src_label_dir = '/home/zrx/ssod/SoftTeacher/data/soda/train_obb/annfiles'#labeled_target_obb


label_list = os.listdir(src_label_dir)

num = 0
small_num = 0

img_num = 0

# label_info = {}

for idx in trange(len(label_list)):
    img_num += 1
    label_file_name = label_list[idx]
    label_file_path = os.path.join(src_label_dir, label_file_name)


    tree=open(label_file_path)
    root=tree.readlines()
    for single_object in root:
        num = num + 1
        x1,y1,x2,y2,x3,y3,x4,y4,classname,_ = single_object.split(' ')
        pbb = [float(x1),float(y1),float(x2),float(y2),float(x3),float(y3),float(x4),float(y4)]
        rbb = pointobb2thetaobb(pbb)
        size = rbb[2]*rbb[3]
        if size <= 1024:
            small_num +=1
        # if classname in label_info.keys():
        #     label_info[classname] += 1
        # else:
        #     label_info[classname] = 1
    

print('img_num:', img_num)
print('instance_num:', num)
print('small_ins_num:', small_num)
print('small proportion:', small_num/num *100)
# print('label_info:', label_info)
# print('num of cls:', len(label_info))