import os
import shutil
from tqdm import trange

image_path = '/data/zrx/Rotated/SSDA_dotav2_set2/train_obb/split_images/images'
label_path = '/data/zrx/Rotated/SSDA_dotav2_set2/train_obb/split_images/annfiles'

image_file_list = os.listdir(image_path)
label_file_list = os.listdir(label_path)

assert len(image_file_list) == len(label_file_list)

cnt = 0
for idx in trange(len(image_file_list)):
    image_file = image_file_list[idx]
    label_file = image_file.split('.png')[0] + '.txt'
    file_name = image_file.split('__')[0]
    image_file_path = os.path.join(image_path,image_file)
    label_file_path = os.path.join(label_path,label_file)
    ann=open(label_file_path)
    a = len(ann.readlines())
    if a == 0:
        os.remove(image_file_path)
        os.remove(label_file_path)
        cnt +=1
    else:
        pass
print('remove {} empty files in train set'.format(str(cnt)))

image_path = '/data/zrx/Rotated/SSDA_dotav2_set2/val_obb/split_images/images'
label_path = '/data/zrx/Rotated/SSDA_dotav2_set2/val_obb/split_images/annfiles'

image_file_list = os.listdir(image_path)
label_file_list = os.listdir(label_path)

assert len(image_file_list) == len(label_file_list)

cnt = 0

for idx in trange(len(image_file_list)):
    image_file = image_file_list[idx]
    label_file = image_file.split('.png')[0] + '.txt'
    file_name = image_file.split('__')[0]
    image_file_path = os.path.join(image_path,image_file)
    label_file_path = os.path.join(label_path,label_file)
    ann=open(label_file_path)
    a = len(ann.readlines())
    if a == 0:
        os.remove(image_file_path)
        os.remove(label_file_path)
        cnt +=1
    else:
        pass
print('remove {} empty files in val set'.format(str(cnt)))

image_path = '/data/zrx/Rotated/SSDA_dotav2_set2/labeled_target_obb/split_images/images'
label_path = '/data/zrx/Rotated/SSDA_dotav2_set2/labeled_target_obb/split_images/annfiles'

image_file_list = os.listdir(image_path)
label_file_list = os.listdir(label_path)

assert len(image_file_list) == len(label_file_list)


cnt = 0
for idx in trange(len(image_file_list)):
    image_file = image_file_list[idx]
    label_file = image_file.split('.png')[0] + '.txt'
    file_name = image_file.split('__')[0]
    image_file_path = os.path.join(image_path,image_file)
    label_file_path = os.path.join(label_path,label_file)
    ann=open(label_file_path)
    a = len(ann.readlines())
    if a == 0:
        os.remove(image_file_path)
        os.remove(label_file_path)
        cnt +=1
    else:
        pass
print('remove {} empty files in labeled_target_obb'.format(str(cnt)))

# image_path = '/data/zrx/Rotated/SSDA_dotav2_set2/unlabeled_target_obb/split_images/images'
# label_path = '/data/zrx/Rotated/SSDA_dotav2_set2/unlabeled_target_obb/split_images/annfiles'

# image_file_list = os.listdir(image_path)
# label_file_list = os.listdir(label_path)

# assert len(image_file_list) == len(label_file_list)


# cnt = 0
# for idx in trange(len(image_file_list)):
#     image_file = image_file_list[idx]
#     label_file = image_file.split('.png')[0] + '.txt'
#     file_name = image_file.split('__')[0]
#     image_file_path = os.path.join(image_path,image_file)
#     label_file_path = os.path.join(label_path,label_file)
#     ann=open(label_file_path)
#     a = len(ann.readlines())
#     if a == 0:
#         os.remove(image_file_path)
#         os.remove(label_file_path)
#         cnt +=1
#     else:
#         pass
# print('remove {} empty files in unlabeled_target_obb'.format(str(cnt)))

