# import os
# import shutil
# from tqdm import trange

# ref_txt = open('/data/zrx/Rotated/DOTA-v2/DOTA_V2_train_GoogleEarth.txt', encoding='utf-8')
# ref_data = ref_txt.readlines()
# ref_name = []
# for d in ref_data:
#     ref_name.append(d.split('\n')[0])

# DA_dotav2_dir = '/data/zrx/Rotated/SSDA_dotav2_set2'
# os.makedirs(DA_dotav2_dir,exist_ok=True)
# DA_dotav2_train_dir_txt = os.path.join(DA_dotav2_dir,'train_obb/split_images/annfiles')
# os.makedirs(DA_dotav2_train_dir_txt,exist_ok=True)
# DA_dotav2_train_dir_img = os.path.join(DA_dotav2_dir,'train_obb/split_images/images')
# os.makedirs(DA_dotav2_train_dir_img,exist_ok=True)
# DA_dotav2_val_dir_txt = os.path.join(DA_dotav2_dir,'val_obb/split_images/annfiles')
# os.makedirs(DA_dotav2_val_dir_txt,exist_ok=True)
# DA_dotav2_val_dir_img = os.path.join(DA_dotav2_dir,'val_obb/split_images/images')
# os.makedirs(DA_dotav2_val_dir_img,exist_ok=True)

# train_image_path = '/data/zrx/Rotated/DOTA-v2/train_obb/split_images/images'
# train_label_path = '/data/zrx/Rotated/DOTA-v2/train_obb/split_images/annfiles'

# image_file_list = os.listdir(train_image_path)
# label_file_list = os.listdir(train_label_path)

# assert len(image_file_list) == len(label_file_list)

# print(ref_name)

# for idx in trange(len(image_file_list)):
#     image_file = image_file_list[idx]
#     label_file = image_file.split('.png')[0] + '.txt'
#     file_name = image_file.split('__')[0]
#     image_file_path = os.path.join(train_image_path,image_file)
#     label_file_path = os.path.join(train_label_path,label_file)
#     if file_name in ref_name:
#         shutil.copy(image_file_path,DA_dotav2_train_dir_img)
#         shutil.copy(label_file_path,DA_dotav2_train_dir_txt)
#     else:
#         shutil.copy(image_file_path,DA_dotav2_val_dir_img)
#         shutil.copy(label_file_path,DA_dotav2_val_dir_txt)



# ################### mv val_file ###########################
# import os
# import shutil
# from tqdm import trange

# ref_txt = open('/data/zrx/Rotated/DOTA-v2/DOTA_V2_val_GoogleEarth.txt', encoding='utf-8')
# ref_data = ref_txt.readlines()
# ref_name = []
# for d in ref_data:
#     ref_name.append(d.split('\n')[0])

# DA_dotav2_dir = '/data/zrx/Rotated/SSDA_dotav2_set2'
# os.makedirs(DA_dotav2_dir,exist_ok=True)
# DA_dotav2_train_dir_txt = os.path.join(DA_dotav2_dir,'train_obb/split_images/annfiles')
# os.makedirs(DA_dotav2_train_dir_txt,exist_ok=True)
# DA_dotav2_train_dir_img = os.path.join(DA_dotav2_dir,'train_obb/split_images/images')
# os.makedirs(DA_dotav2_train_dir_img,exist_ok=True)
# DA_dotav2_val_dir_txt = os.path.join(DA_dotav2_dir,'labeled_target_obb/split_images/annfiles')
# os.makedirs(DA_dotav2_val_dir_txt,exist_ok=True)
# DA_dotav2_val_dir_img = os.path.join(DA_dotav2_dir,'labeled_target_obb/split_images/images')
# os.makedirs(DA_dotav2_val_dir_img,exist_ok=True)

# train_image_path = '/data/zrx/Rotated/DOTA-v2/val_obb/split_images/images'
# train_label_path = '/data/zrx/Rotated/DOTA-v2/val_obb/split_images/annfiles'

# image_file_list = os.listdir(train_image_path)
# label_file_list = os.listdir(train_label_path)

# assert len(image_file_list) == len(label_file_list)

# print(ref_name)

# for idx in trange(len(image_file_list)):
#     image_file = image_file_list[idx]
#     label_file = image_file.split('.png')[0] + '.txt'
#     file_name = image_file.split('__')[0]
#     image_file_path = os.path.join(train_image_path,image_file)
#     label_file_path = os.path.join(train_label_path,label_file)
#     if file_name in ref_name:
#         shutil.copy(image_file_path,DA_dotav2_train_dir_img)
#         shutil.copy(label_file_path,DA_dotav2_train_dir_txt)
#     else:
#         shutil.copy(image_file_path,DA_dotav2_val_dir_img)
#         shutil.copy(label_file_path,DA_dotav2_val_dir_txt)



################### mv test_file ###########################
import os
import shutil
from tqdm import trange

ref_txt = open('/data/zrx/Rotated/DOTA-v2/DOTA_V2_test_GoogleEarth.txt', encoding='utf-8')
ref_data = ref_txt.readlines()
ref_name = []
for d in ref_data:
    ref_name.append(d.split('\n')[0])

DA_dotav2_dir = '/data/zrx/Rotated/SSDA_dotav2_set2'
os.makedirs(DA_dotav2_dir,exist_ok=True)
# DA_dotav2_train_dir_txt = os.path.join(DA_dotav2_dir,'train_obb/split_images/annfiles')
# os.makedirs(DA_dotav2_train_dir_txt,exist_ok=True)
# DA_dotav2_train_dir_img = os.path.join(DA_dotav2_dir,'train_obb/split_images/images')
# os.makedirs(DA_dotav2_train_dir_img,exist_ok=True)
DA_dotav2_val_dir_txt = os.path.join(DA_dotav2_dir,'unlabeled_target_obb/split_images/annfiles')
os.makedirs(DA_dotav2_val_dir_txt,exist_ok=True)
DA_dotav2_val_dir_img = os.path.join(DA_dotav2_dir,'unlabeled_target_obb/split_images/images')
os.makedirs(DA_dotav2_val_dir_img,exist_ok=True)

train_image_path = '/data/zrx/Rotated/DOTA-v2/test_obb/split_images/images'
train_label_path = '/data/zrx/Rotated/DOTA-v2/test_obb/split_images/annfiles'

image_file_list = os.listdir(train_image_path)
label_file_list = os.listdir(train_label_path)

assert len(image_file_list) == len(label_file_list)

print(ref_name)

for idx in trange(len(image_file_list)):
    image_file = image_file_list[idx]
    label_file = image_file.split('.png')[0] + '.txt'
    file_name = image_file.split('__')[0]
    image_file_path = os.path.join(train_image_path,image_file)
    label_file_path = os.path.join(train_label_path,label_file)
    if file_name in ref_name:
        pass
    else:
        shutil.copy(image_file_path,DA_dotav2_val_dir_img)
        shutil.copy(label_file_path,DA_dotav2_val_dir_txt)
        # shutil.copy(image_file_path,DA_dotav2_train_dir_img)
        # shutil.copy(label_file_path,DA_dotav2_train_dir_txt)

