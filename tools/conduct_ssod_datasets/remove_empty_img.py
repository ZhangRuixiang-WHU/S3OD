import os
import shutil
from tqdm import trange

root_dir = '/home/zrx/ssod/SoftTeacher/data/soda/val_obb/'

annfiles_dir = root_dir + 'annfiles'

images_dir = root_dir + 'images'

ref_list = os.listdir(annfiles_dir)

for idx in trange(len(ref_list)):
    ref_file = ref_list[idx]
    name = ref_file.split('.txt')[0]
    img_name = name +'.jpg'
    txt_name = name +'.txt'
    
    img_path = os.path.join(images_dir,img_name)
    txt_path = os.path.join(annfiles_dir,txt_name)
    file = open(txt_path)
    data = file.readlines()
    if len(data) == 0:
        file.close()
        os.remove(img_path)
        os.remove(txt_path)

    # print('jijg')