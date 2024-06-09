import os
import shutil
from tqdm import trange

ref_dir = '/home/zrx/ssod/SoftTeacher/data/soda/annfiles/test'

src_dir = '/home/zrx/ssod/SoftTeacher/data/soda/Images/Images'

save_dir = '/home/zrx/ssod/SoftTeacher/data/soda/images/test'
os.makedirs(save_dir, exist_ok=True)

ref_list = os.listdir(ref_dir)

for idx in trange(len(ref_list)):
    ref_file = ref_list[idx]
    name = ref_file.split('.txt')[0]
    img_name = name +'.jpg'
    src_path = os.path.join(src_dir,img_name)
    shutil.copy(src_path,save_dir)