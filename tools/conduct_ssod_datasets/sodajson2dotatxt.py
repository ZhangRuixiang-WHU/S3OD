import os
from tqdm import trange
import json

json_dir = '/home/zrx/ssod/SoftTeacher/data/soda/org/Annotations/test'
save_txt_dir = '/home/zrx/ssod/SoftTeacher/data/soda/annfiles/test'
os.makedirs(save_txt_dir,exist_ok=True)

file_list = os.listdir(json_dir)

for idx in trange(len(file_list)):
    file_name = file_list[idx]
    name = file_name.split('.json')[0]

    json_path =  os.path.join(json_dir,file_name)
    save_txt_path = os.path.join(save_txt_dir,'{}.txt'.format(name))
    txt_file = open(save_txt_path, 'w')
    txt_file.write('imagesource:GoogleEarth\n')
    txt_file.write('gsd:0.115726939386\n')
    with open(json_path, 'r') as f:
        s1 = json.load(f)
    categories_list = s1['categories']
    for id, cate in enumerate(categories_list):
        assert id == cate['id']
    name_img = s1['images']['file_name'].split('.jpg')[0]
    assert name == name_img
    annos = s1['annotations']
    for ind, ann in enumerate(annos):
        poly = ann['poly']
        cate_id = ann['category_id']
        cls_name = categories_list[cate_id]['name']
        if  cls_name == 'ignore':
            pass
        else:
            str_rbbox = str(poly[0]) + ' ' + str(poly[1]) + ' ' + str(poly[2]) + ' ' + str(poly[3]) + ' ' + str(poly[4]) + ' ' + str(poly[5]) + ' ' + str(poly[6]) + ' ' + str(poly[7])
            txt_file.write(str_rbbox + ' ' + cls_name + ' ' + '0' +'\n')


    # txt_file.write('<?xml version=\'1.0\' encoding=\'utf-8\'?>\n')
    # print("file")