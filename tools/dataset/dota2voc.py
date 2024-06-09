import os
from tqdm import trange
from lxml import etree as ET


def simple_xml_dump_hbb(objects, filename, label_save_file):

    # bboxes, labels= [], []
    # for obj in objects:
    #     bboxes.append(obj['bbox'])
    #     labels.append(obj['label'])

    root=ET.Element("annotations")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root,"size")
    ET.SubElement(size, "width").text = str(1024)
    ET.SubElement(size, "height").text = str(1024)
  
    for obj in objects:

        object=ET.SubElement(root, "object")

        #写入检测框的位置信息
        ET.SubElement(object,"name").text = obj['label']
        ET.SubElement(object,"type").text = 'bndbox'
        bndbox=ET.SubElement(object,"bndbox")
        xmin,ymin,xmax,ymax = obj['bbox']
        ET.SubElement(bndbox,"xmin").text = str(xmin)
        ET.SubElement(bndbox,"ymin").text = str(ymin)
        ET.SubElement(bndbox,"xmax").text = str(xmax)
        ET.SubElement(bndbox,"ymax").text = str(ymax)
        
    tree = ET.ElementTree(root)

    tree.write(label_save_file, pretty_print=True, xml_declaration=True, encoding='utf-8')





save_path = '/home/zrx/ssod/SoftTeacher/data/dota1.5/coco/semi_train/semi-2@1/labels_hbbxml' 
os.makedirs(save_path,exist_ok=True)

label_dir = '/home/zrx/ssod/SoftTeacher/data/dota1.5/coco/semi_train/semi-2@1/labels'
val = os.listdir(label_dir)

for i in trange(len(val)):
    name_txt = val[i]
# for i,name_txt in enumerate(val):
    file_name = name_txt.split('.txt')[0]
    file = open(os.path.join(label_dir,name_txt))
    data = file.readlines()
    l = []
    for j in range(len(data)):
        l.append(list(data[j].split()))
        # if j >1 :
        #     l.append(list(data[j].split()))
    label_save_file = os.path.join(save_path,'{}.xml'.format(file_name))
    objects = []
    for i, childdata in enumerate(l):
        cls_name = childdata[-2]
        pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y, pt4_x, pt4_y = list(map(float, childdata[:-2]))
        lt_x = float(min(pt1_x, pt2_x, pt3_x, pt4_x))
        lt_y = float(min(pt1_y, pt2_y, pt3_y, pt4_y))
        rb_x = float(max(pt1_x, pt2_x, pt3_x, pt4_x))
        rb_y = float(max(pt1_y, pt2_y, pt3_y, pt4_y))
        object_struct = {}

        xmin = lt_x
        ymin = lt_y
        xmax = rb_x
        ymax = rb_y
        object_struct['bbox'] = [xmin,ymin,xmax,ymax]
        object_struct['label'] = cls_name
        
        objects.append(object_struct)
    simple_xml_dump_hbb(objects, file_name, label_save_file)