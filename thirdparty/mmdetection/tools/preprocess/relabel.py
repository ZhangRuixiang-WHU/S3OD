# ! /usr/bin/python
# -*- coding:UTF-8 -*-
# import cv2
import os 
import numpy as np
from tqdm import trange
import xml.etree.ElementTree as ET


def voc_parse(label_file):
    """parse rotation VOC style dataset label file
    
    Arguments:
        label_file {str} -- label file path
    
    Returns:
        dict, {'bbox': [cx, cy, w, h, theta (rad/s)], 'label': class_name} -- objects' location and class
    """
    tree = ET.parse(label_file)
    root = tree.getroot()
    objects = []
    for single_object in root.findall('object'):
        bndbox = single_object.find('bndbox')
        object_struct = {}

        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        object_struct['bbox'] = [xmin,ymin,xmax,ymax]
        object_struct['label'] = single_object.find('name').text
        
        objects.append(object_struct)
    return objects

def simple_xml_dump_hbb(objects, filename, label_save_file):

    # bboxes, labels= [], []
    # for obj in objects:
    #     bboxes.append(obj['bbox'])
    #     labels.append(obj['label'])

    root=ET.Element("annotations")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root,"size")
    ET.SubElement(size, "width").text = str(840)
    ET.SubElement(size, "height").text = str(712)
    ET.SubElement(size, "depth").text = str(3)
  
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

    tree.write(label_save_file, xml_declaration=True, encoding='utf-8') # pretty_print=True,


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

def txt2roxml(src_path, save_xml_path):

    # cls_map = {
    #     '船只':'ship',
    #     '桥梁':'bridge',
    #     '油罐':'oiltank',
    #     '飞机':'plane',
    #     'W':'ship'
    # }
    cls_map = {
        'car':'car',
        'bus':'bus',
        'van':'van',
        'truck':'truck',
        'feright_car':'feright_car',
        'feright car':'feright_car',
        '*': 'feright_car',
        'feright': 'feright_car',
        'truvk':'truck'
    }

    src_file_list = os.listdir(src_path)

    for idx in trange(len(src_file_list)):
        src_file = src_file_list[idx]
        filename = src_file.split('.xml')[0]
        src_file_path = os.path.join(src_path, src_file)

        objects = voc_parse(src_file_path)

        xml_file_path = os.path.join(save_xml_path, '{}.xml'.format(filename))

        for object in objects:
            class_id = object['label']
            if class_id in cls_map:
                object['label'] = cls_map[class_id]
            else:
                object['label'] = class_id
        simple_xml_dump_hbb(objects, filename, xml_file_path)


if __name__ == "__main__":

    ##########################----txt transfer to xml(voc)----##########################

    src_path = '/data/zrx/VisDrone/train/trainlabel_hbb_before/'
    save_xml_path = '/data/zrx/VisDrone/train/trainlabel_hbb/'
    os.makedirs(save_xml_path,exist_ok=True)
    txt2roxml(src_path, save_xml_path)