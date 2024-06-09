# ! /usr/bin/python
# -*- coding:UTF-8 -*-
import cv2
import os 
import numpy as np
from tqdm import trange
import xml.etree.ElementTree as ET


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

def txt2roxml(txt_path, save_xml_path):

    # cls_map = {
    #     '驱逐舰': 'destroyer', 
    #     '两栖舰': 'amphibious', 
    #     '巡洋舰': 'cruiser', 
    #     '其他': 'ship', 
    #     '潜艇': 'submarine', 
    #     '护卫舰': 'frigate', 
    #     '航空母舰': 'aircraft-carrier',
    # }

    cls_map = {'Ship': 'ship', 
               'Cruiser': 'cruiser', 
               'Aircraft-carrier': 'aircraft-carrier',
               'Destroyer': 'destroyer', 
               'Frigate': 'frigate', 
               'Warship': 'ship', 
               'Cargo-vessel': 'ship', 
               'Tugboat': 'ship', 
               'Submarine': 'submarine', 
               'Loose-pulley': 'ship', 
               'Motorboat': 'ship', 
               'Engineering-ship': 'ship', 
               'Amphibious-ship': 'amphibious', 
               'Command-ship': 'ship', 
               'Fishing-boat': 'ship', 
               'Hovercraft': 'ship', }

    txt_file_list = os.listdir(txt_path)

    for idx in trange(len(txt_file_list)):
        txt_file = txt_file_list[idx]
        filename = txt_file.split('.xml')[0]
        txt_file_path = os.path.join(txt_path, txt_file)

        xml_file_path = os.path.join(save_xml_path, '{}.xml'.format(filename))
        xml_file = open(xml_file_path, 'w', encoding='utf-8')
        xml_file.write('<annotation>\n')
        xml_file.write('    <folder>VOC2007</folder>\n')
        xml_file.write('    <filename>' + (filename) + '.png' + '</filename>\n')
        xml_file.write('    <size>\n')
        xml_file.write('        <width>' + '1024' + '</width>\n')
        xml_file.write('        <height>' + '1024' + '</height>\n')
        xml_file.write('        <depth>3</depth>\n')
        xml_file.write('    </size>\n')


        tree = ET.parse(txt_file_path)
        root = tree.getroot()

        for object in root.findall('object'):
            class_id = (object.find('name').text)
            if class_id in cls_map:
                class_name = cls_map[class_id]
            else:
                class_name = class_id
            cx = (object.find('robndbox').find('cx').text)
            cy = (object.find('robndbox').find('cy').text)
            w = (object.find('robndbox').find('w').text)
            h = (object.find('robndbox').find('h').text)
            angle = (object.find('robndbox').find('angle').text)
            xml_file.write('    <object>\n')
            xml_file.write('        <name>' + 'ship' + '</name>\n')
            # xml_file.write('        <name>' + class_name + '</name>\n')
            xml_file.write('        <type>robndbox</type>\n')
            xml_file.write('        <pose>Unspecified</pose>\n')
            xml_file.write('        <truncated>0</truncated>\n')
            xml_file.write('        <difficult>0</difficult>\n')
            xml_file.write('        <robndbox>\n')
            xml_file.write('            <cx>' + str(cx) + '</cx>\n')
            xml_file.write('            <cy>' + str(cy) + '</cy>\n')
            xml_file.write('            <w>'  + str(w)  + '</w>\n' )
            xml_file.write('            <h>'  + str(h)  + '</h>\n' )
            xml_file.write('            <angle>' + str(angle) + '</angle>\n')
            xml_file.write('        </robndbox>\n')
            xml_file.write('    </object>\n')
        xml_file.write('</annotation>')



if __name__ == "__main__":

    ##########################----txt transfer to xml(voc)----##########################
    src_path = '/data/zrx/Rotated/SSDA_ship/target_val/labels/'
    save_xml_path = '/data/zrx/Rotated/SSDA_ship/target_val/labels_ship/'
    os.makedirs(save_xml_path,exist_ok=True)
    txt2roxml(src_path, save_xml_path)