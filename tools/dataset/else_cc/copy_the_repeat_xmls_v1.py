import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
import argparse

    
def simple_obb_xml_dump(objects, save_xml):
    bboxes, pointobbs, labels, scores, rbbox, num = [], [], [], [], [], 0
    for obj in objects:
        bboxes.append(obj['bbox'])
        pointobbs.append(obj['pointobb'])
        labels.append(obj['label'])
        # scores.append(obj['score'])
        rbbox.append(obj['rbbox'])
        num += 1
    
    xml_file = open(save_xml, 'w', encoding='utf-8')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>images</folder>\n')
    xml_file.write('    <filename>' + '320__600_0' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + '800' + '</width>\n')
    xml_file.write('        <height>' + '800' + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
    for idx in range(num):
        name = labels[idx]
        # sco  = str(scores[idx])
        rbb  = rbbox[idx]
        pbb = pointobbs[idx]
        xml_file.write('    <object>\n')
        xml_file.write('        <type>robndbox</type>\n')
        xml_file.write('        <name>' + name + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        # xml_file.write('        <probability>' + sco + '</probability>\n')
        xml_file.write('        <robndbox>\n')
        xml_file.write('            <cx>' + str(rbb[0]) + '</cx>\n')
        xml_file.write('            <cy>' + str(rbb[1]) + '</cy>\n')
        xml_file.write('            <w>' + str(rbb[2]) + '</w>\n')
        xml_file.write('            <h>' + str(rbb[3]) + '</h>\n')
        xml_file.write('            <angle>' + str(rbb[4]) + '</angle>\n')
        xml_file.write('        </robndbox>\n')
        xml_file.write('        <segmentation>\n')
        xml_file.write('            <x1>' + str(pbb[0]) + '</x1>\n')
        xml_file.write('            <y1>' + str(pbb[1]) + '</y1>\n')
        xml_file.write('            <x2>' + str(pbb[2]) + '</x2>\n')
        xml_file.write('            <y2>' + str(pbb[3]) + '</y2>\n')
        xml_file.write('            <x3>' + str(pbb[4]) + '</x3>\n')
        xml_file.write('            <y3>' + str(pbb[5]) + '</y3>\n')
        xml_file.write('            <x4>' + str(pbb[6]) + '</x4>\n')
        xml_file.write('            <y4>' + str(pbb[7]) + '</y4>\n')
        xml_file.write('        </segmentation>\n')
        xml_file.write('    </object>\n')
    xml_file.write('</annotation>')

def thetaobb2pointobb(thetaobb):
    """
    docstring here
        :param self: 
        :param thetaobb: list, [x, y, w, h, theta]
    """
    box = cv2.boxPoints(((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]), thetaobb[4]*180.0/np.pi))
    box = np.reshape(box, [-1, ]).tolist()
    pointobb = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]

    return pointobb

def pointobb2bbox(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
        return [xmin, ymin, xmax, ymax]
    """
    xmin = min(pointobb[0::2])
    ymin = min(pointobb[1::2])
    xmax = max(pointobb[0::2])
    ymax = max(pointobb[1::2])
    bbox = [xmin, ymin, xmax, ymax]
    
    return bbox

def rovoc_parse_ref(label_file,diff_x,diff_y,length):
    tree = ET.parse(label_file)
    root = tree.getroot()
    objects = []
    for single_object in root.findall('object'):
        robndbox = single_object.find('robndbox')
        object_struct = {}
        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        cx = cx + diff_x
        cy = cy + diff_y
        if cx >= 0 and cx < (length-1) and cy >= 0 and cy < (length-1) :
            w = float(robndbox.find('w').text)
            h = float(robndbox.find('h').text)
            angle = float(robndbox.find('angle').text)
            rbbox = [cx, cy, w, h, angle]
            pointobb = thetaobb2pointobb(rbbox)
            bbox_list = pointobb2bbox(pointobb)

            xmin = bbox_list[0]
            ymin = bbox_list[1]
            xmax = bbox_list[2]
            ymax = bbox_list[3]
            
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            # pointobb = bbox2pointobb(bbox_list)
            object_struct['segmentation'] = pointobb
            object_struct['rbbox'] = rbbox
            object_struct['pointobb'] = pointobb
            object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
            object_struct['label'] = single_object.find('name').text
            
            objects.append(object_struct)
        else:
            pass
    return objects

def rovoc_parse_dst(label_file,diff_x,diff_y,length):
    tree = ET.parse(label_file)
    root = tree.getroot()
    objects = []
    for single_object in root.findall('object'):
        robndbox = single_object.find('robndbox')
        object_struct = {}
        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        cx_ = cx - diff_x
        cy_ = cy - diff_y
        if cx_ < 0 or cx_ >= (length-1) or cy_ < 0 or cy_ >= (length-1) :
            w = float(robndbox.find('w').text)
            h = float(robndbox.find('h').text)
            angle = float(robndbox.find('angle').text)
            rbbox = [cx, cy, w, h, angle]
            pointobb = thetaobb2pointobb(rbbox)
            bbox_list = pointobb2bbox(pointobb)

            xmin = bbox_list[0]
            ymin = bbox_list[1]
            xmax = bbox_list[2]
            ymax = bbox_list[3]
            
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            # pointobb = bbox2pointobb(bbox_list)
            object_struct['segmentation'] = pointobb
            object_struct['rbbox'] = rbbox
            object_struct['pointobb'] = pointobb
            object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
            object_struct['label'] = single_object.find('name').text
            
            objects.append(object_struct)
        else:
            pass
    return objects



def parse_args():
    parser = argparse.ArgumentParser(description='Count the number of the files in the dir')
    parser.add_argument('ref_xml_path', help='ref_xml_path')
    parser.add_argument('dst_xml_path', help='dst_xml_path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    
    args = parse_args()
    ref_xml = args.ref_xml_path
    dst_xml = args.dst_xml_path
    save_xml = dst_xml

    # ref_xml = r'D:\roLabelImgV2-main\problems1\problems1\resize_5/320__0_0.xml'
    # dst_xml = r'D:\roLabelImgV2-main\problems1\problems1\resize_5/320__600_0.xml'
    # save_xml = dst_xml.split('.xml')[0] + '_copy.xml'

    (ref_filepath, ref_filename) = os.path.split(ref_xml)
    (ref_name, ref_suffix) = os.path.splitext(ref_filename)
    (dst_filepath, dst_filename) = os.path.split(dst_xml)
    (dst_name, dst_suffix) = os.path.splitext(dst_filename)

    # ref_x1,ref_y1 = ref_name.split('__')[1].split('_')
    # dst_x1,dst_y1 = dst_name.split('__')[1].split('_')
    # ref_x1,ref_y1 = int(ref_x1),int(ref_y1)
    # dst_x1,dst_y1 = int(dst_x1),int(dst_y1)
    
    ref_x1,ref_y1 = 600,2400
    dst_x1,dst_y1 = 600,2486

    resize_ratio = 1
    length = 800*resize_ratio

    ref_x1y1 = (ref_x1*resize_ratio,ref_y1*resize_ratio)
    dst_x1y1 = (dst_x1*resize_ratio,dst_y1*resize_ratio)
    diff_x = ref_x1y1[0] - dst_x1y1[0]
    diff_y = ref_x1y1[1] - dst_x1y1[1]

    object_ref = rovoc_parse_ref(ref_xml,diff_x,diff_y,length)
    object_dst = rovoc_parse_dst(dst_xml,diff_x,diff_y,length)
    objects = object_ref + object_dst

    simple_obb_xml_dump(objects,dst_xml)