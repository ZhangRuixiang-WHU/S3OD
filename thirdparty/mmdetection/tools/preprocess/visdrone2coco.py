import os
import cv2
import mmcv
import argparse
import json
import numpy as np
import xml.etree.ElementTree as ET

def bbox2pointobb(bbox):
    """
    docstring here
        :param self: 
        :param bbox: list, [xmin, ymin, xmax, ymax]
        return [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    xmin, ymin, xmax, ymax = bbox
    x1, y1 = xmin, ymin
    x2, y2 = xmax, ymin
    x3, y3 = xmax, ymax
    x4, y4 = xmin, ymax

    pointobb = [x1, y1, x2, y2, x3, y3, x4, y4]
    
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

def pointobb_extreme_sort(pointobb):
    """
    Find the "top" point and sort all points as the "top right bottom left" order
        :param self: self
        :param points: unsorted points, (N*8) 
    """   
    points_np = np.array(pointobb)
    points_np.resize(4, 2)
    # sort by Y
    sorted_index = np.argsort(points_np[:, 1])
    points_sorted = points_np[sorted_index, :]
    if points_sorted[0, 1] == points_sorted[1, 1]:
        if points_sorted[0, 0] < points_sorted[1, 0]:
            sorted_top_idx = 0
        else:
            sorted_top_idx = 1
    else:
        sorted_top_idx = 0

    top_idx = sorted_index[sorted_top_idx]
    pointobb = pointobb[2*top_idx:] + pointobb[:2*top_idx]
    
    return pointobb

def pointobb_best_point_sort(pointobb):
    """
    Find the "best" point and sort all points as the order that best point is first point
        :param self: self
        :param points: unsorted points, (N*8) 
    """
    xmin, ymin, xmax, ymax = pointobb2bbox(pointobb)
    w = xmax - xmin
    h = ymax - ymin
    reference_bbox = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
    reference_bbox = np.array(reference_bbox)
    normalize = np.array([1.0, 1.0] * 4)
    combinate = [np.roll(pointobb, 0), np.roll(pointobb, 2), np.roll(pointobb, 4), np.roll(pointobb, 6)]
    distances = np.array([np.sum(((coord - reference_bbox) / normalize)**2) for coord in combinate])
    sorted = distances.argsort()

    return combinate[sorted[0]].tolist()

class Convert2COCO():
    def __init__(self, 
                imgpath=None,
                annopath=None,
                imageset_file=None,
                image_format='.jpg',
                anno_format='.txt',
                data_categories=None,
                data_info=None,
                data_licenses=None,
                data_type="instances",
                groundtruth=True,
                small_object_area=0,
                sub_anno_fold=False):
        super(Convert2COCO, self).__init__()

        self.imgpath = imgpath
        self.annopath = annopath
        self.image_format = image_format
        self.anno_format = anno_format

        self.categories = data_categories
        self.info = data_info
        self.licenses = data_licenses
        self.type = data_type
        self.small_object_area = small_object_area
        self.small_object_idx = 0
        self.groundtruth = groundtruth
        self.max_object_num_per_image = 0
        self.sub_anno_fold = sub_anno_fold
        self.imageset_file = imageset_file

        self.imlist = []
        if self.imageset_file:
            with open(self.imageset_file, 'r') as f:
                lines = f.readlines()
            for img_name in lines:
                img_name = img_name.strip('\n')
                self.imlist.append(img_name)
            print("Loading image names from imageset file, image number: {}".format(len(self.imlist)))
        else:
            for img_name in os.listdir(self.imgpath):
                if img_name.endswith(self.image_format):
                    img_name = img_name.split(self.image_format)[0]
                    self.imlist.append(img_name)
                else:
                    continue
                
    def get_image_annotation_pairs(self):
        images = []
        annotations = []
        index = 0
        progress_bar = mmcv.ProgressBar(len(self.imlist))
        imId = 0
        for name in self.imlist:
            imgpath = os.path.join(self.imgpath, name + self.image_format)
            if self.sub_anno_fold:
                annotpath = os.path.join(self.annopath, name, name + self.anno_format)
            else:
                annotpath = os.path.join(self.annopath, name + self.anno_format)

            annotations_coco = self.__generate_coco_annotation__(annotpath, imgpath)

            # if annotation is empty, skip this annotation
            if annotations_coco != [] or self.groundtruth == False:
                img = cv2.imread(imgpath)
                height, width, channels = img.shape
                images.append({"date_captured": "2019",
                                "file_name": name + self.image_format,
                                "id": imId + 1,
                                "license": 1,
                                "url": "http://jwwangchn.cn",
                                "height": height,
                                "width": width})

                for annotation in annotations_coco:
                    index = index + 1
                    annotation["iscrowd"] = 0
                    annotation["image_id"] = imId + 1
                    annotation["id"] = index
                    annotations.append(annotation)

                imId += 1

            if imId % 500 == 0:
                print("\nImage ID: {}, Instance ID: {}, Small Object Counter: {}, Max Object Number: {}".format(imId, index, self.small_object_idx, self.max_object_num_per_image))
            
            progress_bar.update()
            

        return images, annotations

    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """   
        raise NotImplementedError


class VOC2COCO(Convert2COCO):
    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """
        objects = self.__voc_parse__(annotpath, imgpath)

        coco_annotations = []
        
        for object_struct in objects:
            bbox = object_struct['bbox']
            label = object_struct['label']
            segmentation = object_struct['segmentation']
            pointobb = object_struct['pointobb']

            width = bbox[2]
            height = bbox[3]
            area = height * width

            if area < self.small_object_area and self.groundtruth:
                self.small_object_idx += 1
                continue

            coco_annotation = {}
            coco_annotation['bbox'] = bbox
            coco_annotation['category_id'] = label
            coco_annotation['area'] = np.float64(area)
            coco_annotation['segmentation'] = [segmentation]
            coco_annotation['pointobb'] = pointobb

            coco_annotations.append(coco_annotation)
            
        return coco_annotations
    
    def __voc_parse__(self, label_file, image_file):
        tree = ET.parse(label_file)
        root = tree.getroot()
        objects = []
        for single_object in root.findall('object'):
            bndbox = single_object.find('bndbox')
            object_struct = {}

            try: 
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
            except:
                print(label_file)
                # raise Exception
            
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            pointobb = bbox2pointobb([xmin, ymin, xmax, ymax])
            object_struct['segmentation'] = pointobb
            object_struct['pointobb'] = pointobb_sort_function[pointobb_sort_method](pointobb)
            object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
            object_struct['label'] = voc_class[single_object.find('name').text]
            
            objects.append(object_struct)
        return objects

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument(
        '--imagesets',
        type=str,
        nargs='+',
        choices=['trainval', 'test'])
    parser.add_argument(
        '--release_version', default='v1', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # basic dataset information
    info = {"year" : 2022,
            "version" : "1.0",
            "description" : "MSAR-COCO",
            "contributor" : "Ruixiang Zhang",
            "url" : "zhangruixiang-whu.github.io",
            "date_created" : "2022"
            }
    
    licenses = [{"id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    image_format='.jpg'
    anno_format='.xml'

    voc_class = {'car':1, 'feright_car':2, 'bus':3, 'truck':4, 'van':5}
    coco_class = [{'supercategory': 'none', 'id': 1,  'name': 'car',        },
                {'supercategory': 'none', 'id': 2,  'name': 'feright_car',  },
                {'supercategory': 'none', 'id': 3,  'name': 'bus',          },
                {'supercategory': 'none', 'id': 4,  'name': 'truck',        },
                {'supercategory': 'none', 'id': 5,  'name': 'van',          },]


    imagesets = ['train',]
    core_dataset = 'VisDrone'
    groundtruth = True
    # release_version = 'v1'

    pointobb_sort_method = 'best' # or "extreme"
    pointobb_sort_function = {"best": pointobb_best_point_sort,
                            "extreme": pointobb_extreme_sort}

    for imageset in imagesets:
        # /data/zrx/VisDrone/train/
        imgpath = '/data/zrx/{}/{}/trainimgr'.format(core_dataset, imageset)
        annopath = '/data/zrx/{}/{}/trainlabelr_hbb'.format(core_dataset, imageset)
        save_path = '/data/zrx/{}/coco_r/annotations'.format(core_dataset, imageset)
        # # /data/zrx/CMOD/DOTA_4cls
        # imgpath = '/data/zrx/{}/{}/images'.format(core_dataset, imageset)
        # annopath = '/data/zrx/{}/{}/labels'.format(core_dataset, imageset)
        # save_path = '/data/zrx/{}/coco/annotations'.format(core_dataset, imageset)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        voc = VOC2COCO(imgpath=imgpath,
                        annopath=annopath,
                        image_format=image_format,
                        anno_format=anno_format,
                        data_categories=coco_class,
                        data_info=info,
                        data_licenses=licenses,
                        data_type="instances",
                        groundtruth=groundtruth,
                        small_object_area=0)

        images, annotations = voc.get_image_annotation_pairs()

        json_data = {"info" : voc.info,
                    "images" : images,
                    "licenses" : voc.licenses,
                    "type" : voc.type,
                    "annotations" : annotations,
                    "categories" : voc.categories}

        with open(os.path.join(save_path, "instances_{}2017.json".format(imageset)), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)

