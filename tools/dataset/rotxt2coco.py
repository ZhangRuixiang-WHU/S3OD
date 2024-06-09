import argparse
import os
import cv2
import json
import numpy as np
import mmcv
import xml.etree.ElementTree as ET

def pointobb2pointobb(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    return pointobb.tolist()

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

def thetaobb2pointobb(thetaobb):
    """
    docstring here
        :param self: 
        :param thetaobb: list, [x, y, w, h, theta]
    """
    box = cv2.boxPoints(((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]), thetaobb[4] * 180.0 / np.pi))
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


def pointobb2sampleobb(pointobb, rate):
    """
    pointobb to sampleobb
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
        :param rate: 0 < rate < 0.5, rate=0 -> pointobb, rate=0.5 -> center point
        return [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8]
    """
    px1, py1, px2, py2, px3, py3, px4, py4 = pointobb

    sx1, sy1 = (px1 + px2) // 2, (py1 + py2) // 2
    sx2, sy2 = (px2 + px3) // 2, (py2 + py3) // 2
    sx3, sy3 = (px3 + px4) // 2, (py3 + py4) // 2
    sx4, sy4 = (px4 + px1) // 2, (py4 + py1) // 2

    sx5, sy5 = (1 - rate) * px2 + rate * px4, (1 - rate) * py2 + rate * py4
    sx6, sy6 = (1 - rate) * px3 + rate * px1, (1 - rate) * py3 + rate * py1
    sx7, sy7 = (1 - rate) * px4 + rate * px2, (1 - rate) * py4 + rate * py2
    sx8, sy8 = (1 - rate) * px1 + rate * px3, (1 - rate) * py1 + rate * py3

    sampleobb = [sx1, sy1, sx5, sy5, sx2, sy2, sx6, sy6, sx3, sy3, sx7, sy7, sx4, sy4, sx8, sy8]
    sampleobb = [int(point) for point in sampleobb]
    return sampleobb


def thetaobb2hobb(thetaobb, pointobb_sort_fun):
    """
    docstring here
        :param self: 
        :param thetaobb: list, [x, y, w, h, theta]
    """
    pointobb = thetaobb2pointobb(thetaobb)
    sorted_pointobb = pointobb_sort_fun(pointobb)
    first_point = [sorted_pointobb[0], sorted_pointobb[1]]
    second_point = [sorted_pointobb[2], sorted_pointobb[3]]

    end_point = [sorted_pointobb[6], sorted_pointobb[7]]
    
    h = np.sqrt((end_point[0] - first_point[0])**2 + (end_point[1] - first_point[1])**2)

    hobb = first_point + second_point + [h]
    
    return hobb


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


def hobb2pointobb(hobb):
    """
    docstring here
        :param self: 
        :param hobb: list, [x1, y1, x2, y2, h]
    """
    first_point_x = hobb[0]
    first_point_y = hobb[1]
    second_point_x = hobb[2]
    second_point_y = hobb[3]
    h = hobb[4]

    angle_first_second = np.pi / 2.0 - np.arctan2(second_point_y - first_point_y, second_point_x - first_point_x)
    delta_x = h * np.cos(angle_first_second)
    delta_y = h * np.sin(angle_first_second)

    forth_point_x = first_point_x - delta_x
    forth_point_y = first_point_y + delta_y

    third_point_x = second_point_x - delta_x
    third_point_y = second_point_y + delta_y

    pointobb = [first_point_x, first_point_y, second_point_x, second_point_y, third_point_x, third_point_y, forth_point_x, forth_point_y]

    pointobb = [int(_) for _ in pointobb]
    
    return pointobb



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
                                # "file_name": 'train_obb/split_images/images/'+name + self.image_format,
                                "file_name": name + self.image_format,
                                "seg_file_name": name + self.image_format,
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
        # objects = self.__voc_parse__(annotpath, imgpath)
        objects = self.__rotxt_parse__(annotpath, imgpath)

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
            coco_annotation['area'] = np.float(area)
            coco_annotation['segmentation'] = [segmentation]
            coco_annotation['pointobb'] = pointobb

            coco_annotations.append(coco_annotation)
            
        return coco_annotations
    
    def __rotxt_parse__(self, label_file, image_file):
        objects = []
        file = open(label_file)
        data = file.readlines()
        for ids, subdata in enumerate(data):
            object_struct = {}
            x1, y1, x2, y2, x3, y3, x4, y4, classname, _ = subdata.split()
            pointobb=[float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)]
            # thetaobb = pointobb2thetaobb(pointobb)
            bbox_list = pointobb2bbox(pointobb)
            xmin = bbox_list[0]
            ymin = bbox_list[1]
            xmax = bbox_list[2]
            ymax = bbox_list[3]
            
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            # pointobb = bbox2pointobb(bbox_list)
            object_struct['segmentation'] = pointobb
            object_struct['pointobb'] = pointobb_sort_function[pointobb_sort_method](pointobb)
            object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
            object_struct['label'] = voc_class[classname]
        
            objects.append(object_struct)
        return objects

    def __voc_parse__(self, label_file, image_file):
        tree = ET.parse(label_file)
        root = tree.getroot()
        objects = []
        for single_object in root.findall('object'):
            robndbox = single_object.find('robndbox')
            object_struct = {}
            # try:
            #     cx = float(robndbox.find('cx').text)
            # except:
            #     print(label_file)
            cx = float(robndbox.find('cx').text)
            cy = float(robndbox.find('cy').text)
            w = float(robndbox.find('w').text)
            h = float(robndbox.find('h').text)
            angle = float(robndbox.find('angle').text)

            

            # xmin = float(bndbox.find('xmin').text)
            # ymin = float(bndbox.find('ymin').text)
            # xmax = float(bndbox.find('xmax').text)
            # ymax = float(bndbox.find('ymax').text)
            pointobb = thetaobb2pointobb([cx, cy, w, h, angle])
            bbox_list = pointobb2bbox(pointobb)

            xmin = bbox_list[0]
            ymin = bbox_list[1]
            xmax = bbox_list[2]
            ymax = bbox_list[3]
            
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            # pointobb = bbox2pointobb(bbox_list)
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
    info = {"year" : 2020,
            "version" : "1.0",
            "description" : "DOTA_DIOR-COCO",
            "contributor" : "Ruixiang Zhang",
            "url" : "zhangruixiang-whu.github.io",
            "date_created" : "2020"
            }
    
    licenses = [{"id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    image_format='.png'
    anno_format='.txt'

    voc_class = {'large-vehicle': 1, 'small-vehicle': 2, 'plane': 3, 'bridge': 4, 'ship': 5, 'harbor': 6, 'swimming-pool': 7, 'roundabout': 8, 'soccer-ball-field': 9, 
                   'helicopter': 10, 'storage-tank': 11, 'tennis-court': 12, 'baseball-diamond': 13, 'basketball-court': 14, 'ground-track-field': 15, 
                   'container-crane': 16, 'airport': 17,'helipad': 18}
    coco_class = [{'supercategory': 'none', 'id': 1,   'name': 'large-vehicle',     },
                  {'supercategory': 'none', 'id': 2,   'name': 'small-vehicle',     },
                  {'supercategory': 'none', 'id': 3,   'name': 'plane',             },
                  {'supercategory': 'none', 'id': 4,   'name': 'bridge'             },
                  {'supercategory': 'none', 'id': 5,   'name': 'ship',              },
                  {'supercategory': 'none', 'id': 6,   'name': 'harbor',            },
                  {'supercategory': 'none', 'id': 7,   'name': 'swimming-pool',     },
                  {'supercategory': 'none', 'id': 8,   'name': 'roundabout'         },
                  {'supercategory': 'none', 'id': 9,   'name': 'soccer-ball-field', },
                  {'supercategory': 'none', 'id': 10,  'name': 'helicopter',        },
                  {'supercategory': 'none', 'id': 11,  'name': 'storage-tank',      },
                  {'supercategory': 'none', 'id': 12,  'name': 'tennis-court'       },
                  {'supercategory': 'none', 'id': 13,  'name': 'baseball-diamond',  },
                  {'supercategory': 'none', 'id': 14,  'name': 'basketball-court',  },
                  {'supercategory': 'none', 'id': 15,  'name': 'ground-track-field',},
                  {'supercategory': 'none', 'id': 16,  'name': 'container-crane'    },
                  {'supercategory': 'none', 'id': 17,  'name': 'airport'            },
                  {'supercategory': 'none', 'id': 18,  'name': 'helipad'            }]

    imagesets = ['train_obb','val_obb','labeled_target_obb',]#
    core_dataset = 'Rotated/SSDA_dotav2_set2'
    groundtruth = True
    # release_version = 'v1'

    pointobb_sort_method = 'best' # or "extreme"
    pointobb_sort_function = {"best": pointobb_best_point_sort,
                            "extreme": pointobb_extreme_sort}

    for imageset in imagesets:
        # /data/zrx/sn6-det/SAR_v1/
        imgpath = '/data/zrx/{}/{}/split_images/images'.format(core_dataset, imageset)
        annopath = '/data/zrx/{}/{}/split_images/annfiles'.format(core_dataset, imageset)
        save_path = '/data/zrx/{}/annotations'.format(core_dataset, imageset)

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

        with open(os.path.join(save_path, "ssda_set2_{}.json".format(imageset)), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)

