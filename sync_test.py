"""Testing script for lacmus prediction server. All requests sends sequentially.
Measures response times and standard COCO metrics. """

import requests
import os
import json
import base64
import time
from voc2coco import convert_xmls_to_cocojson
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DATA_DIR = '/home/andrey/ds/data/LizaAlertDroneDatasetV4_Spring'
IMG_DIR = os.path.join(DATA_DIR, 'JPEGImages')
XML_DIR = os.path.join(DATA_DIR, 'Annotations')
ENDPOINT = 'http://localhost:5000/image'
LABELS = {'Background': 0, 'Pedestrian': 1}
TIMES = []


def get_predict(img_id):
    """Send request to server and returns prediction response"""
    img_path = os.path.join(IMG_DIR, img_id)

    with open(img_path, 'rb') as image:
        img_bytes = image.read()

    headers = {'Content-Type': 'application/json'}
    data = {'data': base64.encodebytes(img_bytes).decode("utf-8")}
    json_data = json.dumps(data, indent=4)

    t0 = time.time()
    resp = requests.post(ENDPOINT, headers=headers, data=json_data)
    dt = time.time() - t0
    TIMES.append(dt)

    print('Image: {} predicted in {} sec'.format(img_id, dt))
    response = None
    try:
        response = resp.json()
    except:
        print('empty response')
    return response


def build_true_coco(xml_dir):
    ann_paths = []
    for file_name in os.listdir(XML_DIR):
        xml_path = os.path.join(xml_dir, file_name)
        ann_paths.append(xml_path)

    convert_xmls_to_cocojson(ann_paths, LABELS, 'true.json')
    coco_true = COCO('true.json')
    # os.remove('true.json')
    return coco_true


def get_coco_anno(img_id, predict):
    """:return list of coco annotations from single server prediction"""
    out = []
    for obj in predict['objects']:
        anno = {}
        x1 = int(obj['xmin'])
        y1 = int(obj['ymin'])
        w = int(obj['xmax']) - x1
        h = int(obj['ymax']) - y1
        area = w * h
        anno['area'] = area
        anno['bbox'] = [x1, y1, w, h]
        anno['score'] = float(obj['score'])
        anno['category_id'] = 1
        anno['image_id'] = int(img_id.split('.')[0])
        out.append(anno)
    print(out)
    return out


def get_predict_coco_annos(img_dir):
    """:return list of coco annotations for all predicted images"""
    coco_annos = []
    img_ids = os.listdir(img_dir)
    for img_id in img_ids:
        predict = get_predict(img_id)
        anno_list = get_coco_anno(img_id, predict)
        coco_annos += anno_list
    return coco_annos


if __name__ == '__main__':
    true_coco = build_true_coco(XML_DIR)
    coco_annotations = get_predict_coco_annos(IMG_DIR)
    pred_coco = true_coco.loadRes(coco_annotations)
    E = COCOeval(true_coco, pred_coco, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print('Average response time = {} sec'.format(sum(TIMES) / len(TIMES)))
