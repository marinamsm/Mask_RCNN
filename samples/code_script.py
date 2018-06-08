import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import json
from pycocotools.coco import COCO

# COCO = COCO('coco/annotations/instances_val2017.json')

YEAR = 2017
IMAGE_PATH = "../../coco/{0}/images/".format(YEAR)
COCO = COCO('../../coco/{0}/annotations/instances_val{0}.json'.format(YEAR))

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "samples/coco/images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
'''class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']'''

cats = COCO.loadCats(COCO.getCatIds())
nms = [cat['name'] for cat in cats]
class_names = nms
catIds = COCO.getCatIds(catNms=class_names)
img_ids = COCO.getImgIds()


# Load a random image from the images folder
'''file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
# Run detection
results = model.detect([image], verbose=0)'''
# print(img_ids)
imgs = COCO.loadImgs(img_ids)
total = len(imgs)
print(total)
results = []
for num, img in enumerate(imgs):
    if (num + 1) % 100 == 0:
        print('{} from {} images.'.format(num + 1, total))
    #out_scores, out_boxes, out_classes = predict(sess, os.path.join("", img['file_name']))
    image = skimage.io.imread(os.path.join(IMAGE_PATH, img['file_name']))
    r = model.detect([image], verbose=0)[0]
    '''print('R')
    print(r)'''
    names = [class_names[index - 1] for index in r['class_ids']]
    out_classes = [COCO.getCatIds(catNms=name)[0] for name in names]
    out_boxes = r['rois'].tolist()
    out_scores = r['scores'].tolist()
    '''print("score: ")
    print(out_scores)
    print("boxes: ")
    print(out_boxes)
    print("class: ")
    print(out_classes)
    print('img_id: ')
    print(img['id'])'''
    for i in range(len(out_scores)):
        # top, left, bottom, right = box
        y1, x1, y2, x2 = out_boxes[i]
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        out_boxes[i] = [x, y, w, h]
        result = {
            "image_id": img['id'],
            "category_id": out_classes[i],
            "bbox": out_boxes[i],
            "score": float(out_scores[i])
        }
        results.append(result)
# print(results)
with open("instances_val2017_results.json", "w") as f:
    json.dump(results, f)


# Visualize results
#r = results[0]
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
#print(r['rois'], r['class_ids'], r['scores'])
# print('RRRR')
# print(results)
