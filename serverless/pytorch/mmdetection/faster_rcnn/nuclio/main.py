import os
import json
import base64
import io
from PIL import Image
import numpy as np

import torch
from mmdetection.mmdet.apis import (inference_detector, init_detector)

def init_context(context):
    #init model
    context.logger.info("Init context...  0%")
    

    config = "mmdetection/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py"
    checkpoint = "mmdetection/checkpoint/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth"

    # initialize the detector
    if torch.cuda.is_available():
        model = init_detector(config, checkpoint, device='cuda:0')
    else:
        model = init_detector(config, checkpoint, device='cpu')

    context.user_data.model_handler = model
    

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info('This is an unstructured log')
    # inference
    
    context.logger.info("Run faster rcnn model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf)
    image = np.array(image)

    predictions = inference_detector(context.user_data.model_handler, image)

    results = []
    coco_class_list = [
        { "id": 1, "name": "person" },
        { "id": 2, "name": "bicycle" },
        { "id": 3, "name": "car" },
        { "id": 4, "name": "motorcycle" },
        { "id": 5, "name": "airplane" },
        { "id": 6, "name": "bus" },
        { "id": 7, "name": "train" },
        { "id": 8, "name": "truck" },
        { "id": 9, "name": "boat" },
        { "id":10, "name": "traffic_light" },
        { "id":11, "name": "fire_hydrant" },
        { "id":13, "name": "stop_sign" },
        { "id":14, "name": "parking_meter" },
        { "id":15, "name": "bench" },
        { "id":16, "name": "bird" },
        { "id":17, "name": "cat" },
        { "id":18, "name": "dog" },
        { "id":19, "name": "horse" },
        { "id":20, "name": "sheep" },
        { "id":21, "name": "cow" },
        { "id":22, "name": "elephant" },
        { "id":23, "name": "bear" },
        { "id":24, "name": "zebra" },
        { "id":25, "name": "giraffe" },
        { "id":27, "name": "backpack" },
        { "id":28, "name": "umbrella" },
        { "id":31, "name": "handbag" },
        { "id":32, "name": "tie" },
        { "id":33, "name": "suitcase" },
        { "id":34, "name": "frisbee" },
        { "id":35, "name": "skis" },
        { "id":36, "name": "snowboard" },
        { "id":37, "name": "sports_ball" },
        { "id":38, "name": "kite" },
        { "id":39, "name": "baseball_bat" },
        { "id":40, "name": "baseball_glove" },
        { "id":41, "name": "skateboard" },
        { "id":42, "name": "surfboard" },
        { "id":43, "name": "tennis_racket" },
        { "id":44, "name": "bottle" },
        { "id":46, "name": "wine_glass" },
        { "id":47, "name": "cup" },
        { "id":48, "name": "fork" },
        { "id":49, "name": "knife" },
        { "id":50, "name": "spoon" },
        { "id":51, "name": "bowl" },
        { "id":52, "name": "banana" },
        { "id":53, "name": "apple" },
        { "id":54, "name": "sandwich" },
        { "id":55, "name": "orange" },
        { "id":56, "name": "broccoli" },
        { "id":57, "name": "carrot" },
        { "id":58, "name": "hot_dog" },
        { "id":59, "name": "pizza" },
        { "id":60, "name": "donut" },
        { "id":61, "name": "cake" },
        { "id":62, "name": "chair" },
        { "id":63, "name": "couch" },
        { "id":64, "name": "potted_plant" },
        { "id":65, "name": "bed" },
        { "id":67, "name": "dining_table" },
        { "id":70, "name": "toilet" },
        { "id":72, "name": "tv" },
        { "id":73, "name": "laptop" },
        { "id":74, "name": "mouse" },
        { "id":75, "name": "remote" },
        { "id":76, "name": "keyboard" },
        { "id":77, "name": "cell_phone" },
        { "id":78, "name": "microwave" },
        { "id":79, "name": "oven" },
        { "id":80, "name": "toaster" },
        { "id":81, "name": "sink" },
        { "id":83, "name": "refrigerator" },
        { "id":84, "name": "book" },
        { "id":85, "name": "clock" },
        { "id":86, "name": "vase" },
        { "id":87, "name": "scissors" },
        { "id":88, "name": "teddy_bear" },
        { "id":89, "name": "hair_drier" },
        { "id":90, "name": "toothbrush" }
      ]
    for i, pred in enumerate(predictions):
        label = coco_class_list[i]["name"]
        for obj in pred: 
            score = obj[-1]
            if score >= threshold:
                results.append({
                    "confidence": str(float(score)),
                    "label": label,
                    "points": obj[:4].tolist(),
                    "type": "rectangle",
                })
    
    #results = []
    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)

    

    cwd = os.getcwd()
    files = os.listdir(".")

    return context.Response(body='Hello, from nuclio :] \n'+cwd+" \n"+" ".join(files),
                            headers={},
                            content_type='text/plain',
                            status_code=200)