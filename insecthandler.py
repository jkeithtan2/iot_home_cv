import base64
import datetime
import os
import tempfile
from decimal import Decimal

import numpy as np
import boto3 as boto3
import cv2 as cv
import pytz as pytz

import oidlabels

s3 = boto3.resource('s3')
dynamo = boto3.resource('dynamodb')
iothome_table = dynamo.Table('iothome')
s3_bucket_name = os.environ.get('DL_S3')
pbtxt_path = os.environ.get('PBTXT')
graph_path = os.environ.get('GRAPH')
labels = oidlabels.labels
s3_bucket = s3.Bucket(s3_bucket_name)

pbtxt = tempfile.NamedTemporaryFile()
frozen_graph = tempfile.NamedTemporaryFile()
with open(pbtxt.name, 'wb') as pbtxt_file:
    s3_bucket.download_file(pbtxt_path, pbtxt.name)
    pbtxt.flush()
with open(frozen_graph.name, 'wb') as frozen_graph_file:
    s3_bucket.download_file(graph_path, frozen_graph.name)
    frozen_graph.flush()
net = cv.dnn.readNetFromTensorflow(frozen_graph.name,
                                   pbtxt.name)


def check_for_insects(event, context):
    img_str = base64.standard_b64decode(event['data'])
    nparr = np.frombuffer(img_str, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    blob = cv.dnn.blobFromImage(img, size=(512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    cv_out = net.forward()
    detections = {}
    for detection in cv_out[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.3:
            label = labels[int(detection[1])]
            detected_labels = detections.get(label, [])
            # [batchId, classId, confidence, left, top, right, bottom]
            # dynamo does not accept floats
            detection_list = detection.tolist()
            bounding_box = {
                'idx': Decimal(detection_list[1]),
                'score': Decimal(detection_list[2]),
                'left': Decimal(detection_list[3]),
                'top': Decimal(detection_list[4]),
                'right': Decimal(detection_list[5]),
                'bottom': Decimal(detection_list[6])
            }
            detected_labels.append(bounding_box)
            detections[label] = detected_labels
    if detections:
        detection_dt = datetime.datetime.now(tz=pytz.timezone('Asia/Singapore')).strftime("%Y-%m-%dT%H:%M:%S")
        thing_name = 'testthing'
        key_name = f'insects-images/{thing_name}-{detection_dt}'
        s3_bucket.put_object(Key=key_name, Body=img_str)
        iothome_table.put_item(Item={
            'thingid': thing_name,
            'dt': detection_dt,
            'detections': detection_dt,
            'keyName': key_name
        })
