import argparse
import logging
import sys
import time
import glob
import os
import json

from tf_pose import common
from tf_pose import eval
import cv2 as cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
#logger.handlers.clear()
logger.handlers[:]
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def processSingleImage(poseEstimator, inPath, outPath, resize_to_default, scale):
    # estimate human poses from a single image !
    image = common.read_imgfile(inPath, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % inPath)
        sys.exit(-1)

    t = time.time()
    humans = poseEstimator.inference(image, resize_to_default, scale)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds, out: %s' % (inPath, elapsed, outPath))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    cv2.imwrite(outPath, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--out-image', type=str, default='./images/p1-out.jpg')
    parser.add_argument('--image-dir', type=str, default='./images/')
    parser.add_argument('--out-image-dir', type=str, default='./images-out/')
    parser.add_argument('--model', type=str, default='mobilenet_v2_large',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0, # 1.0 is not usable as keypoints would disappear
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--repeat', type=int, default=1)

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    resize_to_default = w > 0 and h > 0

    #estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    logger.info('loaded image: %s ' % (args.image))

    totalTime = 0
    humans = None
    for i in range(0, args.repeat):
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t
        logger.info('[%d]inference image: %s in %.4f seconds.' % (i, args.image, elapsed))
        totalTime += elapsed

    logger.info('average inference time: %.4f seconds.' % (totalTime / args.repeat))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    cv2.imwrite(args.out_image, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    logger.info('result image: %s' % (args.image))

    image_h, image_w = image.shape[:2]
    body_list = [eval.get_keypoint_dict(human, image_w, image_h) for human in humans]
    frame_of_body = { 'w': image_w, 'h': image_h, 'src': args.image, 'body_list': body_list }
    body_json = json.dumps(frame_of_body)
    logger.info('body_json: %s' % (body_json))
