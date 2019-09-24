import sys

isPy3 = sys.version_info > (3, 0)

import signal
import argparse
import logging
import time
import glob
import os
import json

if isPy3:
    import queue
else:
    import Queue as queue

import threading

import tensorflow as tf
from tf_pose import common
from tf_pose import eval
import cv2 as cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logging.getLogger().setLevel(logging.WARNING)

tStart = 0
finishedCount = 0

coco_id_to_string = [
    'nose',# 0
    'neck',# 1
    'right_shoulder',# 2
    'right_elbow',# 3
    'right_wrist',# 4
    'left_shoulder',# 5
    'left_elbow',# 6
    'left_wrist',# 7
    'right_hip',# 8
    'right_knee',# 9
    'right_ankle',# 10
    'left_hip',# 11
    'left_knee',# 12
    'left_ankle',# 13
    'right_eye',# 14
    'left_eye',# 15
    'right_ear',# 16
    'left_ear',# 17
    'background',# 18
]

def convert_body_keypoints(human, image_w, image_h):
    body_keypoints = { 'score': human.score, 'face_box': human.get_face_box(image_w, image_h),
        'upper_body_box': human.get_upper_body_box(image_w, image_h), }
    for i, body_part in human.body_parts.items():
        body_keypoints[coco_id_to_string[i]] = { 'x': body_part.x * image_w, 'y': body_part.y * image_h,
            'score': body_part.score, 'coco_id': i }
    return body_keypoints

class InferTask:
    def __init__(self, inputImagePath, sn):
        self.inputImagePath = inputImagePath
        self.sn = sn
        self.inputImage = None
        self.hasError = False
        self.result = None
        self.originResult = None

    def decodeImage(self):
        if self.inputImage is not None:
            return
        t = time.time()
        self.inputImage = common.read_imgfile(self.inputImagePath, None, None)
        if self.inputImage is None:
            self.hasError = True
            self.result = { 'error': { 'code': 'FILE_FORMAT', 'message': 'error read image: ' + self.inputImagePath } }
            logging.error('InferTask:decodeImage error: ' + json.dumps(self.result))
        elapsed = time.time() - t
        logging.info('decode image: %s in %.4f seconds.' % (self.inputImagePath, elapsed))

    def infer(self, tfPoseEstimator):
        if self.result is not None and not self.hasError:
            return
        if self.inputImage is None:
            self.result = { 'error': { 'code': 'INTERNAL_ERROR', 'message': 'image not decoded: ' + self.inputImagePath } }
        else:
            t = time.time()
            try:
                humans = tfPoseEstimator.inference(self.inputImage, resize_to_default=True, upsample_size=4.0)
                
                image_h, image_w = self.inputImage.shape[:2]
                body_list = [convert_body_keypoints(human, image_w, image_h) for human in humans]
                self.result = { 'w': image_w, 'h': image_h, 'src': self.inputImagePath, 'body_list': body_list }
                self.originResult = humans
                self.hasError = False
            except Exception as e:
                self.result = { 'error': { 'code': 'INTERNAL_ERROR', 'message': 'error infer image: ' + self.inputImagePath + ', ' + str(e) } }
                self.hasError = True
            elapsed = time.time() - t
            logging.info('inference image: %s in %.4f seconds.' % (self.inputImagePath, elapsed))

class ParaInferExecutor:
    # don't set perThreadModel = False because it causes race condition
    def __init__(self, paraCount = 1, perThreadModel = True, gpuMemShare = 0.2, modelName = 'mobilenet_v2_large', useTensorrt = False):
        self.paraCount = paraCount
        self.perThreadModel = perThreadModel
        self.gpuMemShare = gpuMemShare
        self.modelName = modelName
        self.useTensorrt = useTensorrt
        self.pendingTask = queue.Queue(0)
        self.finishedTask = queue.Queue(0)
        self.pendingFinishedTask = queue.PriorityQueue(0)
        self.pendingSn = 0
        self.finishedSn = 0
        self.tfPoseEstimator = None
        self._stop = False

    def load(self):
        if self._stop:
            raise Exception('ParaInferExecutor: stopped')

        tfPoseEstimator = None
        if not self.perThreadModel:
            tfPoseEstimator = self.createPoseEstimator('shared')

        for i in range(0, self.paraCount):
            if self.perThreadModel:
                tfPoseEstimator = self.createPoseEstimator('forThread-' + str(i))
            threading.Thread(target=self.inferThreadRun, args=(tfPoseEstimator,)).start()

    def createPoseEstimator(self, label):
        t = time.time()

        #https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= self.gpuMemShare)
        config = tf.ConfigProto(gpu_options=gpu_options)

        tfPoseEstimator = TfPoseEstimator(get_graph_path(self.modelName), target_size=(1024, 576), tf_config=config, trt_bool=self.useTensorrt) # 432, 368
        elapsed = time.time() - t
        logging.info('ParaInferExecutor:createPoseEstimator modelName=%s, label=%s, gpuMemShare=%.2f, in %.2f seconds.'
            % (self.modelName, label, self.gpuMemShare, elapsed))
        return tfPoseEstimator

    def addTask(self, imagePath, decodeImmediately=False):
        task = InferTask(imagePath, self.pendingSn)
        if decodeImmediately:
            task.decodeImage()
        logging.info('ParaInferExecutor:addTask: %s (%d).' % (imagePath, self.pendingSn))
        self.pendingSn += 1
        self.pendingTask.put(task)

    def stop(self):
        self._stop = True
        self.tfPoseEstimator = None

    def getFinishedTask(self):
        return self.finishedTask.get()
    
    def inferThreadRun(self, tfPoseEstimator):
        logging.info('ParaInferExecutor:inferThreadRun:started thread')
        while not self._stop:
            logging.info('ParaInferExecutor:inferThreadRun:looping')
            task = None
            try:
                task = self.pendingTask.get(0.1)
                logging.info('ParaInferExecutor:starting task: %s (%d); pending: %d' % (task.inputImagePath, task.sn, self.pendingTask.qsize()))
                t = time.time()
                task.decodeImage()
                task.infer(tfPoseEstimator)
                elapsed = time.time() - t
                logging.info('ParaInferExecutor:finished task: %s (%d); finished in queue: %d' % (task.inputImagePath,
                    task.sn, self.pendingFinishedTask.qsize() + self.finishedTask.qsize()))
            except Exception as e:
                if task is not None:
                    logging.exception('ParaInferExecutor: uncaught task error: %s (%d);' % (task.inputImagePath, task.sn))

            if task is not None:
                self.pendingFinishedTask.put((task.sn, task))
                self.feedFinishedTask()
    
        logging.info('ParaInferExecutor:inferThreadRun:stopped thread')

    def feedFinishedTask(self):
        while self.pendingFinishedTask.qsize() > 0:
            try:
                sn, task = self.pendingFinishedTask.get_nowait()
                if sn > self.finishedSn:
                    self.pendingFinishedTask.put((sn, task))
                    break
                else:
                    self.finishedSn += 1
                    self.finishedTask.put(task)
            except:
                pass

def outputLoop(paraInferExecutor):
    logging.info('outputLoop:started thread')
    global tStart
    global finishedCount
    while True:
        task = paraInferExecutor.getFinishedTask()
        finishedCount += 1
        elapsed = time.time() - tStart
        logging.info('outputLoop:finished count=%d, mean fps=%.3f, last task sn=%d' % (finishedCount, finishedCount / elapsed, task.sn))

        # if not task.hasError:
        #     outImage = TfPoseEstimator.draw_humans(task.inputImage, task.originResult, imgcopy=False)
        #     outImagePath = task.inputImagePath + '.' + str(finishedCount) + '.kp.jpg'
        #     cv2.imwrite(outImagePath, outImage, [cv2.IMWRITE_JPEG_QUALITY, 90])
        #     logging.info('result image: %s' % (outImagePath))
        
        sResult = json.dumps(task.result)
        # sys.stdout.write(sResult[0:80])
        sys.stdout.write(sResult)
        sys.stdout.write('\n\n')
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description='infer_server_stdio')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--gpu-mem-share', type=float, default=0.2)
    args = parser.parse_args()
    sys.stderr.write('threads:' + str(args.threads) + '\n')

    signal.signal(signal.SIGINT, lambda s, f : os._exit(0))

    exe = ParaInferExecutor(paraCount=args.threads, perThreadModel=True, gpuMemShare=args.gpu_mem_share, useTensorrt=False)
    exe.load()
    threading.Thread(target=outputLoop, args=(exe,)).start()

    sys.stderr.write('please input image file path\n')
    last_line_empty = False
    while (True):
        line = sys.stdin.readline().strip()

        if (line == ''):
            # if (last_line_empty):
            #     sys.stderr.write('bye\n')
            #     sys.exit(0)
            # sys.stderr.write('(empty line. one more empty line to exit.)\n')
            last_line_empty = True
        else:
            global tStart
            if tStart == 0:
                tStart = time.time()
            last_line_empty = False
            logging.info('your input:' + line + '\n')
            task = exe.addTask(line, True)

if __name__ == '__main__':
    main()
