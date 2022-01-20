from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import math
from form import Ui_Form
import pyrealsense2 as rs
import copy
import cv2
from threading import Thread, Event
import time
import numpy as np
import json
from kalman import KalmanFilter

config = json.load(open('config.json'))

value_change = QtCore.pyqtSignal(object)
width_change = QtCore.pyqtSignal(float)





class MainLineProcessor(Thread):
    def __init__(self, params, parent=None):
        Thread.__init__(self, parent)
        self._stop = Event()
        self.kalman = KalmanFilter(initial_estimate=50.0, initial_est_error=1, initial_measure_error=1)

        self.pipeline = rs.pipeline()
        # Configure streams
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, params['camera']['x'], params['camera']['y'], rs.format.z16,
                                  params['camera']['fps'])
        self.pipeline.start(self.config)
        self.pix_border_w = 10
        self.pix_border_h = 10
        self.work_zone = (params['work_zone'][0], params['work_zone'][1], params['work_zone'][2], params['work_zone'][3])
        self.cam_h = params['camera']['y']
        self.cam_w = params['camera']['x']
        self.area_screen = self.cam_h * self.cam_w
        self.max_dist = params['distance']['max']
        self.min_dist = params['distance']['min']
        self.dilate_enable = params['dilate']['enable']
        self.kernel_size = params['dilate']['kernel']
        self.kernel = np.ones((self.kernel_size, self.kernel_size), 'uint8')
        if self.cam_w == 640:
            self.fov_x_2 = np.tan(np.radians(78.5 / 2))
            self.fov_y_2 = np.tan(np.radians(64 / 2))
        else:
            self.fov_x_2 = np.tan(np.radians(87 / 2))
            self.fov_y_2 = np.tan(np.radians(658 / 2))
        self.img_w_2 = self.cam_w / 2
        self.img_h_2 = self.cam_h / 2

    def stop(self):
        self._stop.set()
        self.pipeline.stop()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        while True:
            try:
                frames = self.pipeline.wait_for_frames()
                depth = frames.get_depth_frame()
                if not depth: continue

                depth2 = np.asanyarray(depth.get_data()) / 1000.0  # distance in meters
                depth1 = np.zeros((self.cam_h, self.cam_w))
                depth1[self.work_zone[1]:self.work_zone[3], self.work_zone[0]:self.work_zone[2]] = depth2[self.work_zone[1]:self.work_zone[3], self.work_zone[0]:self.work_zone[2]]
                original_copy = copy.deepcopy(depth1)

                if depth1.shape[0] != 480:
                    depth1 = cv2.resize(depth1, (640, 480))
                depth1[(depth1 > self.max_dist) | (depth1 < self.min_dist)] = 255
                depth1[(depth1 <= self.max_dist) & (0 < depth1)] = 0

                border = np.zeros((self.cam_h, self.cam_w)) + 255
                border[self.pix_border_h:-self.pix_border_h, self.pix_border_w:-self.pix_border_w] = depth1[
                                                                                                     self.pix_border_h:-self.pix_border_h,
                                                                                                     self.pix_border_w:-self.pix_border_w]
                depth1 = border
                ret, thresh_img = cv2.threshold(depth1, 10, 255, 0)
                thresh_img = thresh_img.astype(np.uint8)
                if self.dilate_enable:
                    thresh_img = cv2.dilate(thresh_img, self.kernel, iterations=1)

                contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnt_sorted = sorted(contours, key=len, reverse=True)
                cnt = cnt_sorted[0]
                area = cv2.contourArea(cnt)
                image_copy = np.zeros((self.cam_h, self.cam_w))
                if self.area_screen * 0.9 < area:
                    if len(cnt_sorted) > 1:
                        cnt = sorted(contours, key=len, reverse=True)[1]
                    else:
                        cnt = None
                if cnt is not None:

                    rbox = cv2.minAreaRect(cnt)
                    pts = cv2.boxPoints(rbox).astype(np.int32)
                    cv2.drawContours(image_copy, [pts], -1, 255, -1)
                    #print("centre ", rbox[0] ," w,h ", rbox[1], " angle ", rbox[2])

                    if rbox[1][0] >= rbox[1][1]:
                        ln_w = int(rbox[1][1] * 0.3)

                        angle2 = np.pi - np.radians(int(180-rbox[2]))

                        x1, y1 = int(rbox[0][0]), int(rbox[0][1])
                        length = rbox[1][0] / 2
                        x1_0 = int(x1 + length * np.cos(angle2))
                        y1_0 = int(y1 + length * np.sin(angle2))

                        angle1 = np.pi +angle2
                        x2_0 = int(x1 + length * np.cos(angle1))
                        y2_0 = int(y1 + length * np.sin(angle1))


                        thresh_img = cv2.line(thresh_img, (x1_0, y1_0), (x2_0, y2_0), (255, 255, 255), ln_w)
                        image_copy = cv2.line(image_copy, (x1_0, y1_0), (x2_0, y2_0), 0, ln_w)

                        w = rbox[1][0]
                        h = rbox[1][1]

                        delta_x = pts[1][0] - pts[0][0]
                        delta_y = pts[0][1] - pts[1][1]



                        #print("w ",int(rbox[1][0]) ," h ", int(rbox[1][1]), "angle ", int(180-rbox[2]), "  ", angle2)
                    else :#rbox[1][0] <rbox[1][1]:
                        angle2= np.pi - np.radians(int(90 - rbox[2]))
                        ln_w = int(rbox[1][0] * 0.3)
                        x1, y1 = int(rbox[0][0]), int(rbox[0][1])
                        length = rbox[1][1] / 2
                        x1_0 = int(x1 + length * np.cos(angle2))
                        y1_0 = int(y1 + length * np.sin(angle2))


                        angle1 = np.pi +angle2
                        x2_0 = int(x1 + length * np.cos(angle1))
                        y2_0 = int(y1 + length * np.sin(angle1))


                        thresh_img = cv2.line(thresh_img, (x1_0, y1_0), (x2_0, y2_0), 255, ln_w)
                        image_copy = cv2.line(image_copy, (x1_0, y1_0), (x2_0, y2_0), 0, ln_w)

                        w = rbox[1][1]
                        h = rbox[1][0]

                        delta_x = pts[2][0] - pts[1][0]
                        delta_y = pts[2][1] - pts[1][1]

                        #print("w ",int(rbox[1][0]) ," h ", int(rbox[1][1]), "angle ",int(90-rbox[2]))

                    A = np.argwhere(image_copy >= 255)
                    res = np.array([original_copy[v[0], v[1]] for v in A])
                    res[res <= 0.1] = np.nan
                    dist = np.nanmean(res)
                    px_mm_x = (dist * 1000 * self.fov_x_2) / self.img_w_2
                    px_mm_y = (dist * 1000 * self.fov_y_2) / self.img_h_2

                    delta_x_mm = delta_x * px_mm_x
                    delta_y_mm = delta_y * px_mm_y
                    res_W = np.sqrt((delta_x_mm ** 2 + delta_y_mm ** 2))
                    #print(res_W)
                    self.kalman.iterative_updates(res_W)







                    # print("pix per mm x ",px_mm_x, " pix per mm y ", px_mm_y)
                    #print(dist, " x ", px_mm_x * w, " y ", px_mm_y * h)


                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # org = (int(rbox[0][0]), int(rbox[0][1]))
                    # fontScale = 1
                    # color = 255
                    # thickness = 3
                    # thresh_img = cv2.putText(thresh_img, "C", org, font,
                    #                          fontScale, color, thickness, cv2.LINE_AA)

                    for p in range(len(pts)):
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        org = (pts[p][0], pts[p][1])
                        fontScale = 1
                        color = 0
                        thickness = 3
                        thresh_img = cv2.putText(thresh_img, str(p), org, font,
                                             fontScale, color, thickness, cv2.LINE_AA)

                    thresh_img = cv2.drawContours(thresh_img, [pts], -1, (0, 255, 0), 1, cv2.LINE_AA)

                    work_r_points = [np.array(([self.work_zone[0], self.work_zone[1]],
                              [self.work_zone[2], self.work_zone[1]],
                              [self.work_zone[2], self.work_zone[3]],
                              [self.work_zone[0], self.work_zone[3]]
                              ))]
                    thresh_img = cv2.drawContours(thresh_img, work_r_points,
                                                  -1, (0, 255, 0), 1, cv2.LINE_AA)

                #print("width ", self.kalman.estimate)
                self.width_change.emit(self.kalman.estimate)
                self.value_change.emit(thresh_img)
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            if self.stopped():
                return






def main():


    cam_w = config['camera']['x']  # 1280
    cam_h = config['camera']['y']  # 720
    cam_fps = config['camera']['fps']  # 720

    max_dist = config['distance']['max']
    min_dist = config['distance']['min']

    dilate_enable = config['dilate']['enable']
    dilate_kernel = config['dilate']['kernel']
    pix_border_h = 10
    pix_border_w = 10
    kernel = np.ones((dilate_kernel, dilate_kernel), 'uint8')
    work_zone = (config['work_zone'][0], config['work_zone'][1], config['work_zone'][2], config['work_zone'][3])
    stream_frame = np.zeros((480, 640))

    #_________________________________________________
    frames = self.pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    if not depth: continue

    depth2 = np.asanyarray(depth.get_data()) / 1000.0  # distance in meters
    depth1 = np.zeros((self.cam_h, self.cam_w))
    depth1[self.work_zone[1]:self.work_zone[3], self.work_zone[0]:self.work_zone[2]] = depth2[
                                                                                       self.work_zone[1]:self.work_zone[
                                                                                           3],
                                                                                       self.work_zone[0]:self.work_zone[
                                                                                           2]]
    original_copy = copy.deepcopy(depth1)

    if depth1.shape[0] != 480:
        depth1 = cv2.resize(depth1, (640, 480))
    depth1[(depth1 > self.max_dist) | (depth1 < self.min_dist)] = 255
    depth1[(depth1 <= self.max_dist) & (0 < depth1)] = 0

    border = np.zeros((self.cam_h, self.cam_w)) + 255
    border[self.pix_border_h:-self.pix_border_h, self.pix_border_w:-self.pix_border_w] = depth1[
                                                                                         self.pix_border_h:-self.pix_border_h,
                                                                                         self.pix_border_w:-self.pix_border_w]
    depth1 = border
    ret, thresh_img = cv2.threshold(depth1, 10, 255, 0)
    thresh_img = thresh_img.astype(np.uint8)
    if self.dilate_enable:
        thresh_img = cv2.dilate(thresh_img, self.kernel, iterations=1)

    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_sorted = sorted(contours, key=len, reverse=True)
    cnt = cnt_sorted[0]
    area = cv2.contourArea(cnt)
    image_copy = np.zeros((self.cam_h, self.cam_w))
    if self.area_screen * 0.9 < area:
        if len(cnt_sorted) > 1:
            cnt = sorted(contours, key=len, reverse=True)[1]
        else:
            cnt = None
    if cnt is not None:

        rbox = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rbox).astype(np.int32)
        cv2.drawContours(image_copy, [pts], -1, 255, -1)
        # print("centre ", rbox[0] ," w,h ", rbox[1], " angle ", rbox[2])

        if rbox[1][0] >= rbox[1][1]:
            ln_w = int(rbox[1][1] * 0.3)

            angle2 = np.pi - np.radians(int(180 - rbox[2]))

            x1, y1 = int(rbox[0][0]), int(rbox[0][1])
            length = rbox[1][0] / 2
            x1_0 = int(x1 + length * np.cos(angle2))
            y1_0 = int(y1 + length * np.sin(angle2))

            angle1 = np.pi + angle2
            x2_0 = int(x1 + length * np.cos(angle1))
            y2_0 = int(y1 + length * np.sin(angle1))

            thresh_img = cv2.line(thresh_img, (x1_0, y1_0), (x2_0, y2_0), (255, 255, 255), ln_w)
            image_copy = cv2.line(image_copy, (x1_0, y1_0), (x2_0, y2_0), 0, ln_w)

            w = rbox[1][0]
            h = rbox[1][1]

            delta_x = pts[1][0] - pts[0][0]
            delta_y = pts[0][1] - pts[1][1]

            # print("w ",int(rbox[1][0]) ," h ", int(rbox[1][1]), "angle ", int(180-rbox[2]), "  ", angle2)
        else:  # rbox[1][0] <rbox[1][1]:
            angle2 = np.pi - np.radians(int(90 - rbox[2]))
            ln_w = int(rbox[1][0] * 0.3)
            x1, y1 = int(rbox[0][0]), int(rbox[0][1])
            length = rbox[1][1] / 2
            x1_0 = int(x1 + length * np.cos(angle2))
            y1_0 = int(y1 + length * np.sin(angle2))

            angle1 = np.pi + angle2
            x2_0 = int(x1 + length * np.cos(angle1))
            y2_0 = int(y1 + length * np.sin(angle1))

            thresh_img = cv2.line(thresh_img, (x1_0, y1_0), (x2_0, y2_0), 255, ln_w)
            image_copy = cv2.line(image_copy, (x1_0, y1_0), (x2_0, y2_0), 0, ln_w)

            w = rbox[1][1]
            h = rbox[1][0]

            delta_x = pts[2][0] - pts[1][0]
            delta_y = pts[2][1] - pts[1][1]

            # print("w ",int(rbox[1][0]) ," h ", int(rbox[1][1]), "angle ",int(90-rbox[2]))

        A = np.argwhere(image_copy >= 255)
        res = np.array([original_copy[v[0], v[1]] for v in A])
        res[res <= 0.1] = np.nan
        dist = np.nanmean(res)
        px_mm_x = (dist * 1000 * self.fov_x_2) / self.img_w_2
        px_mm_y = (dist * 1000 * self.fov_y_2) / self.img_h_2

        delta_x_mm = delta_x * px_mm_x
        delta_y_mm = delta_y * px_mm_y
        res_W = np.sqrt((delta_x_mm ** 2 + delta_y_mm ** 2))
        # print(res_W)
        self.kalman.iterative_updates(res_W)

        # print("pix per mm x ",px_mm_x, " pix per mm y ", px_mm_y)
        # print(dist, " x ", px_mm_x * w, " y ", px_mm_y * h)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # org = (int(rbox[0][0]), int(rbox[0][1]))
        # fontScale = 1
        # color = 255
        # thickness = 3
        # thresh_img = cv2.putText(thresh_img, "C", org, font,
        #                          fontScale, color, thickness, cv2.LINE_AA)

        for p in range(len(pts)):
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (pts[p][0], pts[p][1])
            fontScale = 1
            color = 0
            thickness = 3
            thresh_img = cv2.putText(thresh_img, str(p), org, font,
                                     fontScale, color, thickness, cv2.LINE_AA)

        thresh_img = cv2.drawContours(thresh_img, [pts], -1, (0, 255, 0), 1, cv2.LINE_AA)

        work_r_points = [np.array(([self.work_zone[0], self.work_zone[1]],
                                   [self.work_zone[2], self.work_zone[1]],
                                   [self.work_zone[2], self.work_zone[3]],
                                   [self.work_zone[0], self.work_zone[3]]
                                   ))]
        thresh_img = cv2.drawContours(thresh_img, work_r_points,
                                      -1, (0, 255, 0), 1, cv2.LINE_AA)

    # print("width ", self.kalman.estimate)
    self.width_change.emit(self.kalman.estimate)
    self.value_change.emit(thresh_img)