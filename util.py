from PyQt5 import QtCore, QtGui, QtWidgets
import pyrealsense2 as rs
from random import randint, random
from socketserver import TCPServer, BaseRequestHandler

from threading import Thread, Event
import time
import numpy as np
import json

import cv2
import socket
import struct
import math
import copy


class ThreadProcessCurrentValue(Thread, QtCore.QObject):
    value_change = QtCore.pyqtSignal(object)
    width_change = QtCore.pyqtSignal(float)
    def __init__(self, params, parent=None):
        Thread.__init__(self, parent)
        QtCore.QObject.__init__(self, parent)
        self._stop = Event()
        self.kalman = KalmanFilter(initial_estimate = 50.0, initial_est_error =5,initial_measure_error = 3)


        self.pipeline = rs.pipeline()
        # Configure streams
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, params['camera']['x'], params['camera']['y'], rs.format.z16, params['camera']['fps'])
        self.pipeline.start(self.config)
        self.pix_border_w = 10
        self.pix_border_h = 10
        self.death_width_zone =  params["death_width_percent_zone"]
        self.min_area_roi = params["roi_area"]["min"]
        self.max_area_roi = params["roi_area"]["max"]
        self.work_zone = (params['work_zone'][0], params['work_zone'][1], params['work_zone'][2], params['work_zone'][3])
        self.work_zone_area = (params['work_zone'][3] - params['work_zone'][1]) * (params['work_zone'][2] - params['work_zone'][0])
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
            self.fov_x_2 = np.tan(np.radians(87/ 2))
            self.fov_y_2 = np.tan(np.radians(658/ 2))
        self.img_w_2 = self.cam_w / 2
        self.img_h_2 = self.cam_h / 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.res_W = 0

        self.fontScale = 1
        self.color = 0
        self.thickness = 3


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
                if self.area_screen * 0.95 < area:
                    if len(cnt_sorted) > 1:
                        cnt = sorted(contours, key=len, reverse=True)[1]
                    else:
                        cnt = None
                if cnt is not None:

                    rbox = cv2.minAreaRect(cnt)
                    box_area = rbox[1][0] * rbox[1][1]
                    if self.min_area_roi * self.work_zone_area <=box_area <= self.max_area_roi * self.work_zone_area:

                        pts = cv2.boxPoints(rbox).astype(np.int32)
                        cv2.drawContours(image_copy, [pts], -1, 255, -1)
                        #print("centre ", rbox[0] ," w,h ", rbox[1], " angle ", rbox[2])

                        if rbox[1][0] >= rbox[1][1]:
                            ln_w = int(rbox[1][1] * self.death_width_zone)

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
                            ln_w = int(rbox[1][0] * self.death_width_zone)
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
                        self.res_W = np.sqrt((delta_x_mm ** 2 + delta_y_mm ** 2))
                        #print(res_W)
                        self.kalman.iterative_updates(self.res_W)







                        # print("pix per mm x ",px_mm_x, " pix per mm y ", px_mm_y)
                        #print(dist, " x ", px_mm_x * w, " y ", px_mm_y * h)


                        # font = cv2.FONT_HERSHEY_SIMPLEX
                        # org = (int(rbox[0][0]), int(rbox[0][1]))
                        # fontScale = 1
                        # color = 255
                        # thickness = 3
                        # thresh_img = cv2.putText(thresh_img, "C", org, font,
                        #                          fontScale, color, thickness, cv2.LINE_AA)
                        #Draw points
                        # for p in range(len(pts)):
                        #     font = cv2.FONT_HERSHEY_SIMPLEX
                        #     org = (pts[p][0], pts[p][1])
                        #     fontScale = 1
                        #     color = 0
                        #     thickness = 3
                        #     thresh_img = cv2.putText(thresh_img, str(p), org, font,
                        #                          fontScale, color, thickness, cv2.LINE_AA)

                        thresh_img = cv2.drawContours(thresh_img, [pts], -1, (0, 255, 0), 1, cv2.LINE_AA)

                work_r_points = [np.array(([self.work_zone[0], self.work_zone[1]],
                          [self.work_zone[2], self.work_zone[1]],
                          [self.work_zone[2], self.work_zone[3]],
                          [self.work_zone[0], self.work_zone[3]]
                          ))]
                thresh_img = cv2.drawContours(thresh_img, work_r_points,
                                              -1, (0, 255, 0), 1, cv2.LINE_AA)
                width = self.kalman.estimate
                thresh_img = cv2.putText(thresh_img, str(np.round(width, 2)), (30,30), self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)

                thresh_img = cv2.putText(thresh_img, "Real: " + str(np.round(self.res_W, 2)), (30, 70), self.font, self.fontScale,
                                         self.color, self.thickness, cv2.LINE_AA)

                #print("set frame", thresh_img.shape)

                #print("width ", self.kalman.estimate)
                self.width_change.emit(width)
                self.value_change.emit(thresh_img)
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            if self.stopped():
                return


class FrameSegment(object):
    """
    Object to break down image frame segment
    if the size of image exceed maximum datagram size
    """
    MAX_DGRAM = 2**16
    MAX_IMAGE_DGRAM = MAX_DGRAM - 64 # extract 64 bytes in case UDP frame overflown
    def __init__(self, sock, port, addr=""):
        self.s = sock
        self.port = port
        self.addr = addr

    def udp_frame(self, img):
        """
        Compress image and Break down
        into data segments
        """

        compress_img = cv2.imencode('.jpg', img)[1]
        dat = compress_img.tobytes()
        size = len(dat)
        count = math.ceil(size/(self.MAX_IMAGE_DGRAM))
        array_pos_start = 0
        while count:
            array_pos_end = min(size, array_pos_start + self.MAX_IMAGE_DGRAM)
            self.s.sendto(struct.pack("B", count) +
                dat[array_pos_start:array_pos_end],
                (self.addr, self.port)
                )
            array_pos_start = array_pos_end
            count -= 1


class ThreadStreamsVideo(Thread, QtCore.QObject):
    value_change = QtCore.pyqtSignal(object)
    width_change = QtCore.pyqtSignal(float)
    def __init__(self, ip="", port=5555, parent=None):
        Thread.__init__(self, parent)
        QtCore.QObject.__init__(self, parent)
        self._stop = Event()
        self.get_frame = None

        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(ip, port)

        self.fs = FrameSegment(self.s, addr=ip, port=port)

    def set_frame_get_method(self, method):
        self.get_frame = method



    def stop(self):
        self.s.close()
        self._stop.set()
        #footage_socket.disconnect('tcp://172.16.234.76:5555')



    def stopped(self):
        return self._stop.isSet()

    def run(self):
        while True:
            if self.stopped():
                return
            try:
                if self.get_frame is not None:
                    frame = self.get_frame()
                    self.fs.udp_frame(frame)


            except Exception as e:
                print(e)
                continue




class Server(TCPServer, Thread):
    def __init__(self, server_address, handler, method, parent=None):
        TCPServer.__init__(self, server_address, handler, bind_and_activate=True)
        Thread.__init__(self, parent)
        self._stop = Event()
        self.get_method = method

    def stop(self):
        self._stop.set()
        self.shutdown()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        self.serve_forever()

class MyTCPHandler(BaseRequestHandler):
    def handle(self):
        # self.request - ?????? TCP - ??????????, ???????????????????????? ?? ??????????????
        #b'\x7e\xdd'
        self.data = self.request.recv(1024)
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        v = self.server.get_method()

        #v = rnd.get_random()
        print("value", v)
        self.request.sendall(bytearray(struct.pack("<f", v)))



class KalmanFilter(object):
    def __init__(
        self,
        initial_estimate: float = random(),
        initial_est_error: float = random(),
        initial_measure_error: float = random(),
    ):
        self.estimate = initial_estimate
        self.gain = random()
        self.est_error = initial_est_error
        self.measure_error = initial_measure_error
        self.sensor_values = []


    def calculate_kalman_gain(self) -> None:
        """calculates Kalman gain given error values"""
        self.gain = self.est_error / (self.est_error + self.measure_error)

    def update_estimate(self, sensor_value: int = 0.0) -> None:
        """updates estimate based on Kalman gain"""
        new_estimate = self.estimate + self.gain * (sensor_value - self.estimate)
        self.estimate = new_estimate

    def calculate_estimate_error(self) -> None:
        """calculates error of the updated estimate"""
        self.est_error = (1 - self.gain) * self.est_error

    def iterative_updates(self, v) -> None:
        e = []
        self.calculate_kalman_gain()
        self.update_estimate(sensor_value=v)
        self.calculate_estimate_error()


