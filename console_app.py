from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from util import *
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
from PyQt5.QtCore import pyqtSlot


#connect_type = QtCore.Qt.DirectConnection
connect_type = QtCore.Qt.AutoConnection


class ProcessClass( Ui_Form, QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)
        started = False
        self.config = json.load(open('config.json'))

        self.cam_w = self.config['camera']['x']  # 1280
        self.cam_h = self.config['camera']['y']  # 720
        self.cam_fps = self.config['camera']['fps']  # 720

        self.max_dist = self.config['distance']['max']
        self.min_dist = self.config['distance']['min']

        self.dilate_enable = self.config['dilate']['enable']
        self.dilate_kernel = self.config['dilate']['kernel']
        self.pix_border_h = 10
        self.pix_border_w = 10
        self.kernel = np.ones((self.dilate_kernel, self.dilate_kernel), 'uint8')
        self.work_zone = (self.config['work_zone'][0], self.config['work_zone'][1], self.config['work_zone'][2], self.config['work_zone'][3])
        self.stream_frame = np.zeros((480, 640))
        self.width_result = 0.0

    def start_measure(self):
        self.th_rand = None
        self.video_streamer_th = None
        self.server = None

        if self.th_rand is None:
            try:
                self.th_rand = ThreadProcessCurrentValue(params=self.config)
                self.th_rand.value_change.connect(self.draw_frame)
                self.th_rand.width_change.connect(self.change_width)
                self.th_rand.start()
                print("Camera connected")
            except Exception as e:
                print("Camera disconnected")
                print(e)

        try:
            if self.video_streamer_th is None:
                self.video_streamer_th = ThreadStreamsVideo(ip=self.config["video_stream_config"]["ip"],
                                                            port=int(self.config["video_stream_config"]["port"]))
                self.video_streamer_th.set_frame_get_method(self.get_stream_frame)
                self.video_streamer_th.start()
        except Exception as e:
            print("start stream error")
            print(e)

        try:
            if self.server is None:
                self.server = Server((self.config["latepanda_ip"], self.config["width_server_port"]), MyTCPHandler, self.get_width_res)
                self.server.start()
        except Exception as e:
            print(e)

    @pyqtSlot(object)
    def draw_frame(self, thresh_img):
        self.stream_frame = thresh_img


    def get_width_res(self):
        return self.width_result


    def get_stream_frame(self):
        return self.stream_frame


    def change_width(self, w):
        self.width_result = w



if __name__ == "__main__":
    import sys
    #app = QtWidgets.QApplication(sys.argv)
    # Form = QtWidgets.QWidget()
    import sys

    app = QtWidgets.QApplication(sys.argv)
    # Form = QtWidgets.QWidget()
    ui = ProcessClass()
    ui.start_measure()
    #ui.setupUi(ui)

    # ui.show()
    # ui.frame_label_position = ui.lb_frames.pos()
    # print(ui.frame_label_position)
    sys.exit(app.exec_())



