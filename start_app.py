from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from math import *
from form import Ui_Form
import pyrealsense2 as rs
import copy
import cv2
from threading import Thread, Event
import time
import numpy as np
import json



#connect_type = QtCore.Qt.DirectConnection
connect_type = QtCore.Qt.AutoConnection

class GUI(Ui_Form, QtWidgets.QWidget):
    started = False
    config = json.load(open('config.json'))

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

    def init_start_values(self):
        self.sb_kernel.setValue(self.dilate_kernel)
        if self.dilate_enable:
            self.cb_dilate.setChecked(True)
        else:
            self.cb_dilate.setChecked(False)

        if self.cam_w == 640:
            self.rb_640_480.setChecked(True)
        else:
            self.rb_1280_720.setChecked(True)
        self.sb_max_d.setValue(int(self.config['distance']['max'] * 1000))
        self.sb_min_d.setValue(int(self.config['distance']['min'] * 1000))
        self.th_rand = None
        self.pb_start.clicked.connect(self.start_b, type=connect_type)
        self.pb_stop.clicked.connect(self.stop_b, type=connect_type)
        self.rb_640_480.clicked.connect(self.change_resolution, type=connect_type)
        self.rb_1280_720.clicked.connect(self.change_resolution, type=connect_type)
        self.cb_dilate.stateChanged.connect(self.dilate_change, type=connect_type)
        self.sb_kernel.valueChanged.connect(self.kernel_change, type=connect_type)
        self.sb_max_d.valueChanged.connect(self.change_min_max_dist, type=connect_type)
        self.sb_min_d.valueChanged.connect(self.change_min_max_dist, type=connect_type)


    def change_min_max_dist(self):
        max_d = self.sb_max_d.value()/1000
        min_d = self.sb_min_d.value()/1000
        self.config['distance']['max'] = max_d
        self.config['distance']['min'] = min_d
        if self.th_rand is not None:
            self.th_rand.max_dist = max_d
            self.th_rand.min_dist = min_d



    def change_resolution(self):
        if self.th_rand is not None:
            self.th_rand.stop()
            self.th_rand = None
        if self.rb_640_480.isChecked():
            self.cam_w = 640
            self.cam_h = 480
            self.config['camera']['x']  = 640
            self.config['camera']['y']  = 480
            self.config['camera']['fps']  = 90
        else:
            self.cam_w = 1280
            self.cam_h = 720
            self.config['camera']['x'] = 1280
            self.config['camera']['y'] = 720
            self.config['camera']['fps'] = 30

    def kernel_change(self):
        val =self.sb_kernel.value()
        self.config['dilate']['kernel'] = val
        if self.th_rand is not None:
            self.th_rand.kernel = np.ones((val, val), 'uint8')




    def dilate_change(self):
        if self.cb_dilate.isChecked():
            dil_value = 1
        else:
            dil_value = 0
        self.config['dilate']['enable'] = dil_value
        if self.th_rand is not None:
            self.th_rand.dilate_enable = dil_value

    def start_b(self):
        if self.th_rand is None:
            self.th_rand = ThreadStreamsCurrentValue(params=self.config)
            self.th_rand.value_change.connect(self.draw_frame)
            self.th_rand.start()

    def stop_b(self):
        if self.th_rand is not None:
            self.th_rand.stop()
            self.th_rand = None

    def draw_frame(self, thresh_img):
        image = QtGui.QImage(thresh_img, thresh_img.shape[1], thresh_img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QPixmap(image)
        self.lb_frames.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.th_rand is not None:
            self.th_rand.stop()
        json.dump(self.config, open('config.json', 'w'))
        print("event")



class ThreadStreamsCurrentValue(Thread, QtCore.QObject):
    value_change = QtCore.pyqtSignal(object)
    def __init__(self, params, parent=None):
        Thread.__init__(self, parent)
        QtCore.QObject.__init__(self, parent)
        self._stop = Event()

        self.pipeline = rs.pipeline()
        # Configure streams
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, params['camera']['x'], params['camera']['y'], rs.format.z16, params['camera']['fps'])
        self.pipeline.start(self.config)
        self.pix_border_w = 10
        self.pix_border_h = 10
        self.cam_h = params['camera']['y']
        self.cam_w = params['camera']['x']
        self.max_dist = params['distance']['max']
        self.min_dist = params['distance']['min']
        self.dilate_enable = params['dilate']['enable']
        self.kernel_size = params['dilate']['kernel']
        self.kernel = np.ones((self.kernel_size, self.kernel_size), 'uint8')


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

                depth1 = np.asanyarray(depth.get_data()) / 1000.0  # distance in meters

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



                self.value_change.emit(thresh_img)
            except Exception as e:
                print(e)
                time.sleep(0.1)
                continue
            if self.stopped():
                return


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # Form = QtWidgets.QWidget()
    ui = GUI()
    ui.setupUi(ui)
    ui.init_start_values()
    ui.show()
    sys.exit(app.exec_())