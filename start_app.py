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
    work_zone = (config['work_zone'][0], config['work_zone'][1], config['work_zone'][2], config['work_zone'][3])
    stream_frame = np.zeros((480, 640))
    width_result = 0.0

    def init_start_values(self):
        thresh_img = np.zeros((480,640 ))
        image = QtGui.QImage(thresh_img, thresh_img.shape[1], thresh_img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QPixmap(image)
        self.lb_frames.setPixmap(pixmap)
        self.stream_frame= np.zeros((480, 640))



        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        #self.work_zone = (0,0,640,480)
        self.origin = QPoint()
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
        self.sb_depth_dead_zone.setValue(int(self.config['death_width_percent_zone'] * 100))
        self.le_dest_stream_ip.setText(self.config["video_stream_config"]["ip"])
        self.le_dest_stream_port.setText(str(self.config["video_stream_config"]["port"]))
        self.sb_min_area_w_region.setValue(int(self.config['roi_area']["min"] * 100))
        self.sb_max_area_w_region.setValue(int(self.config['roi_area']["max"] * 100))
        self.le_latepanda_ip.setText(self.config["latepanda_ip"])
        self.le_width_request_port.setText(str(self.config["width_server_port"]))

        self.th_rand = None
        self.video_streamer_th = None
        self.server = None
        #START STOP BUTTONS##################################################
        self.pb_start.clicked.connect(self.start_b, type=connect_type)
        self.pb_stop.clicked.connect(self.stop_b, type=connect_type)
        self.pb_start_video_stream.clicked.connect(self.start_stream, type=connect_type)
        self.pb_stop_video_stream.clicked.connect(self.stop_stream, type=connect_type)
        self.pb_start_width_server.clicked.connect(self.start_width_server, type=connect_type)
        self.pb_stop_width_server.clicked.connect(self.stop_width_server, type=connect_type)
        self.pb_save.clicked.connect(self.save_params, type=connect_type)

        self.rb_640_480.clicked.connect(self.change_resolution, type=connect_type)
        #self.rb_1280_720.clicked.connect(self.change_resolution, type=connect_type)
        self.cb_dilate.stateChanged.connect(self.dilate_change, type=connect_type)
        self.sb_kernel.valueChanged.connect(self.kernel_change, type=connect_type)
        self.sb_max_d.valueChanged.connect(self.change_min_max_dist, type=connect_type)
        self.sb_min_d.valueChanged.connect(self.change_min_max_dist, type=connect_type)
        self.sb_depth_dead_zone.valueChanged.connect(self.change_depth_ignore_ratio, type=connect_type)

        self.sb_min_area_w_region.valueChanged.connect(self.change_min_max_area_of_roi_in_w_region, type=connect_type)
        self.sb_max_area_w_region.valueChanged.connect(self.change_min_max_area_of_roi_in_w_region, type=connect_type)

        self.frame_label_position = self.lb_frames.pos()



    def get_width_res(self):
        return self.width_result

    def save_params(self):
        self.config["latepanda_ip"] = self.le_latepanda_ip.text()
        self.config["width_server_port"] = int(self.le_width_request_port.text())
        self.config["video_stream_config"]["ip"] = self.le_dest_stream_ip.text()
        self.config["video_stream_config"]["port"] = int(self.le_dest_stream_port.text())
        json.dump(self.config, open('config.json', 'w'))

    def start_width_server(self):
        try:
            if self.server is None:
                self.server = Server((self.config["latepanda_ip"], self.config["width_server_port"]), MyTCPHandler, self.get_width_res)
                self.server.start()
                self.lb_width_server_status.setText("Enable")
        except Exception as e:
            print(e)

    def stop_width_server(self):
        if self.server is not None:
            self.server.stop()
            self.lb_width_server_status.setText("Disable")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.cb_work_region.isChecked():
            self.origin = QPoint(event.pos())
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()

    def mouseMoveEvent(self, event):
        if not self.origin.isNull() and self.cb_work_region.isChecked():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton and not self.origin.isNull() and self.cb_work_region.isChecked():
            print("start ",  self.origin, " end ",QPoint(event.pos()))
            xmin = self.frame_label_position.x()
            xmax = self.frame_label_position.x() + 640
            ymin = self.frame_label_position.y()
            ymax = self.frame_label_position.y() + 480
            if (self.origin.x() >= xmin and self.origin.x() <= xmax) and (event.pos().x() >= xmin and event.pos().x() <= xmax) and (self.origin.y() >= ymin and self.origin.y() <= ymax) and (event.pos().y() >= ymin and event.pos().y() <= ymax) :
                self.work_zone = (self.origin.x()-xmin, self.origin.y()-ymin, event.pos().x()-xmin, event.pos().y()-ymin)
                self.config['work_zone'] = [self.work_zone[0], self.work_zone[1], self.work_zone[2], self.work_zone[3]]
                if self.th_rand is not None:
                    self.th_rand.work_zone = self.work_zone
                    self.th_rand.work_zone_area = (self.work_zone[3] - self.work_zone[1]) * (
                                self.work_zone[2] - self.work_zone[0])
            self.rubberBand.hide()


    def change_min_max_dist(self):
        max_d = self.sb_max_d.value()/1000
        min_d = self.sb_min_d.value()/1000
        self.config['distance']['max'] = max_d
        self.config['distance']['min'] = min_d
        if self.th_rand is not None:
            self.th_rand.max_dist = max_d
            self.th_rand.min_dist = min_d

    def change_depth_ignore_ratio(self):
        ratio = self.sb_depth_dead_zone.value()/100
        self.config['death_width_percent_zone'] = ratio
        if self.th_rand is not None:
            self.th_rand.death_width_zone = ratio

    def change_min_max_area_of_roi_in_w_region(self):
        min_a = self.sb_min_area_w_region.value()/100
        max_a = self.sb_max_area_w_region.value()/100
        self.config['roi_area']["min"] = min_a
        self.config['roi_area']["max"] = max_a
        if self.th_rand is not None:
            self.th_rand.min_area_roi = min_a
            self.th_rand.max_area_roi = max_a





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


    def get_stream_frame(self):
        return self.stream_frame

    def dilate_change(self):
        if self.cb_dilate.isChecked():
            dil_value = 1
        else:
            dil_value = 0
        self.config['dilate']['enable'] = dil_value
        if self.th_rand is not None:
            self.th_rand.dilate_enable = dil_value

    def start_stream(self):
        try:
            if self.video_streamer_th is None:
                self.video_streamer_th = ThreadStreamsVideo(ip=self.config["video_stream_config"]["ip"], port=int(self.config["video_stream_config"]["port"]))
                self.video_streamer_th.set_frame_get_method(self.get_stream_frame)
                self.video_streamer_th.start()
                self.lb_stream_status.setText("Enable")
        except Exception as e:
            print("start stream error")
            print(e)

    def stop_stream(self):
        if self.video_streamer_th is not None:
            self.video_streamer_th.stop()
            self.video_streamer_th = None
            self.lb_stream_status.setText("Disable")


    def start_b(self):
        if self.th_rand is None:
            try:
                self.th_rand = ThreadProcessCurrentValue(params=self.config)
                self.th_rand.value_change.connect(self.draw_frame)
                self.th_rand.width_change.connect(self.change_width)
                self.th_rand.start()
                self.lb_status.setText("Connected")
            except Exception as e:
                self.lb_status.setText("Disconnected")
                print(e)

    def stop_b(self):
        try:
            if self.th_rand is not None:
                self.th_rand.stop()
                self.th_rand = None
                self.lb_status.setText("Disconnected")
        except Exception as e:
            self.lb_status.setText("Disconnected")
            print(e)

    def draw_frame(self, thresh_img):
        self.stream_frame = thresh_img
        image = QtGui.QImage(thresh_img, thresh_img.shape[1], thresh_img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QPixmap(image)
        self.lb_frames.setPixmap(pixmap)



    def change_width(self, w):
        self.width_result = w
        self.lb_width.setText(str(int(w)))

    def closeEvent(self, event):
        if self.th_rand is not None:
            self.th_rand.stop()
        json.dump(self.config, open('config.json', 'w'))
        if self.server is not None:
            self.server.stop()
        print("event")








if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # Form = QtWidgets.QWidget()
    ui = GUI()
    ui.setupUi(ui)
    ui.init_start_values()
    ui.show()
    ui.frame_label_position = ui.lb_frames.pos()
    print(ui.frame_label_position)
    sys.exit(app.exec_())