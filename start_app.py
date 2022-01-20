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

def get_coords(x, y, angle, imwidth, imheight):

    x1_length = (x-imwidth) / math.cos(angle)
    y1_length = (y-imheight) / math.sin(angle)
    length = max(abs(x1_length), abs(y1_length))
    endx1 = x + length * math.cos(math.radians(angle))
    endy1 = y + length * math.sin(math.radians(angle))

    x2_length = (x-imwidth) / math.cos(angle+180)
    y2_length = (y-imheight) / math.sin(angle+180)
    length = max(abs(x2_length), abs(y2_length))
    endx2 = x + length * math.cos(math.radians(angle+180))
    endy2 = y + length * math.sin(math.radians(angle+180))

    return int(endx1), int(endy1), int(endx2), int(endy2)

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
        self.work_zone = (0,0,640,480)
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

        self.th_rand = None
        self.video_streamer_th = None
        self.pb_start.clicked.connect(self.start_b, type=connect_type)
        self.pb_stop.clicked.connect(self.stop_b, type=connect_type)

        self.pb_start_video_stream.clicked.connect(self.start_stream, type=connect_type)
        self.pb_stop_video_stream.clicked.connect(self.stop_stream, type=connect_type)

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

        self.server = Server(("localhost", 9999), MyTCPHandler, self.get_width_res)
        self.server.start()

    def get_width_res(self):
        return self.width_result


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
                self.video_streamer_th = ThreadStreamsVideo(ip=self.le_dest_stream_ip.text(), port=int(self.le_dest_stream_port.text()))
                self.video_streamer_th.set_frame_get_method(self.get_stream_frame)
                self.video_streamer_th.start()
        except Exception as e:
            print("start stream error")
            print(e)

    def stop_stream(self):
        if self.video_streamer_th is not None:
            self.video_streamer_th.stop()
            self.video_streamer_th = None


    def start_b(self):
        if self.th_rand is None:
            try:
                self.th_rand = ThreadStreamsCurrentValue(params=self.config)
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

        self.server.stop()
        print("event")



class ThreadStreamsCurrentValue(Thread, QtCore.QObject):
    value_change = QtCore.pyqtSignal(object)
    width_change = QtCore.pyqtSignal(float)
    def __init__(self, params, parent=None):
        Thread.__init__(self, parent)
        QtCore.QObject.__init__(self, parent)
        self._stop = Event()
        self.kalman = KalmanFilter(initial_estimate = 50.0, initial_est_error =1,initial_measure_error = 1)


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


import cv2
import socket
import struct
import math
import copy
class FrameSegment(object):
    """
    Object to break down image frame segment
    if the size of image exceed maximum datagram size
    """
    MAX_DGRAM = 2**16
    MAX_IMAGE_DGRAM = MAX_DGRAM - 64 # extract 64 bytes in case UDP frame overflown
    def __init__(self, sock, port, addr="192.168.0.101"):
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
                    # frame = cv2.resize(frame, (640, 480))  # resize the frame
                    # encoded, buffer = cv2.imencode('.jpg', frame)
                    # jpg_as_text = base64.b64encode(buffer)
                    self.fs.udp_frame(frame)


            except Exception as e:
                print(e)
                continue

from socketserver import TCPServer, BaseRequestHandler

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
        # self.request - это TCP - сокет, подключённый к клиенту
        #b'\x7e\xdd'
        self.data = self.request.recv(1024)
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        v = self.server.get_method()

        #v = rnd.get_random()
        print("value", v)
        self.request.sendall(bytearray(struct.pack("<f", v)))



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