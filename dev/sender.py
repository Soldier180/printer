#!/usr/bin/env python

from __future__ import division
import cv2
import numpy as np
import socket
import struct
import math
import copy

import numpy as np
import pyrealsense2 as rs
import cv2


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
max_dist = 0.4
min_dist = 0.25

def main():
    """ Top level main function """
    # Set up UDP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 12345

    fs = FrameSegment(s, port)

    pipeline = rs.pipeline()

    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)

    # Start streaming
    pipeline.start(config)

    #cap = cv2.VideoCapture(0)
    while True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        depth1 = np.asanyarray(depth.get_data()) / 1000.0  # distance in meters
        original_copy = copy.deepcopy(depth1)
        depth1[(depth1 > max_dist) | (depth1 < min_dist)] = 255
        # depth1[depth1 > max_dist] = 255
        # depth1[depth1 < min_dist] = 255
        # depth1[depth1 <= 0] = 255
        depth1[(depth1 <= max_dist) & (0 < depth1)] = 0
        border = np.zeros((480, 640)) + 255
        border[10:-10, 10:-10] = depth1[10:-10, 10:-10]
        depth1 = border

        ret, thresh_img = cv2.threshold(depth1, 10, 255, 0)
        kernel = np.ones((3, 3), 'uint8')

        thresh_img = cv2.dilate(thresh_img, kernel, iterations=1)

        thresh_img = thresh_img.astype(np.uint8)
        # contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnt_sorted = sorted(contours, key=len, reverse=True)
        # cnt = cnt_sorted[0]
        # area = cv2.contourArea(cnt)
        # image_copy = np.zeros((480, 640))
        # # if (area + (area * 0.2)) > area_screen:
        # #     if len(cnt_sorted) > 1:
        # #         cnt = sorted(contours, key=len, reverse=True)[1]
        #
        # rbox = cv2.minAreaRect(cnt)
        # # print(rbox)
        # pts = cv2.boxPoints(rbox).astype(np.int32)
        # # print(pts)
        #
        # cv2.drawContours(image_copy, [pts], -1, 255, -1)
        #
        #
        # for p in range(len(pts)):
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     org = (pts[p][0], pts[p][1])
        #     fontScale = 1
        #     color = 0
        #     thickness = 3
        #     depth1 = cv2.putText(depth1, str(p), org, font,
        #                          fontScale, color, thickness, cv2.LINE_AA)
        #
        # depth1 = cv2.drawContours(depth1, [pts], -1, 255, 1, cv2.LINE_AA)
        #_, frame = cap.read()
        fs.udp_frame(thresh_img)
    #cap.release()
    cv2.destroyAllWindows()
    s.close()

if __name__ == "__main__":
    main()