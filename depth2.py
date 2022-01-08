import copy

import numpy as np
import pyrealsense2 as rs
import cv2
import pandas as pd


max_dist = 0.4
min_dist = 0.25
w_obj = 106
h_obj = 41
pix_border_w = 10
pix_border_h = 10
cam_w = 640#1280
cam_h = 480#720
area_screen = cam_w * cam_h

#1280*720
# fov_x_2 = np.tan(np.radians(87/2))
# fov_y_2 = np.tan(np.radians(58/2))

#640*480
fov_x_2 = np.tan(np.radians(78.5/2))
fov_y_2 = np.tan(np.radians(64/2))

img_w_2 = cam_w/2
img_h_2 = cam_h/2





# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()

# Configure streams
config = rs.config()
config.enable_stream(rs.stream.depth, cam_w, cam_h, rs.format.z16, 90)

# Start streaming
pipeline.start(config)
for_grapth =[]
while True:
    # This call waits until a new coherent set of frames is available on a device
    # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    if not depth: continue

    depth1 = np.asanyarray(depth.get_data()) / 1000.0 #distance in meters
    original_copy = copy.deepcopy(depth1)
    depth1[(depth1 > max_dist) | (depth1 < min_dist)] = 255
    # depth1[depth1 > max_dist] = 255
    # depth1[depth1 < min_dist] = 255
    # depth1[depth1 <= 0] = 255
    depth1[(depth1 <= max_dist) & (0 < depth1)] = 0



    #add border to max min
    border = np.zeros((cam_h, cam_w)) + 255
    border[pix_border_h:-pix_border_h, pix_border_w:-pix_border_w] = depth1[pix_border_h:-pix_border_h, pix_border_w:-pix_border_w]
    depth1 = border


    ret, thresh_img = cv2.threshold(depth1, 10, 255, 0)
    kernel = np.ones((3, 3), 'uint8')

    thresh_img = cv2.dilate(thresh_img, kernel, iterations=1)


    thresh_img = thresh_img.astype(np.uint8)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_sorted = sorted(contours, key=len, reverse=True)
    cnt = cnt_sorted[0]
    area = cv2.contourArea(cnt)
    image_copy = np.zeros((cam_h, cam_w))
    # if (area + (area * 0.2)) > area_screen:
    #     if len(cnt_sorted) > 1:
    #         cnt = sorted(contours, key=len, reverse=True)[1]

    rbox = cv2.minAreaRect(cnt)
    #print(rbox)
    pts = cv2.boxPoints(rbox).astype(np.int32)
    # print(pts)



    cv2.drawContours(image_copy, [pts], -1, 255, -1)

    w_i, h_i = abs(rbox[1][0]), abs(rbox[1][1])
    w = np.max((w_i, h_i))
    h = np.min((w_i, h_i))
    #print((w,h))





    cv2.imshow('box',image_copy)

    A = np.argwhere(image_copy >= 255)
    #depp = depth1[A]
    # B = np.argwhere(depth1==0)
    #
    # a = set((tuple(i) for i in A))
    # b = set((tuple(i) for i in B))
    #
    # c = a.intersection(b)
    #
    res = np.array([original_copy[v[0],v[1]] for v in A])
    res[res <= 0.1] = np.nan
    dist = np.nanmean(res)

    # pix_per_mm = w/(w_obj * dist)

    px_mm_x = (dist * 1000 * fov_x_2) / img_w_2
    px_mm_y = (dist * 1000 * fov_y_2) / img_h_2
    delta_x = pts[2][0] - pts[1][0]
    delta_y = pts[2][1] - pts[1][1]

    delta_x_mm = delta_x*px_mm_x
    delta_y_mm = delta_y*px_mm_y
    print(delta_x, delta_y)

    wwww = np.sqrt((delta_x_mm**2 + delta_y_mm**2))
    print(wwww)

    # print(dist, " pix mm  ", pix_per_mm / dist, " x ", px_mm_x * w, " y ", px_mm_y * h)
    # print("pix per mm x ",px_mm_x, " pix per mm y ", px_mm_y)
    print(dist, " x ", px_mm_x * w, " y ", px_mm_y * h)



    cv2.imshow('orginal',original_copy)
    # kernel = np.ones((5, 5), np.uint8)
    # numberOfIterations = 7
    # eroded_contours = cv2.erode(image_copy.copy(), kernel, iterations=int(numberOfIterations))
    # cv2.imshow('nier2', eroded_contours)
    # print(eroded_contours.shape)


    # img_contours = np.zeros((480, 640, 3))
    # # draw the contours on the empty image
    # cv2.drawContours(img_contours, contours, 3, (0, 255, 0), 3)
    #cv2.drawContours(depth1, contours, -1, (0, 255, 0), 1)
    #
    # for y in range(480):
    #     for x in range(640):
    #         dist = depth.get_distance(x, y)
    #         image[y,x] = dist
    #

    for p in range(len(pts)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (pts[p][0] , pts[p][1])
        fontScale = 1
        color = 0
        thickness = 3
        depth1 = cv2.putText(depth1, str(p), org, font,
                                 fontScale, color, thickness, cv2.LINE_AA)

    depth1  = cv2.drawContours(depth1, [pts], -1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Depth Stream threshold", depth1)
    key = cv2.waitKey(1)
    # if pressed escape exit program
    if key == ord("s"):
        pass

    if key == ord("a"):
        max_dist+=0.02
        print("distance=", max_dist)
    if key == ord("d"):
        max_dist-=0.02
        print("distance=", max_dist)
    if key == 27:
        cv2.destroyAllWindows()
        break



exit(0)
# except rs.error as e:
#    # Method calls agaisnt librealsense objects may throw exceptions of type pylibrs.error
#    print("pylibrs.error was thrown when calling %s(%s):\n", % (e.get_failed_function(), e.get_failed_args()))
#    print("    %s\n", e.what())
#    exit(1)
# except Exception as e:
#     print(e)
#     pass