#####################################################
##               Read bag from file                ##
#####################################################


# First import library
# import pyrealsense2 as rs
import pyrealsense2.pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()

max_dist = 0.23
try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 15)

    # Start streaming from file
    pipeline.start(config)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

    # Create colorizer object
    colorizer = rs.colorizer()
    arr = 640 *480
    # Streaming loop
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        #print(depth_frame.get_distance(100, 100))
        data = np.asanyarray(depth_frame.get_data())
        print(data.shape)


        # # Colorize depth frame to jet colormap
        # depth_color_frame = colorizer.colorize(depth_frame)
        #
        # # Convert depth_frame to numpy array to render image in opencv
        # depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # depth1 = data / 1000.0
        # depth1[depth1 > max_dist] = 255
        # depth1[depth1 <= 0] = 255
        # depth1[(depth1 <= max_dist) & (0 < depth1)] = 0

        depth1 = data / 1000.0

        depth1[depth1 > max_dist] = 255
        depth1[depth1 <= 0] = 255
        depth1[(depth1 <= max_dist) & (0 < depth1)] = 0

        kernel = np.ones((6, 6), 'uint8')

        depth1 = cv2.erode(depth1, kernel, cv2.BORDER_REFLECT, iterations=1)

        depth1 = cv2.copyMakeBorder(depth1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, 255)
        # depth1depth1.astype(int)
        ret, thresh_img = cv2.threshold(depth1, 10, 255, 0)
        thresh_img = thresh_img.astype(np.uint8)
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_sorted = sorted(contours, key=len, reverse=True)
        cnt = cnt_sorted[0]
        frame_copy = np.zeros((500,660,3))
        frame_copy[:,:,0] = thresh_img
        area = cv2.contourArea(cnt)
        if (area + (area * 0.2)) > arr:
            if len(cnt_sorted) > 1:
                cnt = sorted(contours, key=len, reverse=True)[1]
                # rbox = cv2.minAreaRect(cnt)
                # pts = cv2.boxPoints(rbox).astype(np.int32)
                epsilon = 0.1 * cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, epsilon, True)
                frame_copy = cv2.drawContours(frame_copy, [cnt], -1, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            # rbox = cv2.minAreaRect(cnt)
            # pts = cv2.boxPoints(rbox).astype(np.int32)

            epsilon = 0.001 * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)
            frame_copy = cv2.drawContours(frame_copy, [cnt], -1, (0, 255, 0), 1, cv2.LINE_AA)



        # Render image in opencv window
        cv2.imshow("Depth Stream", frame_copy)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == ord("a"):
            max_dist+=0.01
        if key == ord("d"):
            max_dist-=0.01
        if key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pass