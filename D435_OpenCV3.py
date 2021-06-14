## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import skimage.measure
import cv2
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math as m

H, W = 480, 640
size = 4
new_H, new_W = int(H/size) , int(W/size)
Map = np.zeros((1,1,3))
image_trigger = 0

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

list_alpha = np.linspace(-87/2,87/2,new_W+1)*m.pi/180
list_beta = np.linspace(-58/2,58/2,new_H+1)*m.pi/180
alpha, beta = np.meshgrid(list_alpha[:-1], list_beta[:-1])

Coord_sys = np.zeros((new_H,new_W,3))
x_Coord, y_Coord, z_Coord = np.zeros(new_H*new_W), np.zeros(new_H*new_W), np.zeros(new_H*new_W)

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
count = 0
try:
    while count <= 4:
        fig = plt.figure()
        count += 1

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # resized_depth = depth_image
        resized_depth = skimage.measure.block_reduce(depth_image, (size,size), np.max)
        distort_correction_depth = resized_depth/(np.cos(alpha)*np.cos(beta))
        distort_correction_depth[distort_correction_depth>2000] = 0 # 2m 시야

        x_Coord = distort_correction_depth*np.sin(alpha)/np.sqrt(1+np.tan(beta)*np.tan(beta)*np.cos(alpha)*np.cos(alpha))
        y_Coord = x_Coord*np.tan(beta)/np.tan(alpha)*(-1)
        z_Coord = x_Coord/np.tan(alpha)

        theta = m.pi/2*(count-2)
        rot_y = np.array([[m.cos(theta),0,m.sin(theta)],[0,1,0],[-m.sin(theta),0,m.cos(theta)]])
        x_Coord2 = x_Coord.reshape(new_H*new_W,1)
        y_Coord2 = y_Coord.reshape(new_H*new_W,1)
        z_Coord2 = z_Coord.reshape(new_H*new_W,1)
        Coord_sys2 = np.hstack([x_Coord2,y_Coord2,z_Coord2])
        Coord_sys2 = np.dot(Coord_sys2,rot_y)

        Coord_sys = np.stack((x_Coord,y_Coord,z_Coord),axis=2)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(resized_depth, alpha=0.03), cv2.COLORMAP_JET)
        images = color_image

        Coord_sys = Coord_sys.reshape(new_H*new_W,3)
        if count == 2:
            Map = Coord_sys2
            image_trigger = 1
        elif count > 2:
            Map = np.vstack([Map,Coord_sys2])
        print(Map.shape)
        
        if image_trigger == 1:
            Map = np.around(Map/10)*10
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)
            # 3D plot
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim3d(-1000,1000) # x_coord
            ax.set_ylim3d(-1000,1000) # z_coord
            ax.set_zlim3d(-500,500) # y_coord

            plt.xlabel('x')
            plt.ylabel('z')
            ax.plot(Map[:,0],Map[:,2],Map[:,1],'o',markersize=0.1)
            plt.show()

        # plt.xlabel('x')
        # plt.ylabel('z')
        # ax.plot(Coord_sys[:,0],Coord_sys[:,2],Coord_sys[:,1],'o',markersize=0.1)
        # plt.show()
    np.save('C:/Users/dlqja/Desktop/SLAM_Modules/3D_Domain1',Map)

finally:

    # Stop streaming
    pipeline.stop()