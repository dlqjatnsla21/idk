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
size = 2
new_H, new_W = int(H/size) , int(W/size)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

list_alpha = np.linspace(-75/2,75/2,new_W+1)*m.pi/180
list_beta = np.linspace(-62/2,62/2,new_H+1)*m.pi/180
alpha, beta = np.meshgrid(list_alpha[:-1], list_beta[:-1])
alpha_beta = np.stack((alpha, beta), axis=2)

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
trigger = 0

try:
    while True:
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
        
        # x_Coord = resized_depth*np.sin(alpha)/np.sqrt(1+np.tan(beta)*np.tan(beta)*np.cos(alpha)*np.cos(alpha))
        x_Coord = distort_correction_depth*np.sin(alpha)/np.sqrt(1+np.tan(beta)*np.tan(beta)*np.cos(alpha)*np.cos(alpha))
        y_Coord = x_Coord*np.tan(beta)/np.tan(alpha)*(-1)
        z_Coord = x_Coord/np.tan(alpha)

        x_Coord = np.around(x_Coord/10)*10
        y_Coord = np.around(y_Coord/10)*10
        z_Coord = np.around(z_Coord/10)*10
        
        Coord_sys = np.stack((x_Coord,y_Coord,z_Coord),axis=2)

        if trigger == 0:
            First_map = Coord_sys
            trigger = 1

        Second_map = Coord_sys
        Distance_Vector = Second_map - First_map
        # print(Distance_Vector.shape)
        Distance_Vector = Distance_Vector[100:140,140:180,:]        
        # print(Distance_Vector.shape)
        np.nan_to_num(Distance_Vector, copy=False)
        Sum = Distance_Vector.sum(axis = 0)
        Sum = Sum.sum(axis = 0)/1600
        print('moving vector :',Sum)
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(resized_depth, alpha=0.03), cv2.COLORMAP_JET)
        images = color_image

        Coord_sys_for_plot = Coord_sys.reshape(new_H*new_W,3)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
        
        # 3D plot
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(-1000,1000) # x_coord
        ax.set_ylim3d(0,4000) # z_coord
        ax.set_zlim3d(-500,500) # y_coord

        plt.xlabel('x')
        plt.ylabel('z')
        ax.plot(Coord_sys_for_plot[:,0],Coord_sys_for_plot[:,2],Coord_sys_for_plot[:,1],'o',markersize=0.1)
        # plt.show()

        First_map = Coord_sys

        # # Depth_Image plot
        # cv2.imshow('RealSense', images)
        # cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()