import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt, numpy as np
import math as m
import keyboard

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

W = 640

alpha = np.linspace(-75/2,75/2,W)*m.pi/180

Coord_sys_2D = np.zeros((W,3))
x_Coord, z_Coord = np.zeros(W), np.zeros(W)

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
velocity_record = []
distance_record = []
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())
        
        depth_image_2D = depth_image[239,:]

        distort_correction_depth = depth_image_2D/(np.cos(alpha))
        # distort_correction_depth[distort_correction_depth>2000] = 0 # 1m 시야

        x_Coord = distort_correction_depth*np.sin(alpha)
        z_Coord = x_Coord/np.tan(alpha)

        x_Coord = np.around(x_Coord/10)*10
        z_Coord = np.around(z_Coord/10)*10

        Coord_sys_2D = np.stack((x_Coord,z_Coord),axis=1)

        if trigger == 0:
            trigger = 1
            First_Map = Coord_sys_2D[319,:]
        
        Second_Map = Coord_sys_2D[319,:]
        Distance_Vector = First_Map - Second_Map
        velocity_record.append(Distance_Vector[1])
        distance_record.append(Coord_sys_2D[319,1])
        print(Distance_Vector)

        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # images = depth_colormap

        # fig, ax = plt.subplots()
        # ax.set_xlim(-500,500) # x_coord
        # ax.set_ylim(0,1000) # z_coord
        # ax.scatter(Coord_sys_2D[:,0], Coord_sys_2D[:,1], s=1, alpha=0.5)
        # ax.grid(True)cc
        # plt.show()c
        First_Map = Coord_sys_2D[319,:]
        if keyboard.is_pressed("c") : #Shutdown
            break
    plt.plot(velocity_record)
    plt.show()
finally:

    # Stop streaming
    pipeline.stop()