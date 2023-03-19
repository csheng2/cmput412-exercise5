#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
import numpy as np
from turbojpeg import TurboJPEG, TJPF_GRAY
from image_geometry import PinholeCameraModel
import os

class Image_converter():

    def __init__(self):
        self.br = CvBridge()
        self.host = str(os.environ['VEHICLE_NAME'])
        self.jpeg = TurboJPEG()
        self.camera_model = PinholeCameraModel()


        # self.host = "csc22919"
        self.image_sub = rospy.Subscriber("/csc22919/camera_node/image/compressed", CompressedImage, self.image_callback)
        self.image_pub = rospy.Publisher("/output/detected_image/Image", Image, queue_size=1)
        camera_info_msg = rospy.wait_for_message(f'/'+ self.host +'/camera_node/camera_info', CameraInfo)

        # Initialize static parameters from camera info message
        self.camera_model.fromCameraInfo(camera_info_msg)
        H, W = camera_info_msg.height, camera_info_msg.width

        # find optimal rectified pinhole camera
        rect_K, _ = cv2.getOptimalNewCameraMatrix(
        self.camera_model.K, self.camera_model.D, (W, H), self.rectify_alpha)

        # store new camera parameters
        self._camera_parameters = (rect_K[0, 0], rect_K[1, 1], rect_K[0, 2], rect_K[1, 2])
        self._mapx, self._mapy = cv2.initUndistortRectifyMap(self.camera_model.K, self.camera_model.D, None, rect_K, (W, H), cv2.CV_32FC1)

    def image_callback(self, msg):
        # turn image message into grayscale image
        img = self.jpeg.decode(msg.data, pixel_format=TJPF_GRAY)
        self.publisher(img)
    
    def publisher(self, img):
        # run input image through the rectification map
        img = cv2.remap(img, self._mapx, self._mapy, cv2.INTER_NEAREST)
        if self.image is not None:
            self.image_pub.publish(self.br.cv2_to_imgmsg(img))




def main():
    
    rospy.init_node('image_converter_node')
    IC = Image_converter()    
    rospy.spin()


if __name__ == '__main__':
    main()
