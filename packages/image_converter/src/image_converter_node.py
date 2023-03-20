#!/usr/bin/env python3

import rospy
import cv2
from duckietown.dtros import DTROS, NodeType
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
import numpy as np
from turbojpeg import TurboJPEG, TJPF_GRAY
from image_geometry import PinholeCameraModel
import os

class ImageConverterNode(DTROS):

    def __init__(self, node_name):
        super(ImageConverterNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name

        self.br = CvBridge()
        self.host = str(os.environ['VEHICLE_NAME'])
        self.jpeg = TurboJPEG()

        self.rectify_alpha = 0.0

        # self.host = "csc22919"
        self.image_sub = rospy.Subscriber(f"/{self.host}/camera_node/image/compressed", CompressedImage, self.image_callback)
        self.image_pub = rospy.Publisher("/output/detected_image/Image", Image, queue_size=1)

        # Initialize static parameters from camera info message
        camera_info_msg = rospy.wait_for_message(f'/{self.host}/camera_node/camera_info', CameraInfo)
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(camera_info_msg)
        H, W = camera_info_msg.height, camera_info_msg.width

        # find optimal rectified pinhole camera
        rect_K, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_model.K, self.camera_model.D, (W, H), self.rectify_alpha
        )

        # store new camera parameters
        self._camera_parameters = (rect_K[0, 0], rect_K[1, 1], rect_K[0, 2], rect_K[1, 2])

        self._mapx, self._mapy = cv2.initUndistortRectifyMap(
            self.camera_model.K, self.camera_model.D, None, rect_K, (W, H), cv2.CV_32FC1
        )

    def image_callback(self, msg):
        if not msg:
            return

        # turn image message into grayscale image
        # img = self.jpeg.decode(msg.data, pixel_format=TJPF_GRAY)
        img = self.jpeg.decode(msg.data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # run input image through the rectification map
        img = cv2.remap(img, self._mapx, self._mapy, cv2.INTER_NEAREST)

        self.image_pub.publish(self.br.cv2_to_imgmsg(img, encoding="rgb8"))

if __name__ == '__main__':
    node = ImageConverterNode("image_converter_node")
    rospy.spin()