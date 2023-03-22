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

BLUE_RANGE = [(90, 50, 100), (110, 255, 200)]
BLACK_RANGE = [(0, 0, 0), (179, 75, 80)]

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
        self.image_pub = rospy.Publisher("/output/detected_image", Image, queue_size=1)

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
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # run input image through the rectification map
        img = cv2.remap(img, self._mapx, self._mapy, cv2.INTER_NEAREST)

        frame = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mask for numbers
        mask = cv2.inRange(hsv, BLUE_RANGE[0], BLUE_RANGE[1])
        contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
        )

        max_area = 5000
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx == -1:
            return
        
        # print(max_area)
        
        [X, Y, W, H] = cv2.boundingRect(contours[max_idx])
        cropped_image = frame[Y:Y+H, X:X+W]
        second_mask = cv2.inRange(cropped_image, BLACK_RANGE[0], BLACK_RANGE[1])

        contours, _ = cv2.findContours(
            second_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        max_area = 0
        max_idx = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        mask = np.zeros(cropped_image.shape, np.uint8)
        cv2.drawContours(mask, contours, max_idx, (255, 255, 255), thickness=cv2.FILLED)
        final_mask = cv2.bitwise_and(mask, mask, mask=second_mask)
        # print(second_mask.shape, img.shape)

        # print(self.jpeg.encode(second_mask))
        # print(self.br.cv2_to_imgmsg(second_mask))

        # imgMsg = CompressedImage(format="jpeg", data=self.jpeg.encode(second_mask))
        self.image_pub.publish(self.br.cv2_to_imgmsg(final_mask, "bgr8"))

if __name__ == '__main__':
    node = ImageConverterNode("image_converter_node")
    rospy.spin()