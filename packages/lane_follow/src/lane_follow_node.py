#!/usr/bin/env python3

import rospy, random

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from dt_apriltags import Detector
from turbojpeg import TurboJPEG, TJPF_GRAY
from image_geometry import PinholeCameraModel
import cv2
import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import Header
from duckietown_msgs.msg import Twist2DStamped, LEDPattern

# Color masks
STOP_MASK = [(0, 75, 150), (5, 150, 255)]
ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
BLUE_RANGE = [(90, 50, 100), (110, 255, 200)]
BLACK_RANGE = [(0, 0, 0), (179, 75, 80)]

DEBUG = False
ENGLISH = False

class LaneFollowNode(DTROS):
  def __init__(self, node_name):
    super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
    self.node_name = node_name
    self.veh = rospy.get_param("~veh")

    # Subscribers
    self.sub = rospy.Subscriber(
      f"/{self.veh}/camera_node/image/compressed",
      CompressedImage,
      self.callback,
      queue_size=1,
      buff_size="20MB"
    )

    # Publishers
    self.pub = rospy.Publisher(
      f"/{self.veh}/output/image/mask/compressed",
      CompressedImage,
      queue_size=1
    )
    self.vel_pub = rospy.Publisher(
      f"/{self.veh}/car_cmd_switch_node/cmd",
      Twist2DStamped,
      queue_size=1
    )
    self.color_publisher = rospy.Publisher(f"/{self.veh}/led_emitter_node/led_pattern", LEDPattern, queue_size = 1)
    self.image_pub = rospy.Publisher("/output/detected_image", Image, queue_size=1)

    # Image processing helpers
    self.jpeg = TurboJPEG()
    self.br = CvBridge()

    # Lane-following PID Variables
    self.proportional = None
    if ENGLISH:
      self.offset = -220
    else:
      self.offset = 220
    self.velocity = 0.25
    self.twist = Twist2DStamped(v = self.velocity, omega=0)

    self.P = 0.020
    self.D = -0.007

    # override values if Celina's robot
    if self.veh == "csc22905":
      self.P = 0.049
      self.D = -0.004
      self.offset = 200
      self.velocity = 0.25
    elif self.veh == "csc22916":
      self.P = 0.049
      self.D = -0.004
      self.velocity = 0.25

    self.last_error = 0
    self.last_time = rospy.get_time()

    # Left turn variables
    self.left_turn_duration = 1.5
    self.right_turn_duration = 1
    self.straight_duration = 1
    self.started_action = None

    # Stop variables
    self.next_action = None
    self.stop = False
    self.last_stop_time = None
    self.stop_cooldown = 3
    self.stop_duration = 5
    self.stop_threshold_area = 5000 # minimun area of red to stop at
    self.stop_starttime = None
    
    # ====== April tag variables ======
    # Get static parameters    
    self.tag_size = 0.065
    self.rectify_alpha = 0.0

    # Initialize detector
    self.at_detector = Detector(
      searchpath = ['apriltags'],
      families = 'tag36h11',
      nthreads = 1,
      quad_decimate = 1.0,
      quad_sigma = 0.0,
      refine_edges = 1,
      decode_sharpening = 0.25,
      debug = 0
    )

    # Initialize static parameters from camera info message
    camera_info_msg = rospy.wait_for_message(f'/{self.veh}/camera_node/camera_info', CameraInfo)
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

    self.apriltag_actions = {
      "169": ["right", "left"],
      "162": ["right", "left"],
      "153": ["left", "straight"],
      "133": ["right", "straight"],
      "62": ["left", "straight"],
      "58": ["right", "straight"]
    }

    self.last_detected_apriltag = None

    # Apriltag timer
    self.publish_hz = 2
    self.timer = rospy.Timer(rospy.Duration(1 / self.publish_hz), self.cb_apriltag_timer)
    self.last_message = None

    # Initialize LED color-changing
    self.pattern = LEDPattern()
    self.pattern.header = Header()
    self.signalled = False

    # Shutdown hook
    rospy.on_shutdown(self.hook)

    self.loginfo("Initialized")

  def callback(self, msg):
    if not msg:
      return
    self.last_message = msg

    img = self.jpeg.decode(msg.data)
    crop = img[300:-1, :, :]
    crop_width = crop.shape[1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Mask for road lines
    roadMask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
    # crop = cv2.bitwise_and(crop, crop, mask=roadMask)
    contours, _ = cv2.findContours(
      roadMask,
      cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_NONE
    )

    # Search for lane in front
    max_area = 20
    max_idx = -1
    for i in range(len(contours)):
      area = cv2.contourArea(contours[i])
      if area > max_area:
        max_idx = i
        max_area = area

    if max_idx != -1:
      M = cv2.moments(contours[max_idx])
      try:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        self.proportional = cx - int(crop_width / 2) + self.offset
        if DEBUG:
          cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
          cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
      except:
        pass
    else:
      self.proportional = None
    
    # See if we need to look for stop lines
    if self.stop or (self.last_stop_time and rospy.get_time() - self.last_stop_time < self.stop_cooldown):
      if DEBUG:
        rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
        self.pub.publish(rect_img_msg)
      return
    
    # Mask for stop lines
    crop = img[400:-1, :, :]
    crop_width = crop.shape[1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    stopMask = cv2.inRange(hsv, STOP_MASK[0], STOP_MASK[1])
    # crop = cv2.bitwise_and(crop, crop, mask=stopMask)
    stopContours, _ = cv2.findContours(
      stopMask,
      cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_NONE
    )

    # Search for lane in front
    max_area = self.stop_threshold_area
    max_idx = -1
    for i in range(len(stopContours)):
      area = cv2.contourArea(stopContours[i])
      if area > max_area:
        max_idx = i
        max_area = area

    if max_idx != -1:
      M = cv2.moments(stopContours[max_idx])
      try:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        self.stop = True
        self.stop_starttime = rospy.get_time()
        if DEBUG:
          cv2.drawContours(crop, stopContours, max_idx, (0, 255, 0), 3)
          cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
      except:
        pass
    else:
      self.stop = False

    if DEBUG:
      rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
      self.pub.publish(rect_img_msg)

  def cb_apriltag_timer(self, _):
    '''
    Callback for timer
    '''
    msg = self.last_message
    if not msg:
      return

    self.last_detected_apriltag = None
    # turn image message into grayscale image
    img = self.jpeg.decode(msg.data, pixel_format=TJPF_GRAY)
    # run input image through the rectification map
    img = cv2.remap(img, self._mapx, self._mapy, cv2.INTER_NEAREST)

    # detect tags
    tags = self.at_detector.detect(img, True, self._camera_parameters, self.tag_size)

    if len(tags) == 0:
      return

    # Only save the april tag if it's within a close distance
    min_tag_distance = 2
    min_tag_idx = 0
    for tag in tags:
      distance = tag.pose_t[2][0]
      if distance > min_tag_distance:
        continue
    
    # save tag id if we're about to go to an intersection
    if str(tags[min_tag_idx]) in self.apriltag_actions:
      self.last_detected_apriltag = str(tag.tag_id)
    
    # TODO: if tag not within a range and offset, return (skip number detection)

    # color version of image message
    img = self.jpeg.decode(msg.data)
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

    max_area = 2000
    max_idx = -1
    for i in range(len(contours)):
      area = cv2.contourArea(contours[i])
      if area > max_area:
        max_idx = i
        max_area = area

    if max_idx == -1:
      return
    
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

    self.image_pub.publish(self.br.cv2_to_imgmsg(final_mask, "bgr8"))

  def drive(self):
    if self.stop:
      if rospy.get_time() - self.stop_starttime < self.stop_duration:
        # Stop
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        
        # Determine next action, if we haven't already
        if not self.next_action:
          # Get available action from last detected april tag
          if self.last_detected_apriltag and self.last_detected_apriltag in self.apriltag_actions:
            avail_actions = self.apriltag_actions[self.last_detected_apriltag]
            self.last_detected_apriltag = None
          else:
            avail_actions = [None]

          # If we detect a duckiebot and that turn is valid
          self.next_action = random.choice(avail_actions)
          
          # self.change_color(self.next_action)
      else:
        # Do next action
        if self.next_action == "left":
          # Go left
          if self.started_action == None:
            self.started_action = rospy.get_time()
          elif rospy.get_time() - self.started_action < self.left_turn_duration:
            self.twist.v = self.velocity
            self.twist.omega = 2.5
            self.vel_pub.publish(self.twist)
          else:
            self.started_action = None
            self.next_action = None
        elif self.next_action == "right":
          # Go right
          if self.started_action == None:
            self.started_action = rospy.get_time()
          elif rospy.get_time() - self.started_action < self.right_turn_duration:
            self.twist.v = self.velocity
            self.twist.omega = -2.5
            self.vel_pub.publish(self.twist)
          else:
            self.started_action = None
            self.next_action = None
        elif self.next_action == "straight":
          # Go straight
          if self.started_action == None:
            self.started_action = rospy.get_time()
          elif rospy.get_time() - self.started_action < self.straight_duration:
            self.twist.v = self.velocity
            self.twist.omega = 0
            self.vel_pub.publish(self.twist)
          else:
            self.started_action = None
            self.next_action = None
        else:
          self.stop = False
          self.last_stop_time = rospy.get_time()
    else:
      # Determine Velocity - based on if we're following a Duckiebot or not
      self.twist.v = self.velocity

      # Determine Omega - based on lane-following
      if self.proportional is None:
        self.twist.omega = 0
      else:
        # P Term
        P = -self.proportional * self.P

        # D Term
        d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
        self.last_error = self.proportional
        self.last_time = rospy.get_time()
        D = d_error * self.D

        self.twist.omega = P + D

        # Publish command
        if DEBUG:
          # self.loginfo(self.proportional, P, D, self.twist.omega, self.twist.v)
          print(self.proportional, P, D, self.twist.omega, self.twist.v)
      self.vel_pub.publish(self.twist)

  def hook(self):
    print("SHUTTING DOWN")
    self.twist.v = 0
    self.twist.omega = 0
    self.vel_pub.publish(self.twist)
    for i in range(8):
      self.vel_pub.publish(self.twist)

if __name__ == "__main__":
  node = LaneFollowNode("lanefollow_node")
  rate = rospy.Rate(8)  # 8hz
  while not rospy.is_shutdown():
    node.drive()
    rate.sleep()