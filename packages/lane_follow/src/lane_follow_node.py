#!/usr/bin/env python3

import rospy, random, cv2
import numpy as np

from duckietown.dtros import DTROS, NodeType
from dt_apriltags import Detector
from turbojpeg import TurboJPEG, TJPF_GRAY
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge

from tf import transformations as tr
from tf2_ros import StaticTransformBroadcaster

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Header, ColorRGBA
from duckietown_msgs.msg import Twist2DStamped, LEDPattern

from lane_follow.srv import MLPPredict

# Color masks
STOP_MASK = [(0, 75, 150), (5, 150, 255)]
ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
BLUE_RANGE = [(90, 50, 100), (110, 255, 200)]
BLACK_RANGE = [(0, 0, 0), (179, 75, 80)]

DEBUG = False
ENGLISH = False

"""
Much of the code for the Lane Following node is taken from
the lane_follow package by Justin Francis on eClass
Link: https://eclass.srv.ualberta.ca/mod/resource/view.php?id=6952069
"""
class LaneFollowNode(DTROS):
  def __init__(self, node_name):
    super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
    self.node_name = node_name
    self.veh = rospy.get_param("~veh")
    self.initialized = False

    # Shutdown hook
    rospy.on_shutdown(self.hook)

    # Image processing helpers
    self.jpeg = TurboJPEG()
    self.br = CvBridge()

    # Create server
    rospy.wait_for_service('mlp_predict_server')
    self.mlp_predict = rospy.ServiceProxy('mlp_predict_server', MLPPredict)
    self.predicting = False

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
    self.detection_pub = rospy.Publisher(
      f"/{self.veh}/detection/image/",
      Image,
      queue_size=1
    )
    self.led_publisher = rospy.Publisher(
      f"/{self.veh}/led_emitter_node/led_pattern",
      LEDPattern,
      queue_size=1
    )

    # Lane-following PID Constants
    self.proportional = None
    if ENGLISH:
      self.offset = -220
    else:
      self.offset = 220
    self.velocity = 0.2
    self.twist = Twist2DStamped(v = self.velocity, omega=0)

    self.P = 0.020
    self.D = -0.007

    # Other robot constants (based on experimentation)
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

    # Action variables
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
    self.stop_threshold_area = 5000 # minimum area of red to stop at
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

    # Directions available for every april tag
    self.apriltag_actions = {
      "169": ["right", "left"],
      "162": ["right", "left"],
      "153": ["left", "straight"],
      "133": ["right", "straight"],
      "62": ["left", "straight"],
      "58": ["right", "straight"]
    }

    #  Store last detected apriltag for 
    self.last_detected_apriltag = None

    # Apriltag locations
    self.apriltag_locations = {
      "200": {"x": 0.17, "y": 0.17, "z": 0.075, "yaw": 2.35619, "pitch": 0, "roll": -1.5708},
      "201": {"x": 1.65, "y": 0.17, "z": 0.075, "yaw": -2.35619, "pitch": 0, "roll": -1.5708},
      "94": {"x": 1.65, "y": 2.84, "z": 0.075, "yaw": -0.785398, "pitch": 0, "roll": -1.5708},
      "93": {"x": 0.17, "y": 2.84, "z": 0.075, "yaw": 0.785398, "pitch": 0, "roll": -1.5708},
      "153": {"x": 1.75, "y": 1.252, "z": 0.075, "yaw": 0, "pitch": 0, "roll": -1.5708},
      "133": {"x": 1.253, "y": 1.755, "z": 0.075, "yaw": 3.14159, "pitch": 0, "roll": -1.5708},
      "58": {"x": 0.574, "y": 1.259, "z": 0.075, "yaw": 0, "pitch": 0, "roll": -1.5708},
      "62": {"x": 0.075, "y": 1.755, "z": 0.075, "yaw": 3.14159, "pitch": 0, "roll": -1.5708},
      "169": {"x": 0.574, "y": 1.755, "z": 0.075, "yaw": 1.5708, "pitch": 0, "roll": -1.5708},
      "162": {"x": 1.253, "y": 1.253, "z": 0.075, "yaw": -1.5708, "pitch": 0, "roll": -1.5708},
    }

    # Number to apriltag map -- add number detections here as we detect them
    self.number_apriltag_map = {}

    # Apriltag detection timer
    self.apriltag_hz = 2
    self.last_message = None
    self.timer = rospy.Timer(rospy.Duration(1 / self.apriltag_hz), self.cb_apriltag_timer)

    # Initialize LED color-changing
    self.pattern = LEDPattern()
    self.pattern.header = Header()
    self.initalize_white_leds()

    # Transform broadcaster
    self.broadcaster = StaticTransformBroadcaster()

    self.initialized = True
    self.loginfo("Initialized")

  def callback(self, msg):
    self.last_message = msg
    # Don't detect we don't have a message or if we're predicting a number or node is not done initializing
    if not msg or self.predicting or not self.initialized:
      return

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
    # Don't detect we don't have a message or if we're predicting a number or node is not done initializing
    if not msg or self.predicting or not self.initialized:
      return

    self.last_detected_apriltag = None
    # turn image message into grayscale image
    img = self.jpeg.decode(msg.data)
    # run input image through the rectification map
    img = cv2.remap(img, self._mapx, self._mapy, cv2.INTER_NEAREST)
    bw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect tags
    tags = self.at_detector.detect(bw_image, True, self._camera_parameters, self.tag_size)

    if len(tags) == 0:
      # Publish image before returning
      self.detection_pub.publish(self.br.cv2_to_imgmsg(img, "bgr8"))
      return

    # Only save the april tag if it's within a close distance
    min_tag_distance = 1
    min_tag_idx = -1
    for i in range(len(tags)):
      distance = tags[i].pose_t[2][0]
      # If it's closer than that distance, update
      if distance < min_tag_distance:
        min_tag_idx = i

    # Return if we can't find a close enough april tag
    if min_tag_idx == -1:
      # Publish image before returning
      self.detection_pub.publish(self.br.cv2_to_imgmsg(img, "bgr8"))
      return

    closest_tag_id = str(tags[min_tag_idx].tag_id)

    # Save tag id if we're about to go to an intersection
    if closest_tag_id in self.apriltag_actions:
      self.last_detected_apriltag = closest_tag_id

    # Skip detection if we've already detected that number
    if closest_tag_id in self.number_apriltag_map.values():
      # Publish image before returning
      self.detection_pub.publish(self.br.cv2_to_imgmsg(img, "bgr8"))
      return

    # Stop moving
    self.predicting = True
    self.twist.v = 0
    self.twist.omega = 0
    self.vel_pub.publish(self.twist)

    # Convert image to HSV
    frame = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask for blue regions
    mask = cv2.inRange(hsv, BLUE_RANGE[0], BLUE_RANGE[1])
    blue_contours, _ = cv2.findContours(
      mask,
      cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_NONE
    )

    # Get the highest blue area in the same apriltag horizontal bounding area
    max_area = 2000
    max_blue_idx = -1

    # Detection region wrt to detected apriltag
    [TL, TR, _, _] = tags[min_tag_idx].corners

    for i in range(len(blue_contours)):
      area = cv2.contourArea(blue_contours[i])
      [X, Y, W, H] = cv2.boundingRect(blue_contours[i])
      # Top left's x within a 20 pixel error of contour top left
      TL_Good = abs(TL[0] - X) < 20
      # Top right's x within a 20 pixel error of contour top right
      TR_Good = abs(TR[0] - (X + W)) < 20
      # Only update if the area is greater than 200 pixels squared and within bounding box
      if area > max_area and TL_Good and TR_Good:
        max_blue_idx = i
        max_area = area

    if max_blue_idx == -1:
      self.predicting = False
      # Publish image before returning
      self.detection_pub.publish(self.br.cv2_to_imgmsg(img, "bgr8"))
      return
    
    # Get number (black writing) within the blue region
    [X, Y, W, H] = cv2.boundingRect(blue_contours[max_blue_idx])
    cropped_image = frame[Y:Y+H, X:X+W]
    second_mask = cv2.inRange(cropped_image, BLACK_RANGE[0], BLACK_RANGE[1])

    contours, _ = cv2.findContours(
      second_mask,
      cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_NONE
    )

    # Find largest black writing area (removes noise)
    max_area = 0
    max_idx = 0
    for i in range(len(contours)):
      area = cv2.contourArea(contours[i])
      if area > max_area:
        max_idx = i
        max_area = area

    # Get a mask of the black writing (results in black and white image)
    mask = np.zeros(second_mask.shape, np.uint8)
    cv2.drawContours(mask, contours, max_idx, (255, 255, 255), thickness=cv2.FILLED)
    final_mask = cv2.bitwise_and(mask, mask, mask=second_mask)

    # Send number for prediction
    msg = self.br.cv2_to_imgmsg(final_mask, "mono8")
    prediction = self.mlp_predict(msg)
    self.predicting = False

    if prediction.number < 0:
      # Invalid prediction
      print('Unable to predict number')
      # Publish image before returning
      self.detection_pub.publish(self.br.cv2_to_imgmsg(img, "bgr8"))
      return 
    
    # Print tag and location
    print(
      'Predicted number', prediction.number,
      'under Apriltag id', closest_tag_id,
      f"at location (x = {self.apriltag_locations[closest_tag_id]['x']}, y = {self.apriltag_locations[closest_tag_id]['y']})"
    )

    # Add to apriltag number map
    self.number_apriltag_map[str(prediction.number)] = closest_tag_id
    
    # Publish outline of detection to image publisher
    cv2.drawContours(img, blue_contours, max_blue_idx, (0, 255, 0), 3)
    self.detection_pub.publish(self.br.cv2_to_imgmsg(img, "bgr8"))

    # Publish transform
    self.publish_transform(prediction.number, closest_tag_id)

    # Properly terminate the program if we've found all numbers
    if len(node.number_apriltag_map) == 10:
      # Send an empty image to the service to signal that we're shutting down
      msg.data = bytes()
      try:
        self.mlp_predict(msg)
      except:
        # Node shutdown before sending a response
        pass

      # Shutdown current node
      rospy.signal_shutdown("Found all ten numbers!")
  
  def publish_transform(self, number, tag_id):
    # Initialize transform under its april tag's frame
    static_transform = TransformStamped()
    static_transform.header.stamp = rospy.Time.now()
    static_transform.header.frame_id = "world"
    static_transform.child_frame_id = str(number)

    # Offset and angle are 0
    static_transform.transform.translation.x = self.apriltag_locations[tag_id]['x']
    static_transform.transform.translation.y = self.apriltag_locations[tag_id]['y']
    static_transform.transform.translation.z = self.apriltag_locations[tag_id]['z']
    quat = tr.quaternion_from_euler(
      self.apriltag_locations[tag_id]['yaw'],
      self.apriltag_locations[tag_id]['pitch'],
      self.apriltag_locations[tag_id]['roll']
    )
    static_transform.transform.rotation.x = quat[0]
    static_transform.transform.rotation.y = quat[1]
    static_transform.transform.rotation.z = quat[2]
    static_transform.transform.rotation.w = quat[3]

    # Publish transform
    self.broadcaster.sendTransform(static_transform)

  def drive(self):
    # Don't move if we're predicting or node is not done initializing
    if self.predicting or not self.initialized:
      return
    
    # If we're stopped at an intersection
    if self.stop:
      if rospy.get_time() - self.stop_starttime < self.stop_duration:
        # Stop
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        
        # Determine next action, if we haven't already
        if not self.next_action:
          # Get available actions from last detected april tag
          if self.last_detected_apriltag and self.last_detected_apriltag in self.apriltag_actions:
            avail_actions = self.apriltag_actions[self.last_detected_apriltag]
            self.last_detected_apriltag = None
          else:
            avail_actions = [None]

          # Pick a random direction (random walk)
          self.next_action = random.choice(avail_actions)
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
          # No actions - continue lane-following
          self.stop = False
          self.last_stop_time = rospy.get_time()
    # If we're lane following
    else:
      # Set velocity
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
          self.loginfo(f'{self.proportional}, {P}, {D}, {self.twist.omega}, {self.twist.v}')
      self.vel_pub.publish(self.twist)

  def initalize_white_leds(self):
    '''
    Code for this function was inspired by 
    "duckietown/dt-core", file "led_emitter_node.py"
    Link: https://github.com/duckietown/dt-core/blob/daffy/packages/led_emitter/src/led_emitter_node.py
    Author: GitHub user liampaull
    '''
    self.pattern.header.stamp = rospy.Time.now()
    rgba = ColorRGBA()

    # All white
    rgba.r = 1.0
    rgba.g = 1.0
    rgba.b = 1.0
    rgba.a = 1.0

    self.pattern.rgb_vals = [rgba] * 5
    self.led_publisher.publish(self.pattern)

  def hook(self):
    # Stop moving vehicle
    print("SHUTTING DOWN")
    self.twist.v = 0
    self.twist.omega = 0
    self.vel_pub.publish(self.twist)
    for _ in range(8):
      self.vel_pub.publish(self.twist)

if __name__ == "__main__":
  node = LaneFollowNode("lane_follow_node")
  rate = rospy.Rate(8)  # 8hz
  while not rospy.is_shutdown():
    node.drive()
    rate.sleep()