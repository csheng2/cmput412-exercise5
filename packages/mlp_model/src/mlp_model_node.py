#!/usr/bin/env python3

import rospy, rospkg, cv2, os
from duckietown.dtros import DTROS, NodeType
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from turbojpeg import TurboJPEG, TJPF_GRAY
from std_msgs.msg import Int8MultiArray


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL

class MLP(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()

    self.input_fc = nn.Linear(input_dim, 250)
    self.hidden_fc = nn.Linear(250, 100)
    self.output_fc = nn.Linear(100, output_dim)

  def forward(self, x):
    # x = [batch size, height, width]
    batch_size = x.shape[0]
    x = x.view(batch_size, -1)

    # x = [batch size, height * width]
    h_1 = F.relu(self.input_fc(x))

    # h_1 = [batch size, 250]
    h_2 = F.relu(self.hidden_fc(h_1))

    # h_2 = [batch size, 100]
    y_pred = self.output_fc(h_2)

    # y_pred = [batch size, output dim]
    return y_pred, h_2

class MLPModelNode(DTROS):
  def __init__(self, node_name):
    super(MLPModelNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
    self.node_name = node_name
    self.veh = rospy.get_param("~veh")

    self.br = CvBridge()
    self.host = str(os.environ['VEHICLE_NAME'])
    self.jpeg = TurboJPEG()
    self.model_loaded = False

    self.image_sub = rospy.Subscriber("/output/detected_image", Image, self.image_callback)

    self.rospack = rospkg.RosPack()
    self.path = self.rospack.get_path("mlp_model")
    self.trained_model_path = str(self.path) + "/src/model.pt"

    INPUT_DIM = 28 * 28
    OUTPUT_DIM = 10

    self.model = MLP(INPUT_DIM, OUTPUT_DIM)

    # https://stackoverflow.com/questions/60841650/how-to-test-one-single-image-in-pytorch
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.load_state_dict(torch.load(self.trained_model_path, map_location=self.device))
    self.transform_norm = transforms.Compose([
      transforms.Resize((28, 28)),
      transforms.ToTensor(),
    ])

    self.model_loaded = True

    self.pred_hz = 0.25
    self.timer = rospy.Timer(rospy.Duration(1 / self.pred_hz), self.cb_pred_timer)
    self.last_message = None

    self.loginfo("Initialized")

  def image_callback(self, msg):

    print("received image")
    if not msg:
      return
    
    self.last_message = msg

  def predict_image(self, imgArray):
    # TODO: make sure data types is actually array (currently is bytes)
    img = PIL.Image.fromarray(imgArray)

    img_normalized = self.transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze(0)
    img_normalized = img_normalized.to(self.device)

    with torch.no_grad():
      self.model.eval()
      output, _ = self.model(img_normalized)
      index = output.data.cpu().numpy().argmax()
      return index
  
  def cb_pred_timer(self, _):
    if not self.last_message or not self.model_loaded:
      return
    
    print('predicting....')
    predicted_digit = self.predict_image(self.last_message.data)
    print('predicted class:', predicted_digit)

if __name__ == '__main__':
  node = MLPModelNode("mlp_model_node")
  rospy.spin()