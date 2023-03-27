#!/usr/bin/env python3

import rospy, rospkg, os
import numpy as np
from duckietown.dtros import DTROS, NodeType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL

from mlp_model.srv import MLPPredict, MLPPredictResponse

"""
Code for the MLP class taken from Multilayer_Perceptron.ipynb notebook on eClass
link: https://eclass.srv.ualberta.ca/mod/resource/view.php?id=6964261
"""
class MLP(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()

    self.input_fc = nn.Linear(input_dim, 250)
    self.hidden_fc = nn.Linear(250, 100)
    self.output_fc = nn.Linear(100, output_dim)

  def forward(self, x):
    batch_size = x.shape[0]
    x = x.view(batch_size, -1)
    h_1 = F.relu(self.input_fc(x))
    h_2 = F.relu(self.hidden_fc(h_1))
    y_pred = self.output_fc(h_2)
    return y_pred, h_2

class MLPModelNode(DTROS):
  def __init__(self, node_name):
    super(MLPModelNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
    self.node_name = node_name
    self.veh = rospy.get_param("~veh")
    self.model_loaded = False

    # Shutdown hook
    rospy.on_shutdown(self.hook)

    # Get path to trained model file
    """
    Code to get the path from the path from the source folder
    provided by Zepeng Xiao on class discord
    Link: https://discord.com/channels/1057734312461619313/1084894586566082770/1087792516625080431
    """
    self.rospack = rospkg.RosPack()
    self.path = self.rospack.get_path("mlp_model")
    self.trained_model_path = str(self.path) + "/src/model.pt"

    # Initialize model
    INPUT_DIM = 28 * 28
    OUTPUT_DIM = 10
    self.model = MLP(INPUT_DIM, OUTPUT_DIM)

    """
    Code to load the model from a save model file taken from
    PyTorch's Saving and Loading Models tutorial
    Link: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.load_state_dict(torch.load(self.trained_model_path, map_location=self.device))
    self.transform_norm = transforms.Compose([
      transforms.Resize((28, 28)),
      transforms.ToTensor(),
    ])
    self.model_loaded = True

    # Create service
    self.service = rospy.Service('mlp_predict_server', MLPPredict, self.predict_image)

    self.loginfo("Initialized")

  def predict_image(self, rawImage):
    """
    Instructions to load the image for predicitons adapted from
    'How to test one single image in pytorch' post on stackoverflow
    author: https://stackoverflow.com/users/6353583/nahim-terrazas
    link: https://stackoverflow.com/questions/60841650/how-to-test-one-single-image-in-pytorch
    """
    # Shutdown the service if the image is empty
    if len(rawImage.image.data) == 0:
      rospy.signal_shutdown("Found all ten numbers!")
      return MLPPredictResponse(-1)
    
    # Return if the model is not already loaded
    if not self.model_loaded:
      return MLPPredictResponse(-1)
    
    # Convert the byte array from bytes to numpy array
    try:
      imgArray = np.frombuffer(rawImage.image.data, dtype=np.uint8)
      imgArray = np.reshape(imgArray, (-1, rawImage.image.width))
    except Exception as e:
      print(e)
      return MLPPredictResponse(-1)

    img = PIL.Image.fromarray(imgArray)

    # Apply transformation to get tensor and load to device
    img_normalized = self.transform_norm(img).float()
    img_normalized = img_normalized.to(self.device)

    # Predict number
    with torch.no_grad():
      self.model.eval()
      output, _ = self.model(img_normalized)
      predictions = output.data.cpu()
      index = output.data.cpu().numpy().argmax()
      distribution = nn.functional.softmax(predictions, dim=1)
      probability = distribution.numpy()[0][index]

      # only return predicted digit if we have a relatively high probability
      if probability > 0.9:
        return MLPPredictResponse(index)
      else:
        return MLPPredictResponse(-1)
  
  def hook(self):
    # Shut down service
    print("SHUTTING DOWN")
    self.service.shutdown('Done!')

if __name__ == '__main__':
  node = MLPModelNode("mlp_model_node")
  rospy.spin()