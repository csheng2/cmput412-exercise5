# Exercise 5: ML for Robotics

This repository contains implementation solutions for exercise 5. For information about the project, please read the report at:

[Nadeen Mohamed's site](https://sites.google.com/ualberta.ca/nadeen-cmput-412/written-reports/exercise-5) or [Celina Sheng's site](https://sites.google.com/ualberta.ca/csheng2-cmput-412/exercise-5) or [Sharyat Singh Bhanwala's Site](https://sites.google.com/ualberta.ca/projects/exercise-5)

## Structure

There are two packages in this file: mlp_model and lane_follow. We will discuss the purpose of the python source files for each package (which are located inside the packages `src` folder).

### MLP Model

- `lane_follow_node.py`: Implements a node to that loads out trained MLP. It also defines a service, `mlp_predict_server`, that takes in images and returns the number predicted by the MLP. When it receives a signal with no data, it shuts down the service and terminates the program.

### Lane Follow

- `lane_follow_node.py`: Implements a node to autonomously drive in a Duckietown lane. It connects to the `mlp_predict_server` service, sending it images of numbers that we detect. It continues to lane-follow around the Duckietown, implementing a random walk, until we detect all 10 numbers. At that point, it sends signal to the MLP node via the service to shutdown, and shuts down itself.

## Execution:

To run the program, ensure that the variable `$BOT` stores your robot's host name (ie. `csc229xx`), and run the following commands:

```
dts devel build -f # builds locally
dts devel build -f -H $BOT.local # builds on the robot
dts devel run -R $BOT && dts devel run -H $BOT.local # runs locally and on robot
```

To shutdown the program, enter `CTRL + C` in your terminal.

## Credit:

This code is built from the duckietown template that provides a boilerplate repository for developing ROS-based software in Duckietown (https://github.com/duckietown/template-basic).

Build on top of by Nadeen Mohamed, Celina Sheng, and Sharyat Singh Bhanwala.

Autonomous lane following code was also borrowed from Justin Francis.

Code was also borrowed (and cited in-code) from the following sources:

- https://eclass.srv.ualberta.ca/mod/resource/view.php?id=6952069
- https://stackoverflow.com/questions/60841650/how-to-test-one-single-image-in-pytorch
- https://eclass.srv.ualberta.ca/mod/resource/view.php?id=6964261
- https://pytorch.org/tutorials/beginner/saving_loading_models.html
- https://discord.com/channels/1057734312461619313/1084894586566082770/1087792516625080431
