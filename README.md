# Exercise 5: ML for Robotics

This repository contains implementation solutions for exercise 5. For information about the project, please read the report at:

<!-- TODO: add sharyat's site -->

[Nadeen Mohamed's site](https://sites.google.com/ualberta.ca/nadeen-cmput-412/written-reports/exercise-5) or [Celina Sheng's site](https://sites.google.com/ualberta.ca/csheng2-cmput-412/exercise-5) or [Sharyat Singh Bhanwala's Site]()

## Structure

There are two packages in this file: mlp_model and lane_follow. We will discuss the purpose of the python source files for each package (which are located inside the packages `src` folder).

### MLP Model

<!-- TODO: add info about MLP model -->

### Lane Follow

<!-- TODO: add more info about lane following -->

- `lane_follow_node.py`: Implements a node to autonomously drive in a Duckietown lane. It contains a rosservice, which tells the node whether we want to lane follow or not.

## Execution:

To run the program, ensure that the variable `$BOT` stores your robot's host name (ie. `csc229xx`), and run the following commands:

```
dts devel build -f # builds locally
dts devel build -f -H $BOT.local # builds on the robot
dts devel run -R $BOT.local && dts devel run -H $BOT.local # runs locally and on robot
```

To shutdown the program, enter `CTRL + C` in your terminal.

## Credit:

This code is built from the duckietown template that provides a boilerplate repository for developing ROS-based software in Duckietown (https://github.com/duckietown/template-basic).

Build on top of by Nadeen Mohamed, Celina Sheng, and Sharyat Singh Bhanwala.

Autonomous lane following code was also borrowed from Justin Francis.

Code was also borrowed (and cited in-code) from the following sources:

- <!-- TODO: add sources -->
