#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
if [ ${HOSTNAME:0:6} = 'csc229' ]
then
  # Don't run MLP node on the bot
  dt-exec roslaunch lane_follow lane_follow_node.launch veh:=$VEHICLE_NAME
else
  # dt-exec roslaunch lane_follow lane_follow_node.launch veh:=$VEHICLE_NAME
  dt-exec roslaunch mlp_model mlp_model_node.launch veh:=$VEHICLE_NAME
fi

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
