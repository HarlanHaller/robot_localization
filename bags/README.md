# Bag Files

There are 4 bag files present in this repository. Two of them `macfirst_floor_take_1` and `macfirst_floor_take_1` are used to test play robot movements and laser scan data from a privous robot run for testing purposes. The other two bag recording `pf_in_gazebo` and `pf_in_mac` are bags we recorded to record the preformance of our particle filter. 

## How to run
To run these bags first run rviz (from your root directory)  
`rviz2 -d ./src/robot_localization/rviz/turtlebot_bag_files.rviz`  
Next run  
`ros2 bag play ./src/robot_localization/bags/BAG_NAME --clock`