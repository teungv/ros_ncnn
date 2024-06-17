#!/bin/bash

# Start Xvfb server
Xvfb :1 -screen 0 1024x768x24 &

echo "Starting an Xvfb screen"

# Wait a bit to ensure Xvfb starts properly
sleep 2

# Export DISPLAY environment variable
export DISPLAY=:1

# Start x11vnc
x11vnc -display :1 -nopw -listen localhost -xkb &

echo "Connecting vnc to the Xvfb screen" 

# Wait a bit to ensure x11vnc starts properly
sleep 2

echo "Engaging vncviewer"

vncviewer localhost:0

# Run your ROS launch file
roslaunch your_package your_launch_file.launch
