#!/bin/bash
sudo apt update

#Variable to store if error occured
ERROR_OCCURRED=0
install_apt_package() {
    sudo apt-get install -y "$1" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "\e[32mPackage: $1 installed successfully!\e[0m"
    else
        echo -e "\e[31mAPT Package: Failed to install $1 :(\e[0m"
        ERROR_OCCURRED=1
    fi
}

install_pip_package() {
    pip3 install "$1" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "\e[32mPackage: $1 installed successfully!\e[0m"
    else
        echo -e "\e[31mPIP Package: Failed to install $1 :(\e[0m"
        ERROR_OCCURRED=1
    fi
}

# Install necessary packages
echo -e "\e[34mInstalling required packages...\e[0m"
install_apt_package "python3-pip"
install_apt_package "ros-humble-ros-gz"
install_apt_package "ros-humble-ros-gz-bridge"
install_apt_package "ros-humble-ign-ros2-control"
install_apt_package "ros-humble-teleop-twist-keyboard"
install_apt_package "ros-humble-robot-state-publisher"
install_apt_package "ros-humble-joint-state-publisher"
install_apt_package "ros-humble-joint-state-publisher-gui"
install_apt_package "ros-humble-ros2-controllers"
install_apt_package "ros-humble-topic-tools"
install_apt_package "ros-humble-xacro"
install_apt_package "ros-humble-tf-transformations"
install_apt_package "ros-humble-joint-state-broadcaster"
install_apt_package "ros-humble-joint-trajectory-controller"
install_apt_package "ros-humble-controller-manager"
install_apt_package "ros-humble-gazebo-msgs"
install_apt_package "ros-dev-tools"
install_apt_package "libgz-plugin3-dev"
install_apt_package "libgz-common5-dev"
install_apt_package "libgz-msgs11-dev"
install_apt_package "libgz-transport14-dev"
install_apt_package "nlohmann-json3-dev"


install_pip_package "python-fcl"
install_pip_package "urdf-parser-py"
install_pip_package "networkx==3.4.2"
install_pip_package "transforms3d"
install_pip_package "opencv-contrib-python==4.7.0.72"
install_pip_package "numpy==1.21.5"
install_pip_package "trimesh"
install_pip_package "cryptocode"

# Final message
if [ $ERROR_OCCURRED -eq 1 ]; then
    echo -e "\e[31m---------------------------------\nSetup completed with some errors. Please check the messages above.\n---------------------------------\n\e[0m"
else
    echo -e "\e[32m---------------------------------\nSetup complete!\n---------------------------------\n\e[0m"
fi
