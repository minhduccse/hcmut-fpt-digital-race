
# ROS_Package_example
## I. Installation
 - Ubuntu 16.04 or newer
 - One of these following version of [ROS](https://ros.org)
    - [Lunar Loggerhead](http://wiki.ros.org/lunar)
    - [Melodic Morenia](http://wiki.ros.org/melodic)
    - It is recommended to install the full version
      ```
      $ sudo apt-get install ros-<distro>-desktop-full
      ```
 - Create catkin workspace
    ```
    $ mkdir -p ~/catkin/src
    $ cd ~/catkin/
    $ catkin_make
    $ echo "source ~/catkin/devel/setup.bash" >> ~/.bashrc
    $ source ~/.bashrc
    ```
  
 - Install rosbridge-suite
    ```
    $ sudo apt-get install ros-<distro>-rosbridge-server
    ```
    
 - Run

    ```
    $ roslaunch lane_detect lane_detect.launch
    ```
    

 - Unity info:
 Team: Team1
 URL: ws://127.0.0.1:9090
