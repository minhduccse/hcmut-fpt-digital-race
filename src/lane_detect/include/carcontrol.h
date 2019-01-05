#ifndef CARCONTROL_H
#define CARCONTROL_H
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include "std_msgs/Float32.h"

#include <vector>
#include <math.h>

#include "detectlane.h"

using namespace std;
using namespace cv;

class CarControl 
{
public:
    CarControl();
    ~CarControl();
    void driverCar(const vector<Point> &left, const vector<Point> &right, float velocity);
    
    float errorAngle(const Point &dst);

    void Init(float Kp, float Ki, float Kd);

    void UpdateError(float cte);

    float TotalError();

private:
    ros::NodeHandle node_obj1;
    ros::NodeHandle node_obj2;
    
    ros::Publisher steer_publisher;
    ros::Publisher speed_publisher;

    Point carPos;

    float laneWidth = 40;

    float minVelocity = 10;
    float maxVelocity = 50;

    float preError;

    float error_proportional_;
    float error_integral_;
    float error_derivative_;

    float kP_;
    float kI_;
    float kD_;

    int t_kP;
    int t_kI;
    int t_kD;
    
};

#endif
