#include "carcontrol.h"

CarControl::CarControl()
{
    carPos.x = 120;
    carPos.y = 300;
    steer_publisher = node_obj1.advertise<std_msgs::Float32>("Team1_steerAngle",10);
    speed_publisher = node_obj2.advertise<std_msgs::Float32>("Team1_speed",10);

    error_proportional_ = 0.0;
    error_integral_     = 0.0;
    error_derivative_   = 0.0;
}

CarControl::~CarControl() {}

float CarControl::errorAngle(const Point &dst)
{
    if (dst.x == carPos.x) return 0;
    if (dst.y == carPos.y) return (dst.x < carPos.x ? -90 : 90);
    double pi = acos(-1.0);
    double dx = dst.x - carPos.x;
    double dy = carPos.y - dst.y; 
    if (dx < 0) return -atan(-dx / dy) * 180 / pi;
    return atan(dx / dy) * 180 / pi;
}

void CarControl::Init(float Kp, float Ki, float Kd) {
    kP_ = Kp;
    kI_ = Ki;
    kD_ = Kd;
}

void CarControl::UpdateError(float cte) {
    error_integral_     += cte;
    error_derivative_    = cte - error_proportional_;
    error_proportional_  = cte;
}

float CarControl::TotalError() {
    return (kP_ * error_proportional_ + kI_ * error_integral_ + kD_ * error_derivative_);
}

void CarControl::driverCar(const vector<Point> &left, const vector<Point> &right, float velocity){
    // int i = left.size() - 11;
    // float error = preError;
    // float detectError;
    // while (left[i] == DetectLane::null && right[i] == DetectLane::null) {
    //     i--;
    //     if (i < 0) return;
    // }
    // if (left[i] != DetectLane::null && right[i] !=  DetectLane::null)
    // {
    //     detectError = errorAngle((left[i] + right[i]) / 2);
    //     error = errorAngle((left[i] + right[i]) / 2);
    // } 
    // else if (left[i] != DetectLane::null)
    // {
    //     detectError = errorAngle(left[i]);
    //     error = errorAngle(left[i] + Point(laneWidth / 4, 0));
    // }
    // else
    // {
    //     detectError = errorAngle(right[i]);
    //     error = errorAngle(right[i] - Point(laneWidth / 4, 0));
    // }

    // ROS_INFO("i: %d, Detect: %4.2f, Steer: %4.2f", i, detectError, error);

    double k_p = 0.55;
    double k_i = 0.0185;
    double k_d = 0.2;

    // double k_p = 0.55;
    // double k_i = 0.0185;
    // double k_d = 0.2;

    CarControl::Init(k_p, k_i, k_d);

    int i = left.size() - 11;
    float error = error_proportional_;
    float steer = 0;
    while (left[i] == DetectLane::null && right[i] == DetectLane::null) {
        i--;
        if (i < 0) return;
    }

    if (left[i] != DetectLane::null && right[i] !=  DetectLane::null) {
        error = errorAngle((left[i] + right[i]) / 2);
        CarControl::UpdateError(error);
        steer = CarControl::TotalError();
    } 
    else if (left[i] != DetectLane::null){
        error = errorAngle(left[i]);
        CarControl::UpdateError(error);
        steer = CarControl::TotalError();
    }
    else {
        error = errorAngle(right[i]);
        CarControl::UpdateError(error);
        steer = CarControl::TotalError();
    }
    
    ROS_INFO("i: %d, Error: %0.2f, P: %0.2f, I: %0.2f , D: %0.2f, Steer: %0.2f", i, error, kP_ * error_proportional_, kI_ * error_integral_, kD_ * error_derivative_, steer);

    std_msgs::Float32 angle;
    std_msgs::Float32 speed;
 
    // angle.data = error;
    angle.data = steer;
    speed.data = velocity;

    steer_publisher.publish(angle);
    speed_publisher.publish(speed);    
} 
