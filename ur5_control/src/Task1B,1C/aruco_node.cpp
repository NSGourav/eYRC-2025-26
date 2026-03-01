#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <example_interfaces/srv/set_bool.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <string>
#include <memory>

// ================ Helper Function ================
std::pair<double, double> calculate_rectangle_area(std::vector<cv::Point2f> coordinates)
{
    double area = 0.0, width = 0.0;

    cv::Point2f TL = coordinates[0];
    cv::Point2f TR = coordinates[1];
    cv::Point2f BR = coordinates[2];
    cv::Point2f BL = coordinates[3];

    double len = std::hypot(TR.x - TL.x, TR.y - TL.y);
    width      = std::hypot(TR.x - BR.x, TR.y - BR.y);

    if (len < width) width = len;
    area = len * width;
    return {area, width};
}

// ================ Class Definition ================
class ArucoTF : public rclcpp::Node
{
public:
    ArucoTF() : Node("aruco_tf_publisher"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        // Subscriptions
        color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&ArucoTF::colorimagecb, this, std::placeholders::_1));

        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw", 10,
            std::bind(&ArucoTF::depthimagecb, this, std::placeholders::_1));

        // // Timer
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds((int)(image_processing_rate_ * 1000)),
            std::bind(&ArucoTF::process_image, this));

        // TF Broadcaster
        br_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // Service
        // service_ = this->create_service<example_interfaces::srv::SetBool>(
        //     "detect_aruco",
        //     std::bind(&ArucoTF::control_processing_callback, this,
        //               std::placeholders::_1, std::placeholders::_2));

        // Camera intrinsics
        cam_mat_ = (cv::Mat1d(3, 3) << 931.1829833984375, 0.0, 640.0,
                   0.0, 931.1829833984375, 360.0,
                   0.0, 0.0, 1.0);
        dist_mat_ = cv::Mat::zeros(1, 5, CV_64F);

        // OpenCV 4.5.4 API
        dictionary_ = cv::makePtr<cv::aruco::Dictionary>(
                  cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50));
        detector_params_ = cv::makePtr<cv::aruco::DetectorParameters>();

        processing_enabled_ = true;

        RCLCPP_INFO(this->get_logger(), "Aruco TF Publisher node started.");
    }

private:
    // Variables
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Service<example_interfaces::srv::SetBool>::SharedPtr service_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> br_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    cv::Mat cv_image_;
    cv::Mat depth_image_;
    cv::Mat cam_mat_, dist_mat_;
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    cv::Ptr<cv::aruco::DetectorParameters> detector_params_;


    double image_processing_rate_ = 0.5;
    bool processing_enabled_;

    // ================== Callbacks ==================
    // void control_processing_callback(
    //     const std::shared_ptr<example_interfaces::srv::SetBool::Request> request,
    //     std::shared_ptr<example_interfaces::srv::SetBool::Response> response)
    // {
    //     processing_enabled_ = request->data;
    //     response->success = true;
    //     response->message = processing_enabled_ ? "Aruco enabled." : "Aruco disabled.";
    //     RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
    // }

    void depthimagecb(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            // Convert keeping original encoding
            // cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);

            depth_image_ = cv_ptr->image;
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Depth image conversion failed: %s", e.what());
        }
    }


    void colorimagecb(const sensor_msgs::msg::Image::SharedPtr msg)

    {
    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);

        if (msg->encoding == "bgr8")
            cv::cvtColor(cv_ptr->image, cv_image_, cv::COLOR_BGR2RGB);
        else if (msg->encoding == "rgb8")
            cv_image_ = cv_ptr->image;
        else
            cv::cvtColor(cv_ptr->image, cv_image_, cv::COLOR_GRAY2RGB);
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "Color image conversion failed: %s", e.what());
    }
    }   

    // ================== Transform Utilities ==================
    void publish_transform(double x, double y, double z,
                           double qx, double qy, double qz, double qw, int id)
    {
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = this->get_clock()->now();
        tf_msg.header.frame_id = "camera_link";
        tf_msg.child_frame_id = "2788_cam_" + std::to_string(id);

        tf_msg.transform.translation.x = z;
        tf_msg.transform.translation.y = x;
        tf_msg.transform.translation.z = y;

        tf_msg.transform.rotation.x = qx;
        tf_msg.transform.rotation.y = qy;
        tf_msg.transform.rotation.z = qz;
        tf_msg.transform.rotation.w = qw;

        br_->sendTransform(tf_msg);
    }

    void rot_cam_frame(int id)
    {
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = this->get_clock()->now();
        tf_msg.header.frame_id = "2788_cam_" + std::to_string(id);
        tf_msg.child_frame_id = "2788_rot_tf_" + std::to_string(id);

        tf_msg.transform.translation.x = 0.0;
        tf_msg.transform.translation.y = 0.0;
        tf_msg.transform.translation.z = 0.0;
        tf_msg.transform.rotation.w = 1.0;//0.7068252;
        tf_msg.transform.rotation.x = 0.0;
        tf_msg.transform.rotation.y = 0.0;//-0.7068252;
        tf_msg.transform.rotation.z = 0.0;

        br_->sendTransform(tf_msg);
    }

    void look_up_transform(int id)  
    {
        try
        {
            geometry_msgs::msg::TransformStamped transform =
                tf_buffer_.lookupTransform("base_link", "2788_rot_tf_" + std::to_string(id), tf2::TimePointZero);

            geometry_msgs::msg::TransformStamped tf_msg;
            tf_msg.header.stamp = this->get_clock()->now();
            tf_msg.header.frame_id = "base_link";
            tf_msg.child_frame_id = "2788_base_" + std::to_string(id);

            tf_msg.transform = transform.transform;
            br_->sendTransform(tf_msg);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
        }
    }

    // ================== Image Processing ==================
    void process_image()
    {
        if (!processing_enabled_ || cv_image_.empty() || depth_image_.empty())
            return;

        if (cv_image_.cols <= 0 || cv_image_.rows <= 0) {
            RCLCPP_WARN(this->get_logger(), "Invalid color image dimensions: %d x %d",
                        cv_image_.cols, cv_image_.rows);
            return;
        }
        if (depth_image_.cols <= 0 || depth_image_.rows <= 0) {
            RCLCPP_WARN(this->get_logger(), "Invalid depth image dimensions: %d x %d",
                        depth_image_.cols, depth_image_.rows);
            return;
        }

        try
        {
            std::vector<std::pair<double, double>> centers;
            std::vector<std::array<double, 4>> quats;
            std::vector<int> ids;
            std::vector<double> distances;   // from ArUco pose (RGB)
            std::vector<double> widths;      // marker widths

            detect_aruco(cv_image_, centers, quats, ids, distances, widths);
            // cv::imshow("RGB Image", cv_image_);
            // cv::waitKey(1);
            for (size_t i = 0; i < ids.size(); i++)
            {
                int cX = static_cast<int>(centers[i].first);
                int cY = static_cast<int>(centers[i].second);
                auto [qx, qy, qz, qw] = quats[i];
                int id = ids[i];

                if (cX < 0 || cX >= depth_image_.cols || cY < 0 || cY >= depth_image_.rows) {
                    RCLCPP_WARN(this->get_logger(),
                                "Skipping marker %d, center (%d,%d) outside depth image (%d x %d)",
                                id, cX, cY, depth_image_.cols, depth_image_.rows);
                    continue;
                }

                // Depth image value
                float depth_value = depth_image_.at<float>(cY, cX);
                double c = static_cast<double>(depth_value);  // already in meters


                // Intrinsics
                double sizeCamX = cv_image_.cols;
                double sizeCamY = cv_image_.rows;
                double centerCamX = sizeCamX / 2.0;
                double centerCamY = sizeCamY / 2.0;

                double focalX = 931.1829833984375, focalY = 931.1829833984375;

                // Back-project to 3D
                double x = c * (sizeCamX - cX - centerCamX) / focalX;
                double y = c * (sizeCamY - cY - centerCamY) / focalY;
                double z = c;

                // Publish TF
                publish_transform(x, y, z, qx, qy, qz, qw, id);
                rot_cam_frame(id);
                look_up_transform(id);

                // Debug (optional)
                RCLCPP_INFO(this->get_logger(),
                            "Marker ID %d: Depth Z=%.3f m | RGB Dist=%.3f m | Width=%.2f px",
                            id, z, distances[i], widths[i]);
            }
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error in process_image: %s", e.what());
        }
    }

    void detect_aruco(const cv::Mat &image,
                    std::vector<std::pair<double, double>> &centers,
                    std::vector<std::array<double, 4>> &quats,
                    std::vector<int> &ids,
                    std::vector<double> &distances,
                    std::vector<double> &widths)
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> corners, rejected;

        // OpenCV ArUco detection
        cv::aruco::detectMarkers(gray, dictionary_, corners, markerIds, detector_params_, rejected);

        for (size_t i = 0; i < markerIds.size(); i++)
        {
            auto [area, width] = calculate_rectangle_area(corners[i]);
            if (area < 1500)
                continue;

            ids.push_back(markerIds[i]);
            widths.push_back(width);

            cv::Point2f TL = corners[i][0], BR = corners[i][2];
            double cX = (TL.x + BR.x) / 2.0;
            double cY = (TL.y + BR.y) / 2.0;
            centers.push_back({cX, cY});

            // Pose estimation for this marker
            std::vector<cv::Vec3d> rvecs, tvecs;
            std::vector<std::vector<cv::Point2f>> single_corner{corners[i]};
            cv::aruco::estimatePoseSingleMarkers(single_corner, 0.15, cam_mat_, dist_mat_, rvecs, tvecs);

            // rvec → quaternion
            cv::Mat R_cv;
            //cv::Rodrigues(rvecs[0], R_cv);

            cv::Vec3d rvec_swapped(rvecs[0][1], rvecs[0][0], rvecs[0][2]);
            cv::Rodrigues(rvec_swapped, R_cv);


            Eigen::Matrix3d R;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    R(r, c) = R_cv.at<double>(r, c);

            // Eigen::Quaterniond q(R);
            tf2::Matrix3x3 tf_R(
                R(0,0), R(0,1), R(0,2),
                R(1,0), R(1,1), R(1,2),
                R(2,0), R(2,1), R(2,2)
            );

            double roll, pitch, yaw;
            tf_R.getRPY(roll, pitch, yaw);  // XYZ convention

            // Apply signs like in Python: -rotation.as_euler('xyz')
            roll  = -roll;
            pitch = -pitch;
            yaw   = -yaw;

            // Quaternion from Euler (note order like Python)
            tf2::Quaternion q;
            q.setRPY(-yaw, pitch, roll);  // matches quaternion_from_euler(-euler[2], euler[1], euler[0])
            q.normalize();

            std::array<double, 4> qarr = {q.x(), q.y(), q.z(), q.w()};
            quats.push_back(qarr);

            // Distance from RGB pose (‖tvec‖)
            double dist_rgb = std::sqrt(tvecs[0][0] * tvecs[0][0] +
                                        tvecs[0][1] * tvecs[0][1] +
                                        tvecs[0][2] * tvecs[0][2]);
            distances.push_back(dist_rgb);
        }
    }

};

// ================== MAIN ==================
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArucoTF>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
