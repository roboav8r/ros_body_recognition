#include "ros/ros.h"
#include "std_msgs/String.h"

#include "depthai_ros_msgs/SpatialDetectionArray.h"
#include "ros_openpose/Frame.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lbp.hpp"


/*
DEFINE CONSTANTS/INPUTS
*/
enum config_types{
    oakd
};

const std::vector<std::string> class_labels = {
    "person",        "bicycle",      "car",           "motorbike",     "aeroplane",   "bus",         "train",       "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",     "parking meter", "bench",       "bird",        "cat",         "dog",          "horse",
    "sheep",         "cow",          "elephant",      "bear",          "zebra",       "giraffe",     "backpack",    "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",       "skis",          "snowboard",   "sports ball", "kite",        "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket", "bottle",        "wine glass",  "cup",         "fork",        "knife",        "spoon",
    "bowl",          "banana",       "apple",         "sandwich",      "orange",      "broccoli",    "carrot",      "hot dog",      "pizza",
    "donut",         "cake",         "chair",         "sofa",          "pottedplant", "bed",         "diningtable", "toilet",       "tvmonitor",
    "laptop",        "mouse",        "remote",        "keyboard",      "cell phone",  "microwave",   "oven",        "toaster",      "sink",
    "refrigerator",  "book",         "clock",         "vase",          "scissors",    "teddy bear",  "hair drier",  "toothbrush"};


std::string detection_topic{"/yolov4_publisher/color/yolov4_Spatial_detections"};
std::string image_topic{"/yolov4_publisher/color/image"};
std::string openpose_topic{"/frame"};

bool visualize{true};


static const std::string OPENCV_WINDOW = "Human keypoints";
static const std::string HISTOGRAM_WINDOW = "Color Histogram";
static const std::string LBP_WINDOW = "Local Binary Pattern";


/*
DEFINE BODY RECOGNITION CLASS
*/

class BodyRecognizer
{
    /* 
    PRIVATE MEMBERS
    */

    // Subscribers
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Subscriber openpose_sub_;
    ros::Subscriber yolo_sub_;

    // Publishers
    image_transport::Publisher image_pub_;

    // Variables
    ros_openpose::Frame frame_msg_;
    depthai_ros_msgs::SpatialDetectionArray detection_msg_;
    int kp_patch_width_{30};
    float bp_conf_thresh_{.2};
    float obj_conf_thresh_{.5};
    int detection_font_size_{1};
    const int detection_font_weight_{1};
    std::string detection_label_;
    float distance_to_person_;
    const int nKeypoints_{25};
    
    // Feature extraction 
    const int histSize_{256};
    const int histWidth_{512}; 
    const int histHeight_{400};
    bool uniform_ = true, accumulate_ = false;
    int bin_ = cvRound((double)histWidth_/histSize_);
    cv::Mat bHist_, gHist_, rHist_, lbpImage_, lbpHist_, grayImg_;
    cv::Mat rgbHistImage_ = cv::Mat( histHeight_, histWidth_, CV_8UC3, cv::Scalar(0,0,0) );
    cv::Mat lbpHistImage_ = cv::Mat( histHeight_, histWidth_, CV_8UC3, cv::Scalar(0,0,0) );
    std::vector <cv::Mat> bgrPlanes_; 

    // Feature vector for each person
    struct FeatureHistogram{
        cv::Mat hist = cv::Mat::zeros(1, histSize_*4*nKeypoints_, CV_32F);// 4 is the number of features (LBP,R,G,B)
    };
    std::vector<BodyRecognizer::FeatureHistogram> humanFeatures;

public:

    // Default Constructor
    BodyRecognizer() : it_(nh_)
    {
        // Subscribe to input video feed and publish output video feed
        image_sub_ = it_.subscribe(image_topic, 10, &BodyRecognizer::imageCallback, this);
        openpose_sub_ = nh_.subscribe(openpose_topic, 10, &BodyRecognizer::openposeCallback, this);
        yolo_sub_ = nh_.subscribe(detection_topic, 10, &BodyRecognizer::yoloDetectionCallback, this);

        image_pub_ = it_.advertise("/image_converter/output_video", 1);

        cv::namedWindow(OPENCV_WINDOW);
        cv::namedWindow(HISTOGRAM_WINDOW);
        cv::namedWindow(LBP_WINDOW);

    }

    // Default Destructor
    ~BodyRecognizer()
    {
        cv::destroyWindow(OPENCV_WINDOW);
        cv::destroyWindow(HISTOGRAM_WINDOW);
        cv::destroyWindow(LBP_WINDOW);
    }

    // Callbacks / member functions

    // cv::Mat computeLbp(cv::Mat& img, int histSize, int histRange) {
    //     cv::Mat lbpHist, lbpImg, grayImg;
    //     cv::cvtColor(img, grayImg, CV_BGR2GRAY);
    //     lbp::OLBP_<u_int8_t>(grayImg, lbpImg);
    //     cv::calcHist(&lbpImg,1,0,cv::Mat(),lbpHist,1,&histSize,histRange,uniform,accumulate);
    //     cv::normalize(lbpHist, lbpHist, 0, lbpHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );

    //     return lbpHist;
    // }



    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
        }

        // Compute color histogram/LBP features 
        cv::split(cv_ptr->image,bgrPlanes_);
        float range[] = { 0, 256 }; //the upper boundary is exclusive
        const float* histRange[] = { range };
        
        cv::calcHist(&bgrPlanes_[0],1,0,cv::Mat(),bHist_,1,&histSize_,histRange,uniform_,accumulate_); 
        cv::calcHist(&bgrPlanes_[1],1,0,cv::Mat(),gHist_,1,&histSize_,histRange,uniform_,accumulate_); 
        cv::calcHist(&bgrPlanes_[2],1,0,cv::Mat(),rHist_,1,&histSize_,histRange,uniform_,accumulate_);

        cv::normalize(bHist_, bHist_, 0, rgbHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );
        cv::normalize(gHist_, gHist_, 0, rgbHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );
        cv::normalize(rHist_, rHist_, 0, rgbHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );

        rgbHistImage_ = cv::Mat( histHeight_, histWidth_, CV_8UC3, cv::Scalar(0,0,0) );
        for( int i = 1; i < histSize_; i++ )
        {
            cv::line( rgbHistImage_, cv::Point( bin_*(i-1), histWidth_ - cvRound(bHist_.at<float>(i-1)) ),
                cv::Point( bin_*(i), histHeight_ - cvRound(bHist_.at<float>(i)) ),
                cv::Scalar( 255, 0, 0), 2, 8, 0  );
            cv::line( rgbHistImage_, cv::Point( bin_*(i-1), histWidth_ - cvRound(gHist_.at<float>(i-1)) ),
                cv::Point( bin_*(i), histHeight_ - cvRound(gHist_.at<float>(i)) ),
                cv::Scalar( 0, 255, 0), 2, 8, 0  );
            cv::line( rgbHistImage_, cv::Point( bin_*(i-1), histWidth_ - cvRound(rHist_.at<float>(i-1)) ),
                cv::Point( bin_*(i), histHeight_ - cvRound(rHist_.at<float>(i)) ),
                cv::Scalar( 0, 0, 255), 2, 8, 0  );
        }

        // Compute LBP feature histogram
        cv::cvtColor(cv_ptr->image, grayImg_, CV_BGR2GRAY);
        lbp::OLBP_<u_int8_t>(grayImg_, lbpImage_);
        cv::calcHist(&lbpImage_,1,0,cv::Mat(),lbpHist_,1,&histSize_,histRange,uniform_,accumulate_);
        cv::normalize(lbpHist_, lbpHist_, 0, lbpHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );

        lbpHistImage_ = cv::Mat( histHeight_, histWidth_, CV_8UC3, cv::Scalar(0,0,0) );
        for( int i = 1; i < histSize_; i++ )
        {
            cv::line( lbpHistImage_, cv::Point( bin_*(i-1), histWidth_ - cvRound(lbpHist_.at<float>(i-1)) ),
                cv::Point( bin_*(i), histHeight_ - cvRound(lbpHist_.at<float>(i)) ),
                cv::Scalar( 255, 255, 255), 2, 8, 0  );
        }


        // Draw a box around each detection
        for (auto& detection : detection_msg_.detections)
        {

            detection_label_ = class_labels[((int)detection.results[0].id)];

            if (detection.results[0].score > obj_conf_thresh_ && detection_label_=="person") {
                // Compute distance to person (for scaling the rectangle/patches)
                distance_to_person_ = sqrt(pow(detection.position.x,2) + pow(detection.position.y,2) + pow(detection.position.z,2));

                cv::rectangle(cv_ptr->image, cv::Point(detection.bbox.center.x + detection.bbox.size_x/2, detection.bbox.center.y + detection.bbox.size_y/2), cv::Point(detection.bbox.center.x - detection.bbox.size_x/2, detection.bbox.center.y - detection.bbox.size_y/2), CV_RGB(0,0,255), 2);
                cv::putText(cv_ptr->image, detection_label_+" @ "+std::to_string(distance_to_person_) , cv::Point(detection.bbox.center.x - detection.bbox.size_x/2, detection.bbox.center.y - detection.bbox.size_y/2 -10),cv::FONT_HERSHEY_PLAIN, detection_font_size_,CV_RGB(0,0,255), detection_font_weight_);
                // class_labels(detection.results[0].id)

                // Draw a box around each keypoint
                for (auto& person : frame_msg_.persons)
                {
                    for (auto& body_part : person.bodyParts)
                    {
                        // Only plot boxes within a human bounding box
                        if (body_part.score > bp_conf_thresh_ 
                        && body_part.pixel.x < (detection.bbox.center.x + detection.bbox.size_x/2)
                        && body_part.pixel.x > (detection.bbox.center.x - detection.bbox.size_x/2)
                        && body_part.pixel.y < (detection.bbox.center.y + detection.bbox.size_y/2) 
                        && body_part.pixel.y > (detection.bbox.center.y - detection.bbox.size_y/2))
                        {
                            cv::rectangle(cv_ptr->image, cv::Point(body_part.pixel.x + kp_patch_width_/(2*distance_to_person_), body_part.pixel.y + kp_patch_width_/(2*distance_to_person_)), cv::Point(body_part.pixel.x - kp_patch_width_/(2*distance_to_person_), body_part.pixel.y - kp_patch_width_/(2*distance_to_person_)), CV_RGB(0,255,255), 1);
                        }
                    }
                }

            } // If==human detection loop

        }

        // Update Image Window
        cv::imshow(OPENCV_WINDOW, cv_ptr->image);
        cv::waitKey(3);

        // Update color histogram window
        cv::imshow(HISTOGRAM_WINDOW, rgbHistImage_);
        cv::waitKey(3);

        // Update LBP histogram window
        cv::imshow(LBP_WINDOW, lbpHistImage_);
        cv::waitKey(3);

        // Output modified video stream
        image_pub_.publish(cv_ptr->toImageMsg());

    };

    void openposeCallback(const ros_openpose::Frame::ConstPtr& msg)
    {
        // Save message for later processing
        frame_msg_ = (*msg);
    };

    void yoloDetectionCallback(const depthai_ros_msgs::SpatialDetectionArray::ConstPtr& msg)
    {
        // Save message for later processing
        detection_msg_ = (*msg);
    };

};

/*
MAIN LOOP
*/


int main(int argc, char **argv)
{

    ros::init(argc, argv, "body_recognizer");

    BodyRecognizer br;

    ros::spin();
    return 0;
}