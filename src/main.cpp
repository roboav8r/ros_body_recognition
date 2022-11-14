#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ros/ros.h"
#include "std_msgs/String.h"

#include "depthai_ros_msgs/SpatialDetectionArray.h"
#include "depthai_ros_msgs/SpatialDetection.h"
#include "ros_openpose/Frame.h"
#include "ros_openpose/BodyPart.h"

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
std::vector<std::string> features{"LBP", "RGB"};

static const std::string OPENCV_WINDOW = "Human keypoints";
static const std::string HISTOGRAM_WINDOW = "Color Histogram";
static const std::string LBP_WINDOW = "Local Binary Pattern";
static const std::string KEYPOINT_WINDOW = "Keypoints";


/*
DEFINE BODY RECOGNITION CLASS
*/

class BodyRecognizer
{
    private:
    /* 
    PRIVATE MEMBERS
    */

    // Subscribers
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_; 
    //ros::Subscriber openpose_sub_;
    ros::Subscriber yolo_sub_;

    // Publishers
    image_transport::Publisher raw_image_pub_;

    // Variables
    ros_openpose::Frame frame_msg_;
    depthai_ros_msgs::SpatialDetectionArray last_detection_msg_;
    
    // int kp_patch_width_{30};
    // float bp_conf_thresh_{.2};
    // float obj_conf_thresh_{.5};
    // int detection_font_size_{1};
    // const int detection_font_weight_{1};
    // std::string detection_label_;
    // float distance_to_person_;
    // static constexpr int nKeypoints_{25};
    
    // // Feature extraction 
    // static constexpr int nHist_{4};// 4 is the number of feature types (LBP,R,G,B)
    // static constexpr int histSize_{256};
    // static constexpr int histWidth_{512}; 
    // static constexpr int histHeight_{400};
    // bool uniform_ = true, accumulate_ = false;
    // int bin_ = cvRound((double)histWidth_/histSize_);
    // cv::Mat kpImg_;
    // cv::Mat bHist_, gHist_, rHist_, lbpImage_, lbpHist_, grayImg_;
    // cv::Mat rgbHistImage_ = cv::Mat( histHeight_, histWidth_, CV_8UC3, cv::Scalar(0,0,0) );
    // cv::Mat lbpHistImage_ = cv::Mat( histHeight_, histWidth_, CV_8UC3, cv::Scalar(0,0,0) );
    // std::vector <cv::Mat> bgrPlanes_; 

    // // Feature vector for each person
    // struct FeatureHistogram{
    //     cv::Mat hist = cv::Mat::zeros(1, histSize_*nHist_*nKeypoints_, CV_32F);
    // };

    // BodyRecognizer::FeatureHistogram tempFeature_;
    // std::vector<BodyRecognizer::FeatureHistogram> humanFeatures_;


    public:

    // Default Constructor
    BodyRecognizer() : it_(nh_)
    {
        // Subscribe to input video feed and publish output video feed
        image_sub_ = it_.subscribe(image_topic, 10, &BodyRecognizer::imageCallback, this);
        //openpose_sub_ = nh_.subscribe(openpose_topic, 10, &BodyRecognizer::openposeCallback, this);
        yolo_sub_ = nh_.subscribe(detection_topic, 10, &BodyRecognizer::yoloDetectionCallback, this);

        raw_image_pub_ = it_.advertise("/image_converter/output_video", 1);

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

    // Check if this body part is within a bounding box and above detection threshold
    // TODO also check bounds, fix issue with multiple overlapping bounding boxes
    // bool IsValidBodyPart(const ros_openpose::BodyPart& bp, const depthai_ros_msgs::SpatialDetection& sd) {
    //     if (bp.score > bp_conf_thresh_ 
    //         && bp.pixel.x < (sd.bbox.center.x + sd.bbox.size_x/2)
    //         && bp.pixel.x > (sd.bbox.center.x - sd.bbox.size_x/2)
    //         && bp.pixel.y < (sd.bbox.center.y + sd.bbox.size_y/2) 
    //         && bp.pixel.y > (sd.bbox.center.y - sd.bbox.size_y/2)) 
    //     { return true; }
    //     else { return false; } 
    // }

    // bool DetectionIsPerson(const depthai_ros_msgs::SpatialDetection& sd)
    // {
    //     if (sd.results[0].score > obj_conf_thresh_ && class_labels[((int)sd.results[0].id)]=="person") {return true;}
    //     else {return false;}
    // }

    // void DrawDetectionBox(cv::Mat& image, const depthai_ros_msgs::SpatialDetection& sd, const std::string& label, const float& dist) 
    // {
    //     cv::rectangle(image, cv::Point(sd.bbox.center.x + sd.bbox.size_x/2, sd.bbox.center.y + sd.bbox.size_y/2), cv::Point(sd.bbox.center.x - sd.bbox.size_x/2, sd.bbox.center.y - sd.bbox.size_y/2), CV_RGB(0,0,255), 2);
    //     cv::putText(image, label + " @ "+std::to_string(dist) , cv::Point(sd.bbox.center.x - sd.bbox.size_x/2, sd.bbox.center.y - sd.bbox.size_y/2 -10),cv::FONT_HERSHEY_PLAIN, detection_font_size_,CV_RGB(0,0,255), detection_font_weight_);
    // }

    // void DrawBpBox (cv::Mat& image, float& xmin, float& xmax, float& ymin, float& ymax)
    // {
    //     cv::rectangle(image, cv::Point(xmax, ymax), cv::Point(xmin, ymin), CV_RGB(0,255,255), 1);
    // }

    // cv::Mat computeLbp(cv::Mat& img, int histSize, int histRange) {
    //     cv::Mat lbpHist, lbpImg, grayImg;
    //     cv::cvtColor(img, grayImg, CV_BGR2GRAY);
    //     lbp::OLBP_<u_int8_t>(grayImg, lbpImg);
    //     cv::calcHist(&lbpImg,1,0,cv::Mat(),lbpHist,1,&histSize,histRange,uniform,accumulate);
    //     cv::normalize(lbpHist, lbpHist, 0, lbpHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );

    //     return lbpHist;
    // }

    // // Compute LBP feature histogram
    // cv::cvtColor(cv_ptr->image, grayImg_, CV_BGR2GRAY);
    // lbp::OLBP_<u_int8_t>(grayImg_, lbpImage_);
    // cv::calcHist(&lbpImage_,1,0,cv::Mat(),lbpHist_,1,&histSize_,histRange,uniform_,accumulate_);
    // cv::normalize(lbpHist_, lbpHist_, 0, lbpHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );

    // lbpHistImage_ = cv::Mat( histHeight_, histWidth_, CV_8UC3, cv::Scalar(0,0,0) );
    // for( int i = 1; i < histSize_; i++ )
    // {
    //     cv::line( lbpHistImage_, cv::Point( bin_*(i-1), histWidth_ - cvRound(lbpHist_.at<float>(i-1)) ),
    //         cv::Point( bin_*(i), histHeight_ - cvRound(lbpHist_.at<float>(i)) ),
    //         cv::Scalar( 255, 255, 255), 2, 8, 0  );
    // }



    // // Compute color histogram
    // cv::split(cv_ptr->image,bgrPlanes_);
    // float range[] = { 0, 256 }; //the upper boundary is exclusive
    // const float* histRange[] = { range };
    
    // cv::calcHist(&bgrPlanes_[0],1,0,cv::Mat(),bHist_,1,&histSize_,histRange,uniform_,accumulate_); 
    // cv::calcHist(&bgrPlanes_[1],1,0,cv::Mat(),gHist_,1,&histSize_,histRange,uniform_,accumulate_); 
    // cv::calcHist(&bgrPlanes_[2],1,0,cv::Mat(),rHist_,1,&histSize_,histRange,uniform_,accumulate_);

    // cv::normalize(bHist_, bHist_, 0, rgbHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    // cv::normalize(gHist_, gHist_, 0, rgbHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    // cv::normalize(rHist_, rHist_, 0, rgbHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );

    // rgbHistImage_ = cv::Mat( histHeight_, histWidth_, CV_8UC3, cv::Scalar(0,0,0) );
    // for( int i = 1; i < histSize_; i++ )
    // {
    //     cv::line( rgbHistImage_, cv::Point( bin_*(i-1), histWidth_ - cvRound(bHist_.at<float>(i-1)) ),
    //         cv::Point( bin_*(i), histHeight_ - cvRound(bHist_.at<float>(i)) ),
    //         cv::Scalar( 255, 0, 0), 2, 8, 0  );
    //     cv::line( rgbHistImage_, cv::Point( bin_*(i-1), histWidth_ - cvRound(gHist_.at<float>(i-1)) ),
    //         cv::Point( bin_*(i), histHeight_ - cvRound(gHist_.at<float>(i)) ),
    //         cv::Scalar( 0, 255, 0), 2, 8, 0  );
    //     cv::line( rgbHistImage_, cv::Point( bin_*(i-1), histWidth_ - cvRound(rHist_.at<float>(i-1)) ),
    //         cv::Point( bin_*(i), histHeight_ - cvRound(rHist_.at<float>(i)) ),
    //         cv::Scalar( 0, 0, 255), 2, 8, 0  );
    // }


    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {


        // cv_bridge::CvImagePtr cv_ptr;
        // try
        // {
        // cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        // }
        // catch (cv_bridge::Exception& e)
        // {
        // ROS_ERROR("cv_bridge exception: %s", e.what());
        // return;
        // }

        // kpImg_ = cv::Mat(cv_ptr->image.size(), cv_ptr->image.type(), Scalar::all(0));


        // // Draw a box around each detection
        // for (const depthai_ros_msgs::SpatialDetection& detection : detection_msg_.detections)
        // {

        //     //detection_label_ = class_labels[((int)detection.results[0].id)];
        //     humanFeatures_.clear();

        //     // Iterate through visual detections
        //     if (DetectionIsPerson(detection)) {

        //         // Compute distance to person (for scaling the rectangle/patches)
        //         distance_to_person_ = sqrt(pow(detection.position.x,2) + pow(detection.position.y,2) + pow(detection.position.z,2));

        //         if (visualize) { DrawDetectionBox(cv_ptr->image, detection, class_labels[((int)detection.results[0].id)], distance_to_person_); };

        //         // Iterate through OpenPose detections
        //         for (auto& person : frame_msg_.persons)
        //         {
        //             // for (const ros_openpose::BodyPart body_part : person.bodyParts)
        //             for (int jj = 0; jj < nKeypoints_; jj++)
        //             {
        //                 ros_openpose::BodyPart body_part = std::move(person.bodyParts[jj]);
                       
        //                 // Process body part pixels if body part is a valid detection
        //                 if (IsValidBodyPart(body_part, detection))
        //                 {
        //                     // Get bounding box coordinates
        //                     std::cout << jj << std::endl;
        //                     // TODO account for image boundaries / e.g. check for x >=0, x<= image.cols()
        //                     float xmax = std::min((int)(body_part.pixel.x + kp_patch_width_/(2*distance_to_person_)), cv_ptr->image.cols);
        //                     float xmin = std::max((int)(body_part.pixel.x - kp_patch_width_/(2*distance_to_person_)), 0);
        //                     float ymax = std::min((int)(body_part.pixel.y + kp_patch_width_/(2*distance_to_person_)), cv_ptr->image.rows);
        //                     float ymin = std::max((int)(body_part.pixel.y - kp_patch_width_/(2*distance_to_person_)), 0);
        //                     std::cout << "body part point: " << body_part.point.x << ", " << body_part.point.y << ", " << body_part.point.z << std::endl; 
        //                     std::cout << "xmin " << xmin << std::endl;
        //                     std::cout << "xmax " << xmax << std::endl;
        //                     std::cout << "ymin " << ymin << std::endl;
        //                     std::cout << "ymax " << ymax << std::endl;

        //                     cv::Rect bpRect((xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymin+ymax);

        //                     // Draw box around keypoints
        //                     if (visualize) { DrawBpBox(cv_ptr->image, xmin, xmax, ymin, ymax); }

        //                     // Extract keypoint area and display
        //                     //if (visualize) { cv_ptr->image(bpRect).copyTo(kpImg_(bpRect)); }



        //                 }

        //                 // // TODO compute features
        //                 // for (std::string feature : features ) {
        //                 //     switch (feature)
        //                 //     {
        //                 //     case "LBP":
        //                 //         std::cout << "Processing LBP features" << std::endl;
        //                 //         break;
                            
        //                 //     case "RGB"
        //                 //         std::cout << "Processing LBP features" << std::endl;
        //                 //         break;

        //                 //     default:
        //                 //         std::cout << "Can't find feature" << std::endl;
        //                 //         break;
        //                 //     }
        //                 // }
        //             } // for BodyPart loop
        //         } // OpenPose detection loop
        //     } // If DetectionIsPerson loop
        // } // For detection in spatial detections loop


        // VISUALIZE
        // Update Image Window
        // cv::imshow(OPENCV_WINDOW, cv_ptr->image);
        // cv::waitKey(1);

        // cv::imshow(KEYPOINT_WINDOW, kpImg_);
        // cv::waitKey(1);

        // // Update color histogram window
        // cv::imshow(HISTOGRAM_WINDOW, rgbHistImage_);
        // cv::waitKey(1);

        // // Update LBP histogram window
        // cv::imshow(LBP_WINDOW, lbpHistImage_);
        // cv::waitKey(1);

        // // Output modified video stream
        // image_pub_.publish(cv_ptr->toImageMsg());

    }; // Image callback

    // void openposeCallback(const ros_openpose::Frame::ConstPtr& msg)
    // {
    //     // Save message for later processing
    //     frame_msg_ = (*msg);
    // };

    void yoloDetectionCallback(const depthai_ros_msgs::SpatialDetectionArray::ConstPtr& msg)
    {
        // Save message for later processing
        detection_msg_ = (*msg);
    };

}; // BodyRecognizer Class

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