#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

#include "ros/ros.h"
#include "std_msgs/String.h"

#include "depthai_ros_msgs/SpatialDetectionArray.h"
#include "depthai_ros_msgs/SpatialDetection.h"
#include "ros_openpose/Frame.h"
#include "ros_openpose/BodyPart.h"
#include "ros_body_recognition/SpatialFeature.h"
#include "ros_body_recognition/SpatialFeatureArray.h"


#include "lbp.hpp"


/*
DEFINE CONSTANTS/INPUTS
*/
enum config_types{
    oakd
};

enum body_rec_state
{
    READY=0,
    GOT_IMG=1,
    GOT_DET=2,
    OP_MSG_SENT=3,
    OP_MSG_RCVD=4,
    FEAT_EXTRACTED=5,
    TIMEOUT=6
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

const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS = {
{ 0,      "Nose"},    {13,      "LKnee"},
{ 1,      "Neck"},    {14,     "LAnkle"},
{ 2, "RShoulder"},    {15,       "REye"},
{ 3,    "RElbow"},    {16,       "LEye"},
{ 4,    "RWrist"},    {17,       "REar"},
{ 5, "LShoulder"},    {18,       "LEar"},
{ 6,    "LElbow"},    {19,    "LBigToe"},
{ 7,    "LWrist"},    {20,  "LSmallToe"},
{ 8,    "MidHip"},    {21,      "LHeel"},
{ 9,      "RHip"},    {22,    "RBigToe"},
{10,     "RKnee"},    {23,  "RSmallToe"},
{11,    "RAnkle"},    {24,      "RHeel"},
{12,      "LHip"},    {25, "Background"}};


std::string detection_topic{"yolov4_publisher/color/yolov4_Spatial_detections"};
std::string image_topic{"yolov4_publisher/color/image"};
std::string feat_topic{"spatial_features"};

bool visualize{true};
//std::vector<std::string> features{"LBP", "RGB"};
//const int lbpIndex = 0;
const int rIndex = 2;
const int gIndex = 1;
const int bIndex = 0;
const int hIndex = 3; // hue
const int sIndex = 4; // saturation
const int vIndex = 5; // value

static const std::string OPENCV_WINDOW = "Human Detections";
static const std::string FEAT_HIST_WINDOW = "Feature Histograms";
//static const std::string DEVEL_WINDOW = "Devel";

//static const std::vector<uint> kpIndices_{0,1,2,5,8,9,12}; // Face, chest, shoulders left/right, hip center/left/right
static const std::vector<uint> kpIndices_{1,2,5}; // chest, left shoulder, right shoulder TODO make this a parameter
static const size_t nKeypoints_{kpIndices_.size()};

/*
DEFINE BODY RECOGNITION CLASS
*/

class BodyRecognizer
{
    private:
    /* 
    PRIVATE MEMBERS
    */

    // Member variables
    std::string openpose_in_topic_;
    std::string openpose_out_topic_;
    float img_proc_timeout_;
    body_rec_state control_state_;

    // Subscribers
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_; 
    ros::Subscriber openpose_sub_;
    ros::Subscriber yolo_sub_;

    // Publishers
    image_transport::Publisher raw_image_pub_;
    ros::Publisher feat_pub_;

    // Images and messages from ROS
    ros_openpose::Frame frame_msg_;
    boost::shared_ptr<ros_openpose::Frame const> frame_ptr_;
    depthai_ros_msgs::SpatialDetectionArray last_detection_msg_;
    sensor_msgs::Image last_img_;
    ros_body_recognition::SpatialFeature sf_msg_;
    ros_body_recognition::SpatialFeatureArray sf_array_msg_;

    // Detection-to-openpose association variables
    std::vector<int> validDetections;
    uint maxNumHumans_{0};
    Eigen::MatrixXd associationMatrix_;
    Eigen::Vector2i associationIndices_;

    // Keypoint parameters
    cv_bridge::CvImagePtr cv_ptr_bgr_;
    cv_bridge::CvImagePtr cv_ptr_hsv_;
    cv::Mat kpImg_;

    int kp_patch_width_{30};
    float bp_conf_thresh_{.2};
    float obj_conf_thresh_{.5};
    int detection_font_size_{1};
    const int detection_font_weight_{1};
    std::string detection_label_;
    double distance_to_det_;
    
    // Feature extraction parameters
    int nFeat_{6};// number of feature types (R, G, B, H, S, V)
    int histSize_{4}; // TODO make this a parameter
    int histWidth_{1536}; 
    int histHeight_{400};
    bool uniform_ = true, accumulate_ = false;

    int bin_ = cvRound((double)histWidth_/(histSize_*nFeat_*nKeypoints_));
    cv::Mat featHistImg_;
    // cv::Mat bHist_, gHist_, rHist_, lbpImage_, lbpHist_, grayImg_;
    // cv::Mat rgbHistImage_ = cv::Mat( histHeight_, histWidth_, CV_8UC3, cv::Scalar(0,0,0) );
    // cv::Mat lbpHistImage_ = cv::Mat( histHeight_, histWidth_, CV_8UC3, cv::Scalar(0,0,0) );
    std::vector <cv::Mat> bgrPlanes_;
    std::vector <cv::Mat> hsvPlanes_;
    
    // // Feature vector for each person
    cv::Mat tempFeature_;
    std::vector<cv::Mat> humanFeatures_;
    const uint max_humans_{5};


    public:

    // Default Constructor
    BodyRecognizer() : it_(nh_)
    {

        // Get parameter values and assign to member variables
        nh_.param<std::string>("openpose_in_topic", openpose_in_topic_, "/image_view/output");
        nh_.param<std::string>("openpose_out_topic", openpose_out_topic_, "/frame");
        nh_.param<float>("img_proc_timeout", img_proc_timeout_, 0.25);

        // Subscribe to input video feed and publish output video feed
        image_sub_ = it_.subscribe(image_topic, 10, &BodyRecognizer::imageCallback, this);
        openpose_sub_ = nh_.subscribe(openpose_out_topic_, 10, &BodyRecognizer::openposeCallback, this);
        yolo_sub_ = nh_.subscribe(detection_topic, 10, &BodyRecognizer::yoloDetectionCallback, this);
        raw_image_pub_ = it_.advertise(openpose_in_topic_, 1);
        feat_pub_ = nh_.advertise<ros_body_recognition::SpatialFeatureArray>(feat_topic,1);

        // Configure control state
        control_state_ = READY;

        cv::namedWindow(OPENCV_WINDOW);
        cv::namedWindow(FEAT_HIST_WINDOW);
        //cv::namedWindow(DEVEL_WINDOW);
    }

    // Default Destructor
    ~BodyRecognizer()
    {
        cv::destroyWindow(OPENCV_WINDOW);
        cv::destroyWindow(FEAT_HIST_WINDOW);
        //cv::destroyWindow(DEVEL_WINDOW);
    }

    // Callbacks / member functions

    // Check if this body part is above detection threshold
    bool IsValidBodyPart(const ros_openpose::BodyPart& bp) {
        if (bp.score > bp_conf_thresh_) { return true; }
        else { return false; } 
    }

    bool DetectionIsPerson(const depthai_ros_msgs::SpatialDetection& sd)
    {
        if (sd.results[0].score > obj_conf_thresh_ && class_labels[((int)sd.results[0].id)]=="person") {return true;}
        else {return false;}
    }

    double DistanceToDetection(const depthai_ros_msgs::SpatialDetection& sd)
    {
        return sqrt(pow(sd.position.x,2) + pow(sd.position.y,2) + pow(sd.position.z,2));
    }

    void GetValidDetections()
    {
        validDetections.clear();

        for (int ii=0 ; ii < last_detection_msg_.detections.size(); ii++)
        {
            // Compute distance to detection
            distance_to_det_ = DistanceToDetection(last_detection_msg_.detections[ii]);

            // Only process the detection if it's a human and if spatial estimate is valid
            if (DetectionIsPerson(last_detection_msg_.detections[ii]) && distance_to_det_ > 0.0) {
                validDetections.push_back(ii);
            }
        }
    }

    double MatchLikelihood( depthai_ros_msgs::SpatialDetection& detection, ros_openpose::Person& person){

        // Put bounding box mean and covariance into matrices
        Eigen::MatrixXd boundBoxCov(2,2);
        boundBoxCov << detection.bbox.size_x, 0, 0, detection.bbox.size_y;
        Eigen::MatrixXd boundBoxCovInv = boundBoxCov.inverse();

        // Prepare body part vector and initialize count of valid parts
        int nValidBodyParts{0};
        Eigen::MatrixXd bodyPartDelta(2,1); // difference between bounding box mean and current body part pixel

        double likelihoodSum{0};
        for (int nn = 0; nn < person.bodyParts.size(); nn++) {
            
            // Only consider body part pixels !=(0,0)
            if (person.bodyParts[nn].pixel.x==0 && person.bodyParts[nn].pixel.y==0) {continue;}

            bodyPartDelta << (person.bodyParts[nn].pixel.x - detection.bbox.center.x), (person.bodyParts[nn].pixel.y - detection.bbox.center.y);
            nValidBodyParts+=1;
            likelihoodSum += (double)(bodyPartDelta.transpose()*boundBoxCovInv*bodyPartDelta).sum();

        }

        return pow(2*CV_PI,-nValidBodyParts)*pow(boundBoxCov.determinant(),-nValidBodyParts/2)*exp(-likelihoodSum/2);

    }

    void DrawDetectionBox(cv::Mat& image, const depthai_ros_msgs::SpatialDetection& sd, const cv::Scalar& color ) 
    {
        cv::rectangle(image, cv::Point(sd.bbox.center.x + sd.bbox.size_x/2, sd.bbox.center.y + sd.bbox.size_y/2), cv::Point(sd.bbox.center.x - sd.bbox.size_x/2, sd.bbox.center.y - sd.bbox.size_y/2), color, 2);
        cv::putText(image, class_labels[((int)sd.results[0].id)] + " @ "+std::to_string(sd.position.x) + ","+ std::to_string(sd.position.y) + "," + std::to_string(sd.position.z), cv::Point(sd.bbox.center.x - sd.bbox.size_x/2, sd.bbox.center.y - sd.bbox.size_y/2 -10),cv::FONT_HERSHEY_PLAIN, detection_font_size_,CV_RGB(0,0,255), detection_font_weight_);
    }

    void DrawBpBox (cv::Mat& image, float& xmin, float& xmax, float& ymin, float& ymax, const cv::Scalar& color)
    {
        cv::rectangle(image, cv::Point(xmax, ymax), cv::Point(xmin, ymin), color, 1);
    }

    void DrawKeyPoints(cv::Mat& image, const ros_openpose::Person& person, float& dist, const cv::Scalar& color)
    {
        for (const ros_openpose::BodyPart body_part : person.bodyParts)
        {
            if (IsValidBodyPart(body_part))
            {
                float xmax = std::min((int)(body_part.pixel.x + kp_patch_width_/(2*dist)), image.cols);
                float xmin = std::max((int)(body_part.pixel.x - kp_patch_width_/(2*dist)), 0);
                float ymax = std::min((int)(body_part.pixel.y + kp_patch_width_/(2*dist)), image.rows);
                float ymin = std::max((int)(body_part.pixel.y - kp_patch_width_/(2*dist)), 0);
                
                cv::Rect bpRect((xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymin+ymax);
            
                // Draw box around keypoints
                DrawBpBox(image, xmin, xmax, ymin, ymax, color); 
            }
        }
    }


    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        if (control_state_==READY) {
            // Save message
            last_img_ = (*msg);

            // Set control flags
            control_state_ = GOT_IMG;

        } else if (control_state_==GOT_DET) {
            // Save message
            last_img_ = (*msg);
            raw_image_pub_.publish(*msg);
            control_state_ = OP_MSG_SENT;
        };
    }; // Image callback

    void yoloDetectionCallback(const depthai_ros_msgs::SpatialDetectionArray::ConstPtr& msg)
    {
        if (control_state_==READY) {
            // Save detection message
            last_detection_msg_ = (*msg);
            control_state_ = GOT_DET;

        } else if (control_state_==GOT_IMG) {

            // Publish associated image & set booleans
            last_detection_msg_ = (*msg);
            raw_image_pub_.publish(last_img_);
            control_state_ = OP_MSG_SENT;
        }

    };


    void openposeCallback(const ros_openpose::Frame::ConstPtr& msg)
    {
        if (control_state_ == OP_MSG_SENT) {
            // Save message for later processing
            frame_msg_ = (*msg);

            std::cout << frame_msg_.header.stamp - last_img_.header.stamp << std::endl;

            control_state_ = OP_MSG_RCVD;

            // Extract features
            extractFeatures();

        }

    };

    void extractFeatures() 
    {
        // Find out maximum number of people in the scene
        GetValidDetections();
        
        maxNumHumans_ = min( validDetections.size(), frame_msg_.persons.size() );
        if ( maxNumHumans_==0 ) // TODO check for this on callback
        {    
            std::cout << "No detections from OAK-D or no detections from OpenPose. Exiting extractFeatures()" <<std::endl;
            control_state_ = READY;
            return;
        }

        // Convert ROS message to CV matrix
        try
        {
            cv_ptr_bgr_ = cv_bridge::toCvCopy(last_img_, sensor_msgs::image_encodings::BGR8);
            cv_ptr_hsv_ = cv_bridge::toCvCopy(last_img_, sensor_msgs::image_encodings::BGR8);
            cv::cvtColor(cv_ptr_bgr_->image,cv_ptr_hsv_->image, COLOR_BGR2HSV);
            cv::split(cv_ptr_bgr_->image,bgrPlanes_);
            cv::split(cv_ptr_hsv_->image,hsvPlanes_);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Reserve memory for max number of detections
        humanFeatures_.clear();
        humanFeatures_.reserve(maxNumHumans_);

        // Compute association likelihoods between visual detections and openpose detections
        associationMatrix_ = Eigen::MatrixXd(validDetections.size(),frame_msg_.persons.size()); // initialize matrix. obj. detections in rows, pose detections in columns
        //for ( int& sdIndex : validDetections ){
        for( int sd=0; sd < validDetections.size(); sd++) {
            for (int op=0; op < frame_msg_.persons.size(); op++){
                associationMatrix_(sd,op) = MatchLikelihood( last_detection_msg_.detections[validDetections[sd]], frame_msg_.persons[op]);
            }
        }
        // std::cout << "Association Matrix: " << std::endl;
        // std::cout << associationMatrix_ << std::endl;

        // Find match indices
        // TODO consider the general case; currently assumes spatial detections < op detections
        //associationIndices_.resize(validDetections.size());
        Eigen::MatrixXf::Index maxIndex[validDetections.size()];
        Eigen::VectorXf maxVal(validDetections.size());
        for (int ii=0; ii < validDetections.size(); ++ii ) {
            maxVal(ii) =  associationMatrix_.row(ii).maxCoeff(&maxIndex[ii]);
            std::cout << (maxIndex[ii]) << std::endl;
            //associationIndices_(ii) = (*maxIndex);
        }

        // std::cout << "Max Index" << std::endl;
        // std::cout << (*maxIndex) << std::endl;
        // std::cout << associationIndices_ << std::endl;
        // std::cout << "Max Values" << std::endl;
        // std::cout << maxVal << std::endl;

        // Create empty ROS message
        sf_array_msg_.header = last_detection_msg_.header;
        sf_array_msg_.feature_objects.clear();

        for (int match=0; match < validDetections.size(); match++)
        {

            // Generate unique color for this match
            cv::Scalar color = CV_RGB(0,255,255);

            // Visualize bounding boxes and keypoints, if visualization specified
            distance_to_det_ = DistanceToDetection(last_detection_msg_.detections[validDetections[match]]);

            // Initialize person
            ros_openpose::Person person = frame_msg_.persons[maxIndex[match]];

            // Initialize a feature vector and image for this person
            tempFeature_ = cv::Mat::zeros(histSize_*nFeat_*nKeypoints_, 1,CV_32F);
            featHistImg_ = cv::Mat( histHeight_, histWidth_, CV_8UC3, cv::Scalar(0,0,0) );

            float range[] = { 0, 256 }; //the upper boundary is exclusive
            const float* histRange[] = { range };
            
            // Initialize 
            sf_msg_ = ros_body_recognition::SpatialFeature();
            sf_msg_.pose.position.x =  last_detection_msg_.detections[validDetections[match]].position.x;
            sf_msg_.pose.position.y = -last_detection_msg_.detections[validDetections[match]].position.y;
            sf_msg_.pose.position.z =  last_detection_msg_.detections[validDetections[match]].position.z;

            for (int kp=0; kp < nKeypoints_; kp++)
            {

                if (IsValidBodyPart(person.bodyParts[kpIndices_[kp]]))
                {
                    // Compute bounding box pixel locations
                    float xmax = std::min((int)(person.bodyParts[kpIndices_[kp]].pixel.x + kp_patch_width_/(2*distance_to_det_)), cv_ptr_bgr_->image.cols);
                    float xmin = std::max((int)(person.bodyParts[kpIndices_[kp]].pixel.x - kp_patch_width_/(2*distance_to_det_)), 0);
                    float ymax = std::min((int)(person.bodyParts[kpIndices_[kp]].pixel.y + kp_patch_width_/(2*distance_to_det_)), cv_ptr_bgr_->image.rows);
                    float ymin = std::max((int)(person.bodyParts[kpIndices_[kp]].pixel.y - kp_patch_width_/(2*distance_to_det_)), 0);
                    //cv::Rect bpRect((xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymin+ymax);
                    cv::Rect bpRect(xmin, ymin, xmax-xmin, ymax-ymin);
                    
                    // Compute keypoint region mask
                    cv::Mat mask = cv::Mat::zeros(cv_ptr_bgr_->image.size(), CV_8U);
                    mask(bpRect) = cv::Scalar(255);

                    // Find histogram indices of this keypoint
                    cv::Rect bIndices(0,bIndex*histSize_ + kp*nFeat_*histSize_,1,histSize_);
                    cv::Rect gIndices(0,gIndex*histSize_ + kp*nFeat_*histSize_,1,histSize_);
                    cv::Rect rIndices(0,rIndex*histSize_ + kp*nFeat_*histSize_,1,histSize_);
                    cv::Rect hIndices(0,hIndex*histSize_ + kp*nFeat_*histSize_,1,histSize_);
                    cv::Rect sIndices(0,sIndex*histSize_ + kp*nFeat_*histSize_,1,histSize_);
                    cv::Rect vIndices(0,vIndex*histSize_ + kp*nFeat_*histSize_,1,histSize_);
                    // featHist(bIndices)

                    // Compute feature histograms of this region
                    cv::calcHist(&bgrPlanes_[0],1,0,mask,tempFeature_(bIndices),1,&histSize_,histRange,uniform_,accumulate_); 
                    cv::calcHist(&bgrPlanes_[1],1,0,mask,tempFeature_(gIndices),1,&histSize_,histRange,uniform_,accumulate_);
                    cv::calcHist(&bgrPlanes_[2],1,0,mask,tempFeature_(rIndices),1,&histSize_,histRange,uniform_,accumulate_);
                    cv::calcHist(&hsvPlanes_[0],1,0,mask,tempFeature_(hIndices),1,&histSize_,histRange,uniform_,accumulate_); 
                    cv::calcHist(&hsvPlanes_[1],1,0,mask,tempFeature_(sIndices),1,&histSize_,histRange,uniform_,accumulate_);
                    cv::calcHist(&hsvPlanes_[2],1,0,mask,tempFeature_(vIndices),1,&histSize_,histRange,uniform_,accumulate_); 
                    // cv::calcHist(&bgrPlanes_[2],1,0,cv::Mat(),rHist_,1,&histSize_,histRange,uniform_,accumulate_);

                    // Compute each feature for this region
                    // std::cout << "rows in tempFeature_: " << tempFeature_.rows << std::endl;
                    // std::cout << "cols in tempFeature_: " << tempFeature_.cols << std::endl;
                    
                    // TODO normalize by number pf pixels in keypoint patch, result is between 0-1
                    float kpArea = (xmax-xmin)*(ymax-ymin);
                    cv::divide(tempFeature_, cv::Scalar(kpArea),tempFeature_,1,-1);
                    //cv::normalize(tempFeature_, tempFeature_, 0, featHistImg_.rows, cv::NORM_MINMAX, -1, cv::Mat() );
                    //cv::normalize(tempFeature_, tempFeature_, 0, 256, cv::NORM_MINMAX, -1, cv::Mat() );
                    
                    // cv::normalize(bHist_, bHist_, 0, rgbHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );
                    // cv::normalize(gHist_, gHist_, 0, rgbHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );
                    // cv::normalize(rHist_, rHist_, 0, rgbHistImage_.rows, cv::NORM_MINMAX, -1, cv::Mat() );


                    // Draw box around keypoint
                    DrawBpBox(cv_ptr_bgr_->image, xmin, xmax, ymin, ymax, color);
                    
                    // if (visualize){
                    //     cv::imshow(DEVEL_WINDOW, mask);
                    //     cv::waitKey(1);
                    // }

                    for( int i = 1; i < histSize_; i++ )
                    {
                        int iBlue = i + bIndex*histSize_ + kp*nFeat_*histSize_; // 1-15
                        int iGrn = i + gIndex*histSize_ + kp*nFeat_*histSize_; // 17-31
                        int iRed = i + rIndex*histSize_ + kp*nFeat_*histSize_;
                        int iHue = i + hIndex*histSize_ + kp*nFeat_*histSize_;
                        int iSat = i + sIndex*histSize_ + kp*nFeat_*histSize_;
                        int iVal = i + vIndex*histSize_ + kp*nFeat_*histSize_;

                        // std::cout << "iBlue: " << iBlue << ", IGrn: " << iGrn << std::endl;
                        cv::line( featHistImg_, 
                            cv::Point( bin_*(iBlue-1), histWidth_ - cvRound(featHistImg_.rows*tempFeature_.at<float>(iBlue-1)) ),
                            cv::Point( bin_*(iBlue), histHeight_ - cvRound(featHistImg_.rows*tempFeature_.at<float>(iBlue)) ),
                            cv::Scalar( 255, 0, 0), 2, 8, 0  );
                        cv::line( featHistImg_, 
                            cv::Point( bin_*(iGrn-1), histWidth_ - cvRound(featHistImg_.rows*tempFeature_.at<float>(iGrn-1)) ),
                            cv::Point( bin_*(iGrn), histHeight_ - cvRound(featHistImg_.rows*tempFeature_.at<float>(iGrn)) ),
                            cv::Scalar( 0, 255, 0), 2, 8, 0  );
                        cv::line( featHistImg_, 
                            cv::Point( bin_*(iRed-1), histWidth_ - cvRound(featHistImg_.rows*tempFeature_.at<float>(iRed-1)) ),
                            cv::Point( bin_*(iRed), histHeight_ - cvRound(featHistImg_.rows*tempFeature_.at<float>(iRed)) ),
                            cv::Scalar( 0, 0, 255), 2, 8, 0  );
                        cv::line( featHistImg_, 
                            cv::Point( bin_*(iHue-1), histWidth_ - cvRound(featHistImg_.rows*tempFeature_.at<float>(iHue-1)) ),
                            cv::Point( bin_*(iHue), histHeight_ - cvRound(featHistImg_.rows*tempFeature_.at<float>(iHue)) ),
                            cv::Scalar( 255, 255, 0), 2, 8, 0  );
                        cv::line( featHistImg_, 
                            cv::Point( bin_*(iSat-1), histWidth_ - cvRound(featHistImg_.rows*tempFeature_.at<float>(iSat-1)) ),
                            cv::Point( bin_*(iSat), histHeight_ - cvRound(featHistImg_.rows*tempFeature_.at<float>(iSat)) ),
                            cv::Scalar( 0, 255, 255), 2, 8, 0  );
                        cv::line( featHistImg_, 
                            cv::Point( bin_*(iVal-1), histWidth_ - cvRound(featHistImg_.rows*tempFeature_.at<float>(iVal-1)) ),
                            cv::Point( bin_*(iVal), histHeight_ - cvRound(featHistImg_.rows*tempFeature_.at<float>(iVal)) ),
                            cv::Scalar( 255, 0, 255), 2, 8, 0  );
                    }

                } // keypoint valid

            } // Keypoint

            // TODO add features to vector
            sf_msg_.features = tempFeature_;

            // Visualization
            if (visualize) { DrawDetectionBox(cv_ptr_bgr_->image, last_detection_msg_.detections[validDetections[match]], color); };

            // Add this human's features to the array
            humanFeatures_.push_back(tempFeature_);
            sf_array_msg_.feature_objects.push_back(sf_msg_);

            // Plot feature histogram

        } // SD/OP match

        feat_pub_.publish(sf_array_msg_);


        // VISUALIZE
        // Update Image Window
        if (visualize){
            cv::imshow(OPENCV_WINDOW, cv_ptr_bgr_->image);
            cv::waitKey(1);

            // Update feature histogram window
            cv::imshow(FEAT_HIST_WINDOW, featHistImg_);
            cv::waitKey(1);
        }

        
        control_state_ = READY;
    } // extractFeatures

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
