#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <ctime>
#include <signal.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

using namespace cv;
using namespace std;

//string point_cloud_frame_id = "";
//ros::Time point_cloud_time;

//Task kill flag
static volatile int keepRunning = 1;
//static volatile double xscale = 0.003787727;
//static volatile double yscale = 0.00419224;

//Initialize the Mats needed for calculations
static Mat frame;
static Mat HSV;
static Mat threshld;
static Mat Gaussian;
static Mat Can;
static Mat ctv;
static Mat Hough;
static Mat output;

//Vector that holds the detected lines
static vector<Vec4i> lines;

//The HSV ranges
static int H_MIN, H_MAX, S_MIN, S_MAX, V_MIN, V_MAX;

//cnt+c handeler (may not be needed with ros.spin())
void intHandler(int dummy) {
	keepRunning = 0;
}

//ROS Comms initialization
class ImageConverter {
	//Initialize node handeler
	ros::NodeHandle nh_;
	
	//Set up subscriber
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	cv_bridge::CvImagePtr cv_ptr;

public:
	//Subscribe to the ROS topic
	ImageConverter():it_(nh_) {
		// Subscribe to input video feed and publish output video feed
		image_sub_ = it_.subscribe("/zed/rgb/image_rect_color", 1,
		&ImageConverter::imageCb, this);
	}

	//Main image transform function
	void imageCb(const sensor_msgs::ImageConstPtr& msg) {
		
		//*Initialization Phase*//
		
		//Try to convert
		try {
			cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		//Catch errors and print to screen
		} catch (cv_bridge::Exception& e) {
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}
		

		//Get the current frame
		frame = cv_ptr->image;

		//*Perspective shift phase*//
		
		//Initialize perspective shift quadralaterals
		Point2f inputQuad[4];
		Point2f outputQuad[4];

		// Lambda Matrix
		Mat lambda(2, 4, CV_32FC1);

		// Set the lambda matrix the same type and size as input
		lambda = Mat::zeros(frame.rows, frame.cols, frame.type());

		// The 4 points that select quadilateral on the input , from top-left in clockwise order
		// These four pts are the sides of the rect box used as input
		inputQuad[0] = Point2f(100, 360);
		inputQuad[1] = Point2f(1180, 360);
		inputQuad[2] = Point2f(1940, 720);
		inputQuad[3] = Point2f(-660, 720);
		// The 4 points where the mapping is to be done , from top-left in clockwise order
		outputQuad[0] = Point2f(0, 0);
		outputQuad[1] = Point2f(frame.cols, 0);
		outputQuad[2] = Point2f(frame.cols, frame.rows);
		outputQuad[3] = Point2f(0, frame.rows);

		// Get the Perspective Transform Matrix i.e. lambda
		lambda = getPerspectiveTransform(inputQuad, outputQuad);
		// Apply the Perspective Transform just found to the src image
		warpPerspective(frame, output, lambda, output.size());

		//*Discovery Phase*//
		//Convert the frame to different kinds of Mats
		cvtColor(output, HSV, COLOR_BGR2HSV);
		inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), threshld);
		inRange(HSV, Scalar(0, 0, 0), Scalar(0, 0, 0), Hough);
		morphOps(threshld);

		//Detect Edges
		Canny(threshld, Can, 50, 200, 3);

		//Convert to black and white
		cvtColor(Can, ctv, CV_GRAY2BGR);

		//Detect lines
		HoughLinesP(Can, lines, 1, CV_PI/180, 50, 20, 10);

		//*Data conversion phase*//
		int index = 0;
		//For all points
		for (size_t i = 0; i < lines.size(); i++) {
			//Convert take one point out of the array
			Vec4i l = lines[i];
			line(ctv, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
		}
		
		//*Show Phase*//
		//Make and populate the windows that show the Mats
		namedWindow( "Input Image", WINDOW_NORMAL );
		imshow("Input Image", frame);
		waitKey(25);
			
		namedWindow( "Color Filtering", WINDOW_NORMAL );
		imshow("Color Filtering", threshld);
		waitKey(25);
			
		namedWindow( "Line Tracking", WINDOW_NORMAL );
		imshow("Line Tracking", ctv);
		waitKey(25);
		
		//Initialize the sliders
		char HMINSlide[50];
		char HMAXSlide[50];
		char SMINSlide[50];
		char SMAXSlide[50];
		char VMINSlide[50];
		char VMAXSlide[50];
		sprintf( HMINSlide, "H_MIN");
		sprintf( HMAXSlide, "H_MAX");
		sprintf( SMINSlide, "S_MIN");
		sprintf( SMAXSlide, "S_MAX");
		sprintf( VMINSlide, "V_MIN");
		sprintf( VMAXSlide, "V_MAX");
 
		//Open a window for the sliders and populate
		namedWindow( "Calibration", WINDOW_NORMAL );
		createTrackbar( HMINSlide, "Calibration", &H_MIN, 255);
		createTrackbar( HMAXSlide, "Calibration", &H_MAX, 255);
		createTrackbar( SMINSlide, "Calibration", &S_MIN, 255);
		createTrackbar( SMAXSlide, "Calibration", &S_MAX, 255);
		createTrackbar( VMINSlide, "Calibration", &V_MIN, 255);
		createTrackbar( VMAXSlide, "Calibration", &V_MAX, 255);
	}

	void morphOps(Mat &thresh) {
		//create structuring element that will be used to "dilate" and "erode" image.
		//the element chosen here is a 3px by 3px rectangle

		Mat erodeElement = getStructuringElement(MORPH_RECT, Size(2, 2));
		//dilate with larger element so make sure object is nicely visible
		Mat dilateElement = getStructuringElement(MORPH_RECT, Size(3, 3));

		//Erode the image
		//erode(thresh, thresh, erodeElement);
		//erode(thresh, thresh, erodeElement);

		//Dilate the image
		dilate(thresh, thresh, dilateElement);
		dilate(thresh, thresh, dilateElement);
	}

	cv_bridge::CvImagePtr getImage() {
		return cv_ptr;
	}
};

int main(int argc, char** argv) {
	//Get the file with HSV values and inport them
	std::fstream HSVfile("LineTrackingFiles/HSV.txt", std::ios_base::in);
	HSVfile >> H_MIN >> H_MAX >> S_MIN >> S_MAX >> V_MIN >> V_MAX;
	HSVfile.close();
	
	//Initialize signal handeler
	signal(SIGINT, intHandler);
	
	//Initialize topic
	ros::init(argc, argv, "image_converter");
	ImageConverter ic;

	//Spin until cntr+c
	ros::spin();
	
	//When the exit signal comes, save the HSV values to the file
	ofstream HSVfiles;
	HSVfiles.open("LineTrackingFiles/HSV.txt");
	HSVfiles << H_MIN << " " << H_MAX << " " << S_MIN << " " << S_MAX << " " << V_MIN << " " << V_MAX;
			
	//End the program
	return 0;
}
