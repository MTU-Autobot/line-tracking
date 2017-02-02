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

//typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

string point_cloud_frame_id = "";
ros::Time point_cloud_time;

static volatile int keepRunning = 1;
static volatile double xscale = 0.003787727;
static volatile double yscale = 0.00419224;
static Mat frame;
static Mat HSV;
static Mat threshld;
static Mat Gaussian;
static Mat Can;
static Mat ctv;
static Mat Hough;
static Mat output;
static vector<Vec4i> lines;
static int H_MIN, H_MAX, S_MIN, S_MAX, V_MIN, V_MAX;

void intHandler(int dummy) {
	keepRunning = 0;
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

class ImageConverter {
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	cv_bridge::CvImagePtr cv_ptr;
	ros::Publisher pub_cloud = nh_.advertise<sensor_msgs::PointCloud2> ("lines", 1);
	//ros::Publisher pub = nh_.advertise<PointCloud> ("points", 1);

public:
	ImageConverter():it_(nh_) {
		// Subscrive to input video feed and publish output video feed
		image_sub_ = it_.subscribe("/camera/rgb/image_rect_color", 1,
		&ImageConverter::imageCb, this);
	}

	void imageCb(const sensor_msgs::ImageConstPtr& msg) {
		try {
			cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		} catch (cv_bridge::Exception& e) {
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
		}

		//**Initialization Phase
		//Initialize the tags

		////polysync::message::LiDARPointStream pointStream;

		//Get the file with HSV values and inport them
		std::fstream HSVfile("LineTrackingFiles/HSV.txt", std::ios_base::in);
		HSVfile >> H_MIN >> H_MAX >> S_MIN >> S_MAX >> V_MIN >> V_MAX;

		//Get the current frame
		frame = cv_ptr->image;

		//**Perspective shift phase
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

		//**Discovery Phase
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

		//**Data conversion phase
		//Begin writing point stream
		////pointStream.setStreamLength(2 * lines.size());
		//PointCloud::Ptr cloud (new PointCloud);
		//cloud->header.frame_id = "some_tf_frame";
		pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
		point_cloud.width = 1280;
		point_cloud.height = 720;
		int size = 1280*720;
		point_cloud.points.resize(size);

		int index = 0;
		//For all points
		for (size_t i = 0; i < lines.size(); i++) {
			//Convert take one point out of the array
			Vec4i l = lines[i];
			line(ctv, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);

			//Calculate distances
			double p0 = ((yscale) * ((l[0]) - 320)) + 0.17;
			double p1 = (xscale * (480 - (l[1]))) + 0.7366;
			double p2 = ((yscale) * ((l[2]) - 320)) + 0.17;
			double p3 = (xscale * (480 - (l[3]))) + 0.7366;

			//Add points to stream
			////pointStream.streamPushBack(
			////	{ LAYER_NONE, ECHO_NONE, POINT_NONE, 0.0, { p1, p0, 5.0 } }
			////);
			////pointStream.streamPushBack(
			////	{ LAYER_NONE, ECHO_NONE, POINT_NONE, 0.0, { p3, p2, 5.0 } }
			////);
			//cloud->points.push_back (pcl::PointXYZ(p1, p0, 5.0));
			//cloud->points.push_back (pcl::PointXYZ(p3, p2, 5.0));
			point_cloud.points[index].y = p1;
			point_cloud.points[index].z = 5.0;
			point_cloud.points[index].x = p0;
			index++;
			point_cloud.points[index].y = p3;
			point_cloud.points[index].z = 5.0;
			point_cloud.points[index].x = p2;

			//Print the points to the screen
			cout << p1 << " " << p0 << endl;
			cout << p3 << " " << p2 << endl;
		}
		//Space things out
		cout << endl;

		//**Polysync send phase
		//Publish the message to PolySync
		sensor_msgs::PointCloud2 output;
		pcl::toROSMsg(point_cloud, output); // Convert the point cloud to a ROS message
		output.header.frame_id = point_cloud_frame_id; // Set the header values of the ROS message
		output.header.stamp = point_cloud_time;
		output.height = 1280;
		output.width = 720;
		output.is_bigendian = false;
		output.is_dense = false;
		pub_cloud.publish(output);
		////polysync::message::publish(pointStream);
		//cloud->header.stamp = ros::Time::now().toNSec();
		//pub.publish (cloud);
		//Wait for a second before we repeat
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
	signal(SIGINT, intHandler);
	ros::init(argc, argv, "image_converter");
	ImageConverter ic;


	//Open the camera stream
	//Start the camera
	//VideoCapture webcam << ic.getImage();

	//Initialize and connect to PolySync.
	////polysync::Node node("CameraRPublisher");

	//Loop until we get a ctrl-c

	ros::spin();
	//When the exit signal comes, save the logs

	//Find HSV of point for calibration purposes
	cout << "Finding HSV..." << endl;
	Mat RGB = frame(Rect(562, 270, 1, 1));
	Mat detector;
	cvtColor(RGB, detector, CV_BGR2HSV);
	cv::Vec3b pixel = detector.at<cv::Vec3b>(0, 0);
	int H = pixel.val[0];
	int S = pixel.val[1];
	int V = pixel.val[2];
	cout << H * 1 << " " << S * 1 << " " << V * 1 << endl;

	//Save all the images to files
	cout << "Saving images..." << endl;
	imwrite("LineTrackingFiles/frame.jpg", frame);
	imwrite("LineTrackingFiles/HSV.jpg", HSV);
	imwrite("LineTrackingFiles/threshld.jpg", threshld);
	imwrite("LineTrackingFiles/hough.jpg", ctv);
	imwrite("LineTrackingFiles/output.jpg", output);
	imwrite("LineTrackingFiles/can.jpg", Can);

	//Save text files of all of the points
	cout << "Saving points..." << endl;

	//Start the files
	ofstream pixels;
	ofstream points;
	pixels.open("LineTrackingFiles/pixel.txt");
	points.open("LineTrackingFiles/point.txt");

	//Run through the points and save them to a file
	for (size_t i = 0; i < lines.size(); i++) {
		//Get points for the array
		Vec4i l = lines[i];
		line(ctv, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);

		//Convert to meters
		double p0 = ((yscale) * ((l[0]) - 320)) + 0.17;
		double p1 = (xscale * (480 - (l[1]))) + 0.7366;
		double p2 = ((yscale) * ((l[2]) - 320)) + 0.17;
		double p3 = (xscale * (480 - (l[3]))) + 0.7366;

		//Save the points to a file
		pixels << l[1] << " , " << l[0] << "\r\n";
		points << p1 << " , " << p0 << "\r\n";
		pixels << l[3] << " , " << l[2] << "\r\n";
		points << p3 << " , " << p2 << "\r\n";
	}

	//Close the files
	pixels.close();
	points.close();

	//End the program
	return 0;
}
