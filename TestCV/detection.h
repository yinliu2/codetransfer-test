//*********************************************************************************
// main header for face detection
//
//*********************************************************************************

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <QDebug>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <array>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

typedef array<int,7> detectarray;
typedef std::vector<array<int,7>> vector2d;

#define THRESHOLDNMS_DN_1  0.5  //for descending order
#define THRESHOLDNMS_DN_2  0.7
#define THRESHOLDNMS_UP  0.8  //for ascending
#define LINE_AA 16   //line stype for putText
#define THRESHOLD_INFG   0.15
#define THRESHOLD_RECTOVERLAP  0.65
#define THRESHOLDOVERLAP_1 0.65
#define THRESHOLDOVERLAP_2 0.85

static const String input_folder_ = "./surveillance/multi_face_turning/";
static const String data_folder_root_str_ = "./";
static const String data_folder_str_ = "surveillance/";
static const String curr_file_index_str_ = "02";

// recording parameters
static const int kOutFont = FONT_HERSHEY_SIMPLEX;
static int v_codecs = CV_FOURCC('M', 'J', 'P', 'G');
static const int kFps = 4;
static const int outfont = FONT_HERSHEY_SIMPLEX;
// const int kLineType = 16;

static const String upper_output_video_name = "upper_output.avi";
static const String fg_output_video_name = "fgoutput.avi";
static const String mask_video_name = "fgmask_output.avi";
static const String final_output_video_name = "final_output.avi";

cv::VideoWriter upper_output_video;
cv::VideoWriter fg_output_video;
cv::VideoWriter mask_video;
cv::VideoWriter final_output_video;

// video setting
static const size_t frame_width_ = 720;
static const size_t frame_height_ = 576;
static const cv::Size frame_size_ = cv::Size(frame_width_, frame_height_);
static const size_t frame_area_ = frame_width_*frame_height_;
static const size_t current_scale_ = 1;  //default

// detection setting
static const int fg_min_face_size_ = 56;
static const size_t fg_threshold_ =12;
static const bool b_upper_rescale_ = false;

// initialize the current frame buffer, the image shown buffer, and the foreground mask buffer
static cv::Mat fgmask = cv::Mat::zeros(frame_height_, frame_width_, CV_8U);
static cv::Mat fgmask_rgb = cv::Mat::zeros(frame_height_, frame_width_, CV_8UC3);
static cv::Mat curr_frame_org = cv::Mat::zeros(frame_height_, frame_width_, CV_8UC3);
static cv::Mat curr_frame_4show = cv::Mat::zeros(frame_height_, frame_width_, CV_8UC3);
