//*********************************************************************************
// class for upperbody detection
// input: video frames
// output: detection rectangles
//
//*********************************************************************************
#ifndef UPPERBODYDETECTOR_H
#define UPPERBODYDETECTOR_H
#define UPPERDETECTOR_TEST_
#define UPPERDETECTOR_DEBUG_

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <stdio.h>
#include <array>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

#define LINE_AA 16   //line stype for putText

#define THRESHOLD_PREV_OVERLAP 0.65
//#define THRESHOLD_MASKMNS 0.5

typedef array<int,7> detectarray;
typedef std::vector<array<int,7>> vector2d;

class UpperbodyDetector
{
private:   
    const int kMinFaceSize = 40;
    const int kNetKind = 10;

    CascadeClassifier upperbodyhaar_;
    const double kScaleFactor = 1.2;

    const bool kMultiScale  = true;
    const int kStride  = 3;   
    const char *kCascadeUpBodyName = "/home/ying/anaconda2/share/OpenCV/haarcascades/haarcascade_mcs_upperbody.xml";
    const float kThresholdNms = 0.5;         //THRESHOLD_MASKMNS 0.5
    //const float kThresholdOverlap = 0.65;  //THRESHOLD_PREV_OVERLAP 0.65
    const int kFlag =  0 | CASCADE_SCALE_IMAGE;
    const int kMinNeighbors= 3;  //2

    bool b_rescale_ = false;
    float frame_scale_;
    int width_;
    int height_;
    cv::Size min_size_;
    cv::Size detection_size_;

#ifdef UPPERDETECTOR_DEBUG_
    bool b_show_record_;
    const int kOutFont = FONT_HERSHEY_SIMPLEX;
    int v_codecs_;
    int fps_;
    cv::VideoWriter output_video_;
    String output_video_name_;
#endif

    float FindInitialScale();
    size_t CalculateRectArea(const cv::Rect &rect);
    size_t CalculateRectArea(const detectarray &rectarr);
    void DetectCascade(CascadeClassifier &classifier, const cv::Mat &gray, std::vector<Rect> &regions); // (unfinished) handle b_rescale_ = true;

#ifdef UPPERDETECTOR_DEBUG_
    void DrawRectangleOnFrame(cv::Mat &frame, const vector2d &detectedrects, const cv::Scalar color=cv::Scalar(0,255,0));
    void InitRecorder();
    void ReleaseDisplayRecorder();
#endif

public:
    UpperbodyDetector(int frame_width, int frame_height, bool b_do_rescale=false);
    ~UpperbodyDetector();
    void Run(const cv::Mat &current_frame_input, const int curr_frame_index, vector2d &rectangles);   
    vector2d MaskNms(const vector2d &rectangles, const float threshold_nms, cv::Size frame_size_);
    int CheckPrevFrameRectNoMask(const vector2d &new_rects, const vector2d &prev_rects, vector2d &rect_noprev,float threshold=THRESHOLD_PREV_OVERLAP);

#ifdef UPPERDETECTOR_DEBUG_
    void OpenDisplayRecorder();
    void CloseDisplayRecorder();
    void DoDisplayRecord(const cv::Mat &frame, const int frame_index, const vector2d &rectangles);
#endif
};

#endif // UPPERBODYDETECTOR_H
