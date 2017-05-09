//*********************************************************************************
// class for forground detection
// input: video frames
// output: detection rectangles
//
//*********************************************************************************
#ifndef MOTIONDETECTOR_H
#define MODTIONDETECTOR_H
#define FGDETECTOR_TEST_
#define FGDETECTOR_DEBUG_


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
#include <string>

#include <fstream>
#include <ctime>

using namespace std;
using namespace cv;

#define MAX_BINARY_VALUE 255
#define LINE_AA 16   //line stype for putText

typedef array<int,7> detectarray;
typedef std::vector<array<int,7>> vector2d;


class MotionDector
{
private:
    const bool b_motion_exist_ = true;
    size_t current_scale_;
    size_t frame_pixel_num_;
    int min_face_size_;
    bool b_show_record_;
    cv::Ptr<cv::BackgroundSubtractorMOG2> pMog2_; //cv::Ptr<cv::BackgroundSubtractorMOG2> pMog2_;BackgroundSubtractor
    const bool kMogDetectShadows = false;
    const float kMog2LearningRate = 0.01;  //0.01, The const qualifier demands that the variable not be changed after its initialization
    vector<vector<Point> > contours_;
    int frame_num_;
    static size_t width_;
    static size_t height_;
    static size_t cnt_threshold_;
    static Mat frame1gray_;
    static Mat frame2gray_;
    static Mat framediff_;
    static Mat fgmask_;
    static Mat fgmask_copy_;

#ifdef FGDETECTOR_DEBUG_
    const int kOutFont = FONT_HERSHEY_SIMPLEX;
    int v_codecs_;
    int fps_;
    cv::VideoWriter output_video_;
    cv::VideoWriter mask_video_;
    String output_video_name_;
    String mask_video_name_;

    static Mat fgmask_rgb_;
#endif

    void ProcessImage(const cv::Mat &frame);
    bool IsSomethingMoved();
    size_t  CalculateRectArea(const cv::Rect &rect);
    size_t  CalculateRectArea(const detectarray &rectarr);
    float CalculateIoU(const detectarray &rect_1, const detectarray &rect_2);
    float CalculateIoM(const detectarray &rect_1, const detectarray &rect_2);
    void LocalNms(vector2d &rectangles);

#ifdef FGDETECTOR_DEBUG_
    void DrawRectangleOnFrame(cv::Mat &frame, const vector2d &detectedrects, const cv::Scalar color=cv::Scalar(0,255,0));
    static void ChangeParam(int v_new_threshold, void *);
    void InitRecorder();
    void ReleaseDisplayRecorder();
#endif

public:
    MotionDector(int frame_width, int frame_height, int v_threshold, int v_minSize=56); //bool b_show_record_input=false
    ~MotionDector();
    void InitilizeFirstFrame(const cv::Mat &first_frame);
    void Run(const cv::Mat &curr_frame_input, const int curr_frame_index, vector2d &rectangles);
    void GetFgMask(cv::Mat &fgmask_mat);
    void CleanFgMaskBuf();
    int CheckPrevRectsOnFg(const vector2d &fg_rects, const vector2d &prev_rects, vector2d &rect_outfg, const cv::Mat &fg_mask,\
                            const float threshold_overlap, const float threshold_infg);

#ifdef FGDETECTOR_DEBUG_
    void OpenDisplayRecorder();
    void CloseDisplayRecorder();
    void DoDisplayRecord(const cv::Mat &frame, const int frame_index, const vector2d &rectangles);
#endif
};


#endif // MOTIONDETECTOR_H
