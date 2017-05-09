#include "upperbodydetector.h"

#ifdef UPPERDETECTOR_DEBUG_
// //////////////////////////////////////////////
// Initialize and open recorders for dector test
// param : open recorder is ture/false
// return: no
// //////////////////////////////////////////////
void UpperbodyDetector::InitRecorder()
{
    if(b_show_record_)
    {
        v_codecs_ = CV_FOURCC('M', 'J', 'P', 'G');
        fps_ = 4;
        output_video_name_ = "upperclass_output.avi";
        cv::Size s = cv::Size(width_, height_);
        output_video_.open(output_video_name_, v_codecs_, fps_, s); // For writing the detection video
    }
}

// ////////////////////////////////////////////////////////////////
// Display and record foreground detection result for current frame
// ////////////////////////////////////////////////////////////////
void UpperbodyDetector::DoDisplayRecord(const cv::Mat &frame, const int frame_index, const vector2d &rectangles)
{
    if (b_show_record_)
    {
        // draw detected rectangles on the current frame
        if (!rectangles.empty())
        {
            String str = "frame# = " + to_string(frame_index);
            static cv::Mat frame_4show = frame.clone();

            DrawRectangleOnFrame(frame_4show, rectangles); // draw rectangles imageframe

            putText(frame_4show, str, cv::Point(25,30), kOutFont, 1, cv::Scalar(0,0,0), 1, LINE_AA); // put date on the frame cv::LINE_AA=16
            output_video_.write(frame_4show);
            cv::imshow("fg class: result", frame_4show);
            waitKey(1);

        }
    }

}
// ////////////////////////////////////////////////////////////////
// close image show windows and release recorders for detector test
// ////////////////////////////////////////////////////////////////
void UpperbodyDetector::ReleaseDisplayRecorder()
{
    if (b_show_record_)
    {
        destroyAllWindows();
        waitKey(1);
        if (output_video_.isOpened())
        {
            output_video_.release();
        }

    }

}

// ////////////////////////////////////////////////////////////////
// enable display and recorder
// ////////////////////////////////////////////////////////////////
void UpperbodyDetector::OpenDisplayRecorder()
{
    b_show_record_ = true;
    InitRecorder();
}

// ////////////////////////////////////////////////////////////////
// disable display and recorder and release related objects
// ////////////////////////////////////////////////////////////////
void UpperbodyDetector::CloseDisplayRecorder()
{
    b_show_record_ = false;
    ReleaseDisplayRecorder();
}

// ////////////////////////////////////
// draw rectangles on image frame
// function overloadding for cv:Rect input
// param : rectangle reference
// return: calculated total region pixel number
// ////////////// //////////////////////
// draw rectangles imageframe
void UpperbodyDetector::DrawRectangleOnFrame(cv::Mat &frame, const vector2d &detectedrects, const cv::Scalar color) //const vector2d &prev_rects,
{
    //static const CvScalar bluecolor = cv::Scalar(255,   0,   0);
    static const int thickness = 4;
    static const int linekind = 8;
    static const int shift = 0;
    static size_t rectcount = detectedrects.size() > 0 ? detectedrects.size():0;

    for (size_t i = 0; i < rectcount; ++i)
    {
        const detectarray &rect = detectedrects[i];

        // draws a rectangle with given coordinates of the upper left
        // and lower right corners into an image
        CvPoint ul, lr;
        ul.x = rect[0];
        ul.y = rect[1];
        lr.x = rect[2];
        lr.y = rect[3];

        cv::rectangle(frame, ul, lr, color, thickness, linekind, shift);  // use C++ API instead of C API. Use the rectangle function in the "cv::Rectangle" namespace instead of "cvRectangle":

    }
}
#endif

UpperbodyDetector::UpperbodyDetector(int frame_width, int frame_height, bool b_do_rescale) //: width(frame_width), height(frame_height)
{
    b_rescale_ = b_do_rescale;
    width_  = frame_width;
    height_ = frame_height;

    //load upperbody detector
    upperbodyhaar_.load(kCascadeUpBodyName);
    const bool detect_status = !upperbodyhaar_.empty();

    if(!detect_status)
    {
        cout<<"failed to load upper body CascadeClassifier"<<endl;
        exit (EXIT_FAILURE);
    }

    //initialization based on the determined scale
    if(!b_rescale_ ) // if b_rescale_ = false
    {
        min_size_       = cv::Size(kMinFaceSize, kMinFaceSize);
        frame_scale_    = 1;
        detection_size_ = cv::Size(width_, height_);//rescale issue has been considered in detection_size_
    }
    else // downsample rate = 4
    {
        min_size_      = cv::Size(kNetKind, kNetKind);   // for b_scale = true
        frame_scale_   = FindInitialScale();             // find initial scale
        int height_scaled = int(height_ / frame_scale_);    // resized new height
        int width_scaled  = int(width_ /  frame_scale_);    // resized new width
        detection_size_ = cv::Size(width_scaled, height_scaled);
    }

}


UpperbodyDetector::~UpperbodyDetector(){}

// ////////////////////////////////////
// param net_kind: what kind of net (12, 24, or 48)
// param kMinFaceSize: minimum face size
// return:    returns scale factor
// ////////////////////////////////////
float UpperbodyDetector::FindInitialScale()
{
    return float(kMinFaceSize) / kNetKind;
}

// ////////////////////////////////////
// calculate region area by pixel number
// function overloadding for cv:Rect input
// param :
// return: calculated total region pixel number
// ////////////////////////////////////
size_t  UpperbodyDetector::CalculateRectArea(const cv::Rect &rect)
{
    size_t rect_area;
    int x1, y1, x2, y2;
    x1 = rect.x;                //rect[0] first rectangle top left x
    y1 = rect.y;                //rect[1] first rectangle top left y
    x2 = rect.x + rect.width;   //rect[2] first rectangle bottom right x
    y2 = rect.y + rect.height;  //rect[3] first rectangle bottom right y
    rect_area = (y2-y1+1) * (x2-x1+1);

    return rect_area;
}

// ////////////////////////////////////
// calculate region area by pixel number
// function overloadding for face-rectangle inptut
// param :
// return: calculated total region pixel number
// ////////////////////////////////////
size_t  UpperbodyDetector::CalculateRectArea(const detectarray &rectarr)
{
    size_t rect_area;
    int x1, y1, x2, y2;
    x1 = rectarr[0];
    y1 = rectarr[1];
    x2 = rectarr[2];
    y2 = rectarr[3];
    rect_area = (y2 - y1+1) * (x2 - x1+1);

    return rect_area;
}

// ////////////////////////////////////
// select those rectangles that are overlapped with
// the previous detected regions under a threshold
// param: new detected rectangles
// param: previous detected rectangles
// param: selection threshold
// return: selected new detected rectangles
// ////// //////////////////////////////
int UpperbodyDetector::CheckPrevFrameRectNoMask(const vector2d &new_rects, const vector2d &prev_rects, vector2d &rect_noprev, float threshold)
{
    rect_noprev = {};
    cv::Mat prev_rectmask_crop;
    cv::Mat prev_rectmask = cv::Mat::zeros(height_, width_, CV_8U);
    int rect_pixel_num;
    int prev_rectmask_sum;
    float overlap_rate;
    int  x1,x2,y1,y2;

    for (size_t i=0; i<prev_rects.size(); ++i)
    {
        const detectarray &rect = prev_rects[i];
        x1 = rect[0];
        y1 = rect[1];
        x2 = rect[2];
        y2 = rect[3];
        prev_rectmask(cv::Range(y1,y2), cv::Range(x1,x2)) = 1; // mark foreground region as "1"
    }

    for (size_t i=0; i< new_rects.size(); ++i)
    {
        const detectarray &rect = new_rects[i];
        x1 = rect[0];
        y1 = rect[1];
        x2 = rect[2];
        y2 = rect[3];
        rect_pixel_num = (y2-y1+1) * (x2-x1+1);
        prev_rectmask_crop = prev_rectmask(cv::Range(y1,y2),cv::Range(x1,x2));
        prev_rectmask_sum = cv::sum(prev_rectmask_crop)[0];
        overlap_rate = float(prev_rectmask_sum)/float(rect_pixel_num);
        if(overlap_rate < threshold)
        {
            rect_noprev.push_back(rect);
        }
    }

    return rect_noprev.size();

 }

// ////////////////////////////////////
// param curr_frame: detected regions (2D vector)
// param bodie: detected uppderbody regions
// return: processed detected regions (2D vector)
// ////////////////////////////////////
vector2d  UpperbodyDetector::MaskNms(const vector2d &rectangles, const float threshold_nms, cv::Size frame_size_)
{
    //for rectangles sorted from smallest area to the largest
    vector2d rect_new ={};
    cv::Mat rect_mask_crop;
    cv::Mat detection_mask;

    detection_mask = cv::Mat::zeros(frame_size_.height, frame_size_.width, CV_8U);

    if (!rectangles.empty())
    {
        int x1_0, x2_0, y1_0, y2_0;
        int x1, x2, y1, y2;
        const detectarray &rect0= rectangles[0];
        x1_0 = rect0[0];
        y1_0 = rect0[1];
        x2_0 = rect0[2];
        y2_0 = rect0[3];
        detection_mask(cv::Range(y1_0, y2_0), cv::Range(x1_0, x2_0)) = 1;
        rect_new.push_back(rect0);

        for (size_t i=1; i< rectangles.size(); ++i)
        {
            const detectarray &rect = rectangles[i];
            x1 = rect[0];
            y1 = rect[1];
            x2 = rect[2];
            y2 = rect[3];
            rect_mask_crop = detection_mask(cv::Range(y1, y2), cv::Range(x1, x2));
            int rect_mask_crop_sum = cv::sum(rect_mask_crop)[0];
            int rect_area_num = (y2-y1+1) * (x2-x1+1);
            float overlap_rate = float(rect_mask_crop_sum)/float(rect_area_num);

            if(overlap_rate < threshold_nms) //threshold_nms=0.5
            {
                rect_new.push_back(rect);  //append current rectangle on vector<rectangle>
                detection_mask(cv::Range(y1, y2), cv::Range(x1, x2))=1;
                printf ("mask nms");
            }

        }
    }//end if (!rectangles.empty())

    return rect_new;

}

// ////////////////////////////////////
// apply upperbody detection
// ////////////////////////////////////
void UpperbodyDetector::DetectCascade(CascadeClassifier &classifier, const Mat &gray, std::vector<Rect> &regions)
{
    //static const Size maxSize;
    classifier.detectMultiScale(gray, regions, kScaleFactor, kMinNeighbors, kFlag, min_size_); // if b_rescale=true; apply resized min_size_
    //upperbodyhaar_.detectMultiScale(frame_mat_gray, body_rects, kScaleFactor, kMinNeighbors, kFlag, min_size_);
}

// ////////////////////////////////////
// Upperbody detection
// param image: image to detect faces
// param min_face_size: minimum face size to detect (in pixels)
// param stride: stride (in pixels)
// param multiScale: whether to find faces under multiple scales or not
// param scale_factor: scale to apply for pyramid
// param threshold: score of patch must be above this value to pass to next net
// return: list of rectangles after global NMS
// ////////////////////////////////////
void UpperbodyDetector::Run(const cv::Mat& current_frame, const int curr_frame_index, vector2d &rectangles)
{
    // list of rectangles [x11, y11, x12, y12, confidence, current_scale] (corresponding to original image)
    static std::vector<cv::Rect> body_rects={};
    rectangles={}; // clear result list
    static cv::Mat frame_mat_gray;
    static cv::Mat current_frame_mat;

    std::cout <<"upperbody detecion ..." << std::endl;

    if(!current_frame.empty())
    {
        // First resize the input frame (skip resize in this version)
        if (b_rescale_)
        {
            cv::resize(current_frame, current_frame_mat, detection_size_); //resized image: downscale, CV_INTER_LINEAR
        }
        else
        {
            current_frame_mat = current_frame.clone();
        }

        // convert RGB to GRAY
        cvtColor(current_frame_mat, frame_mat_gray, CV_BGR2GRAY);
        equalizeHist(frame_mat_gray,frame_mat_gray); // need??

        // set the upperbody detector and detect// kScaleFactor, kMinNeighbors, kFlag, min_size_
        //upperbodyhaar_.detectMultiScale(frame_mat_gray, body_rects, kScaleFactor, kMinNeighbors, kFlag, min_size_);
        DetectCascade(upperbodyhaar_, frame_mat_gray, body_rects);  //rescale issue has been considered in min_size_

        if (!body_rects.empty())
        {
            detectarray curr_upprect; //array<int,7>

            for (size_t irect=0; irect< body_rects.size(); ++irect)
            {
                const cv::Rect &curr_rect = body_rects[irect];
                if ((curr_rect.width >= min_size_.width) & (curr_rect.height >= min_size_.height))  //if confidence >= threshold
                {
                    curr_upprect[0] = curr_rect.x;
                    curr_upprect[1] = curr_rect.y;
                    curr_upprect[2] = curr_rect.x + curr_rect.width;
                    curr_upprect[3] = curr_rect.y + curr_rect.height;
                    curr_upprect[4] = 0;
                    curr_upprect[5] = 1;
                    curr_upprect[6] = CalculateRectArea(curr_rect);  //rect_area
                    rectangles.push_back(curr_upprect);
                }
            }

            if (rectangles.size()>1)
            {
                sort(rectangles.begin(), rectangles.end(),[](const detectarray &a, const detectarray &b) {\
                    return a[6] > b[6]; });  //descending order, rank from large to small
                rectangles = MaskNms(rectangles, kThresholdNms, detection_size_); //rescale issue has been considered in  detection_size_, THRESHOLD_MASKMNS = 0.5
            }

            // need to resize detected retangles back to original frame size
            if (b_rescale_)
            {
                // resize retangles back to original frame size
                for (size_t irect=0; irect< rectangles.size(); ++irect)
                {
                     detectarray &curr_upprect = rectangles[irect];
                     curr_upprect[0] = int(curr_upprect[0]*frame_scale_); //x1*current_scale
                     curr_upprect[1] = int(curr_upprect[1]*frame_scale_); //y1*current_scale
                     curr_upprect[2] = int(curr_upprect[2]*frame_scale_); //x2*current_scale
                     curr_upprect[3] = int(curr_upprect[3]*frame_scale_); //y2*current_scale
                }
            }

#ifdef UPPERDETECTOR_DEBUG_
            if (b_show_record_)
            {
                DoDisplayRecord(current_frame, curr_frame_index,rectangles);
            }
#endif

        } // end if (!body_rects.empty())

    } // end if(detect_status & !current_frame_mat.empty())
    else
    {
        printf ("no upperbody detected in frame %d", curr_frame_index); //need to write log file
    }


}


