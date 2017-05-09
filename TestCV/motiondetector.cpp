#include "motiondetector.h"

//need to do is simply to provide a definition for your static member variable
size_t MotionDector::width_  = 720;
size_t MotionDector::height_ = 576;
size_t MotionDector::cnt_threshold_ = 12;
cv::Mat MotionDector::frame1gray_ = cv::Mat::zeros(height_, width_, CV_8U);  // = np.zeros((frame_height, frame_width, 1), dtype = np.uint8); // need release
cv::Mat MotionDector::frame2gray_ = cv::Mat::zeros(height_, width_, CV_8U);
cv::Mat MotionDector::framediff_  = cv::Mat::zeros(height_, width_, CV_32F); //Will hold the thresholded result
cv::Mat MotionDector::fgmask_     = cv::Mat::zeros(height_, width_, CV_8U);
cv::Mat MotionDector::fgmask_copy_ = cv::Mat::zeros(height_, width_, CV_8U);

#ifdef FGDETECTOR_DEBUG_
cv::Mat MotionDector::fgmask_rgb_ = cv::Mat::zeros(height_, width_, CV_8UC3);
#endif

#ifdef FGDETECTOR_DEBUG_
// ///////////////////////////////////////
// param : threshold for moving detection
// return: no
// ///////////////////////////////////////
// The callback function either as a global function or a static member function.
// To make it more OOP look, you might prefer to implement it as a static member function.
void MotionDector::ChangeParam(int v_new_threshold, void *)
{
    if(v_new_threshold>0)
    {
        cnt_threshold_ = v_new_threshold;
    }
    else
    {
        printf ("threshold value can not be negative !");
        exit (EXIT_FAILURE);
    }
}

// //////////////////////////////////////////////
// Initialize and open recorders for dector test
// param : open recorder is ture/false
// return: no
// //////////////////////////////////////////////
void MotionDector::InitRecorder()
{
    if(b_show_record_)
    {
        v_codecs_ = CV_FOURCC('M', 'J', 'P', 'G');
        fps_ = 4;
        output_video_name_ = "fgclass_output.avi";
        mask_video_name_   = "fgclass_mask.avi";
        Size s = Size(width_, height_);
        output_video_.open(output_video_name_, v_codecs_, fps_, s); // For writing the detection video
        mask_video_.open(mask_video_name_, v_codecs_, fps_, s); // For writing the mask video
    }
}

// ////////////////////////////////////////////////////////////////
// Display and record foreground detection result for current frame
// ////////////////////////////////////////////////////////////////
void MotionDector::DoDisplayRecord(const cv::Mat &frame, const int frame_index, const vector2d &rectangles)
{
    if (b_show_record_)
    {
        // draw detected rectangles on the current frame
        if (!rectangles.empty())
        {
            String str = "frame# = " + to_string(frame_index);
            static cv::Mat frame_4show = frame.clone();

            cvtColor(fgmask_, fgmask_rgb_, CV_GRAY2BGR);
            DrawRectangleOnFrame(frame_4show, rectangles); // draw rectangles imageframe      

            putText(frame_4show, str, cv::Point(25,30), kOutFont, 1, cv::Scalar(0,0,0), 1, LINE_AA); // put date on the frame cv::LINE_AA=16
            output_video_.write(frame_4show);
            cv::imshow("fg class: result", frame_4show);
            waitKey(1);

            putText(fgmask_rgb_, str, cv::Point(25,30), kOutFont, 1, cv::Scalar(0,0,0), 1, LINE_AA); // put date on the frame cv::LINE_AA=16
            mask_video_.write(fgmask_rgb_);
            cv::imshow("fg class: mask",  fgmask_rgb_);
            waitKey(1000/10);
        }
    }

}
// ////////////////////////////////////////////////////////////////
// close image show windows and release recorders for detector test
// ////////////////////////////////////////////////////////////////
void MotionDector::ReleaseDisplayRecorder()
{
    if (b_show_record_)
    {
        destroyAllWindows();
        waitKey(1);
        if (output_video_.isOpened())
        {
            output_video_.release();
        }
        if (mask_video_.isOpened())
        {
            mask_video_.release();
        }
    }

}

// ////////////////////////////////////////////////////////////////
// enable display and recorder
// ////////////////////////////////////////////////////////////////
void MotionDector::OpenDisplayRecorder()
{
    b_show_record_ = true;

    const int alpha_slider_max = 100;
    int alpha_slider = int(cnt_threshold_);
    cv::namedWindow("fg class: result");
    createTrackbar("Detection treshold: ", "fg class: result", &alpha_slider, alpha_slider_max, ChangeParam);

    InitRecorder();

}

// ////////////////////////////////////////////////////////////////
// disable display and recorder and release related objects
// ////////////////////////////////////////////////////////////////
void MotionDector::CloseDisplayRecorder()
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
void MotionDector::DrawRectangleOnFrame(cv::Mat &frame, const vector2d &detectedrects, const cv::Scalar color) //const vector2d &prev_rects,
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

// /////////////////////////////////////////
// Initialize the first frame of the stream
// param : original first-frame
// return: no
// /////////////////////////////////////////
void  MotionDector::InitilizeFirstFrame(const cv::Mat &first_frame)
{
    cvtColor(first_frame, frame1gray_, CV_BGR2GRAY);
}


MotionDector::MotionDector(int frame_width, int frame_height, int v_threshold, int v_minSize)
{
    width_ = frame_width;
    height_ = frame_height;
    current_scale_ = 1;
    frame_pixel_num_ = width_ * height_;
    min_face_size_ = v_minSize;
    b_show_record_ = false; //b_show_record_input;

    pMog2_ = new BackgroundSubtractorMOG2(100, 5, false); //300, 16, false, varThreshold=5, false MOG2 approach (history, varThreshold, bShadowDetection=true)
    frame_num_ = 0;

    cnt_threshold_ = v_threshold;  // static variable

}

// ////////////////////////////////////
// class deconstructor
// ////////////////////////////////////
MotionDector::~MotionDector()
{
}

// ////////////////////////////////////
// param :  current frame
// return:  no
// ////////////////////////////////////
//void MotionDector::ProcessImage(IplImage* frame)
void MotionDector::ProcessImage(const cv::Mat &frame)
{
    cvtColor(frame, frame2gray_, CV_BGR2GRAY);//kMog2LearningRate
    pMog2_->operator ()( frame2gray_, fgmask_, kMog2LearningRate);//update the background model: apply(frame, fgmask_, learningRate=0.1);
    cv::threshold(fgmask_, fgmask_, 10, MAX_BINARY_VALUE, cv::THRESH_BINARY);

    cv::Mat kernelmat = cv::getStructuringElement(cv::MORPH_CROSS,cv::Size(5, 5));//,cv::Point(-1, -1)
    cv::morphologyEx(fgmask_, fgmask_, cv::MORPH_OPEN, kernelmat);
    cv::morphologyEx(fgmask_, fgmask_, cv::MORPH_CLOSE, kernelmat);
    cv::dilate(fgmask_,fgmask_, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
    fgmask_copy_= fgmask_.clone(); // make a copy for out-class fetch
    cv::findContours( fgmask_, contours_, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE); //, cv::Point(0, 0)

#ifdef FGDETECTOR_TEST_
    ofstream fglogfile;
    time_t t = time(0);
    struct tm *now = localtime(&t);
    String logfilename = "logfile_"+std::to_string(now->tm_year)+std::to_string(now->tm_mon)\
            +std::to_string(now->tm_mday)+"-"+std::to_string(now->tm_hour)+"-"+std::to_string(now->tm_min);

    fglogfile.open(logfilename+".txt");
    fglogfile << "fgmask contours: "<<std::endl;

    for (size_t icnt=0; icnt< contours_.size(); ++icnt)
    {
        cv::Rect rect = cv::boundingRect(contours_[icnt]);

        fglogfile << std::to_string(icnt)<<"  " << std::to_string(rect.x)<<"  "<< std::to_string(rect.y)<<"  "<<\
                       std::to_string(rect.x + rect.width)<<"  "<< std::to_string(rect.y + rect.height)<<std::endl ;
    }
    fglogfile.close();
#endif

}

// ///////////////////////////////////////////
// Param : no
// return: whether something is moving or not
// ///////////////////////////////////////////
bool MotionDector::IsSomethingMoved()
{
    size_t nb=0;
    size_t avg;
    for (size_t ix=0; ix<height_; ++ix) //Iterate the hole image
    {
        for (size_t iy=0; iy<width_; ++iy)
        {
            if (framediff_.at<int>(ix,iy)<2) //If the pixel is black keep it
            {
                nb+=1;
            }
        }
    }
    avg = size_t((nb*100.0)/frame_pixel_num_+0.5);  //Calculate the average of black pixel in the image

    if(avg>cnt_threshold_) //If over the ceiling trigger the alarm
    {
        return true;
    }
    else
    {
        return false;
    }

}


// ////////////////////////////////////
// calculate region area by pixel number
// function overloadding for cv:Rect input
// param : rectangle reference
// return: calculated total region pixel number
// ////////////// //////////////////////
size_t  MotionDector::CalculateRectArea(const cv::Rect &rect)
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
size_t  MotionDector::CalculateRectArea(const detectarray &rectarr)
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


// ////////////////////////////////////////////////////////////////
// param :  two detected face regions stored with array<int,7> in
//          format [x11, y11, x12, y12, confidence, current_scale_]
// return:  IoU ratio (intersection over union) of two rectangles
// ////////////////////////////////////////////////////////////////
float MotionDector::CalculateIoU(const detectarray &rect_1, const detectarray &rect_2)
{
    int x_overlap, y_overlap, intersection, union_area;
    int x11, y11, x12, y12, x21, y21, x22, y22;
    int x1_max, x2_min, y1_max, y2_min;

    x11 = rect_1[0];   //rect[0] first rectangle top left x
    y11 = rect_1[1];   //rect[1] first rectangle top left y
    x12 = rect_1[2];   //rect[2] first rectangle bottom right x
    y12 = rect_1[3];   //rect[3] first rectangle bottom right y

    x21 = rect_2[0];   //rect[0] second rectangle top left x
    y21 = rect_2[1];   //rect[1] second rectangle top left y
    x22 = rect_2[2];   //rect[2] second rectangle bottom right x
    y22 = rect_2[3];   //rect[3] second rectangle bottom right y

    x2_min = std::min(x12,x22);
    x1_max = std::max(x11,x21);
    y2_min = std::max(x11,x21);
    y1_max = std::max(y11,y21);

    x_overlap = std::max(0, x2_min - x1_max);
    y_overlap = std::max(0, y2_min - y1_max);
    intersection = x_overlap * y_overlap;
    union_area = (x12-x11) * (y12-y11) + (x22-x21) * (y22-y21) - intersection;

    return float(intersection)/float(union_area);
}

// ////////////////////////////////////////////////////////////////////
// param : two detected face regions stored with array<int,7> in
//         format [x11, y11, x12, y12, confidence, current_scale_]
// return: IoM ratio (intersection over min-area) of two rectangles
// ////////////////////////////////////////////////////////////////////
float MotionDector::CalculateIoM(const detectarray &rect_1, const detectarray &rect_2)
{
    int x_overlap, y_overlap, intersection, rect1_area, rect2_area, min_area;
    int x11, y11, x12, y12, x21, y21, x22, y22;
    int x1_max, x2_min, y1_max, y2_min;

    x11 = rect_1[0];   //rect[0] first rectangle top left x
    y11 = rect_1[1];   //rect[1] first rectangle top left y
    x12 = rect_1[2];   //rect[2] first rectangle bottom right x
    y12 = rect_1[3];   //rect[3] first rectangle bottom right y

    x21 = rect_2[0];   //rect[0] second rectangle top left x
    y21 = rect_2[1];   //rect[1] second rectangle top left y
    x22 = rect_2[2];   //rect[2] second rectangle bottom right x
    y22 = rect_2[3];   //rect[3] second rectangle bottom right y

    x2_min = std::min(x12,x22);
    x1_max = std::max(x11,x21);
    y2_min = std::max(x11,x21);
    y1_max = std::max(y11,y21);

    x_overlap = std::max(0, x2_min - x1_max);
    y_overlap = std::max(0, y2_min - y1_max);

    intersection = x_overlap * y_overlap;
    rect1_area = (y12 - y11) * (x12 - x11);
    rect2_area = (y22 - y21) * (x22 - x21);
    min_area = min(rect1_area, rect2_area);

    return float(intersection)/float(min_area);
}

// /////////////////////////////////////////////
// param : detected face region list (2D vector)
// return: no
// ////////////////////////////////////
void MotionDector::LocalNms(vector2d &rectangles)
{
    int rect_number=rectangles.size();
    const int kThresholdIoU = 0.3; // threshold of IoU of two rectangles
    int kThresholdIoM = 0.6; // threshold of IoM of two rectangles
    int i_curr_rect=0;
    int i_curr_rect_to_compare;
    int rects_to_compare;

    while(i_curr_rect < rect_number-1) // start from first element to second last element
    {
        rects_to_compare = rect_number-i_curr_rect-1; // elements after current element to compare
        i_curr_rect_to_compare = i_curr_rect + 1; // start comparing with element after current
        while (rects_to_compare > 0) // while there is at least one element after currrent to compare
        {
            if ((CalculateIoM(rectangles[i_curr_rect],rectangles[i_curr_rect_to_compare]) >= kThresholdIoM)&&\
                    (rectangles[i_curr_rect][5] == rectangles[i_curr_rect_to_compare][5]))
            {
                rectangles.erase(rectangles.begin()+i_curr_rect_to_compare); // delete the rectangle at the same scale
                rect_number-=1;
            }
            else if((CalculateIoU(rectangles[i_curr_rect], rectangles[i_curr_rect_to_compare]) >= kThresholdIoU)&&\
                    (rectangles[i_curr_rect][5] == rectangles[i_curr_rect_to_compare][5]))
            {
                rectangles.erase(rectangles.begin()+i_curr_rect_to_compare);    // delete the rectangle at the same scale
                rect_number-=1;
            }
            else
            {
                i_curr_rect_to_compare += 1;  // skip to next rectangl
            }
            rects_to_compare -= 1;
        }
        i_curr_rect+=1;  // finished comparing for current rectangle
    }

}

// /////////////////////////////////////////////////////////
// check the overlap beteen foreground detected results and previous detected regions
// param: detected foreground region list (2D vector)
// param: detected face region list (2D vector) from the previous frame
// param(output): reference list (2D vector) of previous detected face regions that is not in foreground
// return: length of the result region list (2D vector)
// /////////////////////////////////////////////////////////
int  MotionDector::CheckPrevRectsOnFg(const vector2d &fg_rects, const vector2d &prev_rects, vector2d &rect_outfg, const cv::Mat &fg_mask,\
                        const float threshold_overlap, const float threshold_infg)
{
    cv::Mat fg_rect_mask = cv::Mat::zeros(height_, width_, CV_8U);
    rect_outfg = {};

    cv::Mat mask_copy = fg_mask.clone();
    //mask_copy.setTo(1, mask_copy == 255); // replace pixel_value=255 to 1

    int x1, x2, y1, y2;

    if (!fg_rects.empty())
    {
        for(size_t irect=0; irect<fg_rects.size(); ++irect)
        {
            const detectarray &rect = fg_rects[irect];
            x1 = rect[0];
            y1 = rect[1];
            x2 = rect[2];
            y2 = rect[3];
            fg_rect_mask(cv::Range(y1,y2), cv::Range(x1,x2)) = 1; // mark detected foreground region as "1"
        }
    }

    // the moving mask inside the retangles
    cv::Mat mask_multiplied =  mask_copy.mul(fg_rect_mask);
    mask_copy.setTo(1, mask_multiplied == 255);

    if (!prev_rects.empty())
    {
        for(size_t irect=0; irect<prev_rects.size(); ++irect)
        {
            const detectarray &rect = prev_rects[irect];
            int rect_area_num = CalculateRectArea(rect);  //int rect_num = (y2-y1+1) * (x2-x1+1);
            x1 = rect[0];
            y1 = rect[1];
            x2 = rect[2];
            y2 = rect[3];

            cv::Mat fg_rect_mask_crop = fg_rect_mask(cv::Range(y1,y2),cv::Range(x1,x2));
            cv::Mat mask_crop = mask_copy(cv::Range(y1,y2),cv::Range(x1,x2));

            int fg_rect_mask_sum = cv::sum(fg_rect_mask_crop)[0];
            int fg_mask_sum = cv::sum(mask_crop)[0];

            float overlap_rate = float(fg_rect_mask_sum)/float(rect_area_num);
            float move_rate  = float(fg_mask_sum)/float(rect_area_num);

            if(overlap_rate<threshold_overlap and move_rate<threshold_infg)
            {
                rect_outfg.push_back(rect);
            }

        }
        return rect_outfg.size();
    } // end if (!prev_rects.empty())
    else
    {
        return 0;
    }// end else (!prev_rects.empty())

}

// ////////////////////////////////////
// get private foreground detection mask
// param : foreground mask matrix to be filled
// return: filled detected foreground mask matrix
// ////////////////////////////////////
void MotionDector::GetFgMask(cv::Mat &fgmask_mat)
{
    // Deep copies.
    fgmask_mat=fgmask_copy_.clone();

    //cv::Mat fgmask_rgb = Mat::zeros(height_, width_, CV_8UC3);
    //cvtColor(fgmask_copy_, fgmask_rgb, CV_GRAY2BGR);
    //imwrite( "Binary_GetFgMask_out.jpg", fgmask_rgb);
}

void MotionDector::CleanFgMaskBuf()
{
    fgmask_copy_ = cv::Mat::zeros(height_, width_, CV_8U);

}

// /////////////////////////////////////////////
// run foreground detection
// param : current frame, current frame index(#)
// return: no
// output: detected face region list (2D vector)
// /////////////////////////////////////////////
//vector<int*>  MotionDector::Run(cv::Mat& curr_frame_input, int curr_frame_index)
void  MotionDector::Run(const cv::Mat &curr_frame, const int curr_frame_index, vector2d &rectangles)
{
    //started = time.time()  // set timer
    rectangles={};
    ProcessImage(curr_frame);
    if(b_motion_exist_) // will use IsSomethingMoved() in later updates
    {
        size_t contour_area_v;
        float contour_perc;
        int contour_perc_x100;
        cv::Rect contour_rect;
        detectarray current_rectangle; // typedef array<int,7> detectarray
        for (size_t icnt=0; icnt< contours_.size(); ++icnt)
        {
            contour_area_v = contourArea(contours_[icnt]);
            contour_perc = float(contour_area_v*1000.0)/float(frame_pixel_num_);
            contour_perc_x100 = int(contour_perc*100.0);
            if (contour_perc < float(cnt_threshold_)) // if the contour is too small, ignore it
            {
                continue;
            }

            // compute the bounding box for the contour, draw it on the frame, and update the text
            contour_rect = boundingRect(contours_[icnt]);

            if (contour_rect.width < min_face_size_ or contour_rect.height < min_face_size_)
            {
                continue;
            }

            current_rectangle[0] = contour_rect.x;    // first rectangle top left x
            current_rectangle[1] = contour_rect.y;    // first rectangle top left y
            current_rectangle[2] = contour_rect.x + contour_rect.width;  // first rectangle bottom right x
            current_rectangle[3] = contour_rect.y + contour_rect.height; // first rectangle bottom right y
            current_rectangle[4] = contour_perc_x100;
            current_rectangle[5] = current_scale_ ;
            current_rectangle[6] = CalculateRectArea(contour_rect);  //rect_area

            rectangles.push_back(current_rectangle);  //append current_rectangle on vector<rectangle>

        }

        // sort rectangles according to confidence("reac_area" value) reverse, so that it ranks from large to small
        // sort(rectangles.begin(), rectangles.end(),[](const std::vector<int> &a, const std::vector<int> &b) {return a[6] > b[6]; });
        if(rectangles.size()>1)
        {
            sort(rectangles.begin(), rectangles.end(),[](const detectarray &a, const detectarray &b) {\
                return a[6] > b[6]; });  //descending order
            LocalNms(rectangles);
        }

        frame1gray_ =frame2gray_; // frame2 is copy to frame1 to calculate the next frame difference

#ifdef FGDETECTOR_DEBUG_
        if (b_show_record_)
        {
            DoDisplayRecord(curr_frame, curr_frame_index,rectangles);
        }
#endif

    }


}//end run



