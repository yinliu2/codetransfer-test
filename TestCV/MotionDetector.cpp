#include "MotionDetector.hpp"

class MotionDector
{
    private:
    const bool b_motion_exist_ = true;
    size_t width_;
    size_t height_;
    size_t current_scale_;
    size_t frame_pixel_num_;
    int min_face_size_;
    static size_t cnt_threshold_;
    bool b_show_window_;
    bool b_record_;
    const int kOutFont = FONT_HERSHEY_SIMPLEX;
    int v_codecs_;
    cv::VideoWriter output_video_;
    cv::VideoWriter mask_video_;
    int fps_;
    String output_video_name_;
    String mask_video_name_;
    //cv::Ptr<cv::BackgroundSubtractorMOG2> pMog2_; //??? Ptr<BackgroundSubtractorMOG2>
    Ptr<BackgroundSubtractor> pMog2_;
    const bool kMogDetectShadows = false;
    const float kMog2LearningRate = 0.01;  //The const qualifier demands that the variable not be changed after its initialization
    //self.doMotionDetect = doMotionDetect
    static Mat frame1gray_;  // = np.zeros((frame_height, frame_width, 1), dtype = np.uint8); // need release
    static Mat frame2gray_;
    static Mat framediff_;
    static Mat fgmask_;
    static Mat fgmask_rgb_;
    vector<vector<Point> > contours_;
    int frame_num_;

    public:
    // The callback function either as a global function or a static member function.
    // To make it more OOP look, you might prefer to implement it as a static member function.
    static void ChangeParam(int v_new_threshold, void *)
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


    void InitRecorder()
    {
        v_codecs_ = CV_FOURCC('M', 'J', 'P', 'G');
        fps_ = 4;
        output_video_name_ = "fgoutput.avi";
        mask_video_name_   = "fgmask_output.avi";
        //writer = cv2.VideoWriter(output_video_name_,v_codecs_, 20,  (width_,height_));
        //maskwriter = cv2.VideoWriter('maskoutput.avi',fourcc, 20, (width_,height_));
        Size s = Size(width_, height_);
        output_video_.open(output_video_name_, v_codecs_, fps_, s); // For writing the detection video
        mask_video_.open(mask_video_name_, v_codecs_, fps_, s); // For writing the mask video
    }

    void ReleaseRecorder()
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

    MotionDector(int frame_width, int frame_height, int v_threshold, int v_minSize, bool b_show_window_input)
    {
        // No need for assignment here
        // def __init__(self, doMotionDetect=True, frame_width=320, frame_height=240, cnt_threshold=13, b_show_window_=True, minSize = 56):
        width_ = frame_width;
        height_ = frame_height;
        current_scale_ = 1;
        frame_pixel_num_ = width_ * height_;
        min_face_size_ = v_minSize;
        cnt_threshold_ = v_threshold;
        b_show_window_ = b_show_window_input;
        b_record_ = true;
        //kOutFont = FONT_HERSHEY_SIMPLEX;
        //pMog2_ = cv::createBackgroundSubtractorMOG2(kMogDetectShadows);
        pMog2_ = new BackgroundSubtractorMOG2(100, 5, false); //MOG2 approach (history, varThreshold, bShadowDetection)
        //pMog2_->set("detectShadows", false);
        //self.doMotionDetect = doMotionDetect
        frame_num_ = 0;
        frame1gray_ = Mat::zeros(height_, width_, CV_8U);
        frame2gray_ = Mat::zeros(height_, width_, CV_8U);
        //Will hold the thresholded result
        //Mat image(size, CV_8U, (void*)dataPtr);
        framediff_ = Mat::zeros(height_, width_, CV_32F);
        fgmask_ = Mat::zeros(height_, width_, CV_8U);
        //self.cnts = None
        //trigger_time = 0 //Hold timestamp of the last detection
        if (b_record_)
        {
            InitRecorder();
        }
        if (b_show_window_)
        {
            const int alpha_slider_max = 100;
            int alpha_slider = int(cnt_threshold_);
            cvNamedWindow("Image");
            //createTrackbar("Detection treshold: ", "Image", &cnt_threshold_, 100, ChangeParam);
            createTrackbar("Detection treshold: ", "Image", &alpha_slider, alpha_slider_max, ChangeParam);
            fgmask_rgb_ = Mat::zeros(height_, width_, CV_8U);
        }
    }

    ~MotionDector()
    {
        //delete pMog2_;
    }

    void ProcessImage(IplImage* frame)
    {
        //const float kMog2LearningRate = 0.01;
        cvtColor(Mat(frame), frame2gray_, CV_BGR2GRAY);
        //update the background model 
        pMog2_->operator ()(Mat(frame), fgmask_, kMog2LearningRate);//>apply(frame, fgmask_, learningRate=0.1);
        threshold(fgmask_, fgmask_, 10, 255, THRESH_BINARY);
        Mat kernelmat = getStructuringElement( MORPH_CROSS, Size(5, 5), Point(-1, -1));
        morphologyEx(fgmask_, fgmask_, MORPH_OPEN, kernelmat);
        morphologyEx(fgmask_, fgmask_, MORPH_CLOSE, kernelmat);
        dilate(fgmask_,fgmask_, Mat(), Point(-1, -1), 1, 1, 1);
        //vector<vector<Point> > contours_;
        // Find contours_
        findContours( fgmask_, contours_, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    }

    bool IsSomethingMoved()
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

    size_t  CalculateRectArea(CvRect rect)
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

    float CalculateIoU(int* rect_1, int* rect_2)
    {
        //param rect_1: list in format [x11, y11, x12, y12, confidence, current_scale_]
        //param rect_2: list in format [x21, y21, x22, y22, confidence, current_scale_]
        //return:       returns IoU ratio (intersection over union) of two rectangles
        int x_overlap, y_overlap, intersection, union_area;
        int x11, y11, x12, y12, x21, y21, x22, y22;
        int x1_max, x2_min, y1_max, y2_min;

        x11 = *(rect_1+0);   //rect[0] first rectangle top left x
        y11 = *(rect_1+1);   //rect[1] first rectangle top left y
        x12 = *(rect_1+2);   //rect[2] first rectangle bottom right x
        y12 = *(rect_1+3);   //rect[3] first rectangle bottom right y

        x21 = *(rect_2+0);   //rect[0] second rectangle top left x
        y21 = *(rect_2+1);   //rect[1] second rectangle top left y
        x22 = *(rect_2+2);   //rect[2] second rectangle bottom right x
        y22 = *(rect_2+3);   //rect[3] second rectangle bottom right y

        x2_min = std::min(x12,x22);
        x1_max = std::max(x11,x21);
        y2_min = std::max(x11,x21);
        y1_max = std::max(y11,y21);

        x_overlap = std::max(0, x2_min - x1_max);
        y_overlap = std::max(0, y2_min - y1_max);
        intersection = x_overlap * y_overlap;
        union_area = (x12-x11) * (y12-y11) + (x22-x21) * (y22-y21) - intersection;

        return float(intersection)/union_area;
    }

    float CalculateIoM(int rect_1[], int rect_2[])
    {
        //param rect_1: list in format [x11, y11, x12, y12, confidence, current_scale_]
        //param rect_2: list in format [x21, y21, x22, y22, confidence, current_scale_]
        //return: returns IoM ratio (intersection over min-area) of two rectangles
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
        return float(intersection)/min_area;
    }

    void LocalNms(vector<int*>& rectangles)
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
                    i_curr_rect_to_compare += 1;  // skip to next rectangl
                rects_to_compare -= 1;
            }
            i_curr_rect+=1;  // finished comparing for current rectangle
        }

    }

    //int RecArea(vector<int>& rectarr)
    //{
    //    int rectarea;
    //    int x1, y1, x2, y2;
    //    x1 = rectarr[0];
    //    y1 = rectarr[1];
    //    x2 = rectarr[2];
    //    y2 = rectarr[3];
    //    rectarea = (y2 - y1+1) * (x2 - x1+1);
    //    return rectarea;
    //}

    vector<int*>  run(IplImage curr_frame, int curr_frame_index)
    {
        //started = time.time()
        vector<int*> rectangles={}; //if nothing moved; std::vector<T> have constructors initializing the object to be empty, so the empty initialization can be removed.
        ProcessImage(&curr_frame);
        if(b_motion_exist_) //IsSomethingMoved()
        {
            //Mat fgmask_rgb;
            size_t contour_area_v;
            float contour_perc;
            int contour_perc_x100;
            CvRect contour_rect;
            int current_rectangle[7];
            //vector<std::array<int*> > rectangles;
            //vector<int*> rectangles;
            //int rect_array[4];
            //int rect_area;
            //int x1, y1, x2, y2;
            for (size_t icnt=0; icnt< contours_.size(); icnt++)
            {
                contour_area_v = contourArea(contours_[icnt]);
                contour_perc = (contour_area_v*1000.0)/frame_pixel_num_;
                contour_perc_x100 = int(contour_perc*100);
                if (contour_perc < cnt_threshold_) // if the contour is too small, ignore it
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
                //rect_array = {x1, y1, x2, y2};
                current_rectangle[4] = contour_perc_x100;
                current_rectangle[5] = current_scale_ ;
                current_rectangle[6] = CalculateRectArea(contour_rect);  //rect_area

                //current_rectangle = {rect_array, contour_perc_x100, current_scale_, rect_area}; //find corresponding patch on image

                rectangles.push_back(current_rectangle);  //append current_rectangle on vector<rectangle>

            }
            // sort rectangles according to confidence("reac_area" value) reverse, so that it ranks from large to small
            //sort(rectangles.begin(), rectangles.end(),[](const std::vector<int>& a, const std::vector<int>& b) {return a[6] > b[6]; });
            sort(rectangles.begin(), rectangles.end(),[](const int* a, int* b) {\
                return a[6] > b[6]; });  //descending order
            LocalNms(rectangles);

            // draw detected rectangles on the current frame
            for (size_t irect=0; irect< rectangles.size(); irect++)
            {
                int * rect = rectangles[irect];
                static const int thickness = 2;
                // draws a rectangle with given coordinates of the upper left
                // and lower right corners into an image
                CvPoint ul, lr;
                ul.x = rect[0];
                ul.y = rect[1];
                lr.x = rect[2];
                lr.y = rect[3];
                cvRectangle(&curr_frame, ul, lr, cvScalar(0, 255, 0), thickness);
            }

            String str = "frame# = " + to_string(curr_frame_index);
            Mat frame_copy(&curr_frame);
            putText(frame_copy , str, Point(25,30), kOutFont, 1, Scalar(0,0,0), 1, 16); // put date on the frame cv::LINE_AA=16
            output_video_.write(frame_copy );

            frame1gray_ =frame2gray_; // frame2 is copy to frame1 to calculate the next frame difference

            if (b_show_window_)
            {
                cvtColor(fgmask_, fgmask_rgb_, CV_GRAY2BGR);
                String str = "frame# = " + to_string(curr_frame_index);
                putText(fgmask_rgb_, str, Point(25,30), kOutFont, 1, Scalar(255,255,255), 1, 16); // put date on the frame cv::LINE_AA=16

                mask_video_.write(fgmask_rgb_);
                cvShowImage("Image", &curr_frame);
                cvShowImage("FG",  &fgmask_rgb_);
                int c = cvWaitKey(1000/10);
                if (c == 27 or c == 10) //Break if user enters 'Esc'
                {
                    ReleaseRecorder();
                    //cvReleaseImage(&img);
                }
            }

            if (b_show_window_)
            {
                //cvDestroyAllWindow();
                //cvWaitKey(1);
                //output_video_.release();
                //mask_video_.release();
                ReleaseRecorder();
            }

        }

        return  rectangles;

    }//end run


};
