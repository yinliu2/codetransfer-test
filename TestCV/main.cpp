#include "detection.h"
#include "motiondetector.h"
#include "upperbodydetector.h"

// extend cv::VideoCapture.  The const_cast<>()s
// work around the missing member const on cv::VideoCapture::get().
struct CvVideoCapture: VideoCapture
{
            double getFramesPerSecond() const {
                CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
                const double fps = p->get(CV_CAP_PROP_FPS);
                return fps ? fps : 30.0;        // for MacBook iSight camera
            }

            int getFourCcCodec() const {
                CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
                return p->get(CV_CAP_PROP_FOURCC);
            }

            std::string getFourCcCodecString() const {
                char result[] = "????";
                CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
                const int code = p->getFourCcCodec();
                result[0] = ((code >>  0) & 0xff);
                result[1] = ((code >>  8) & 0xff);
                result[2] = ((code >> 16) & 0xff);
                result[3] = ((code >> 24) & 0xff);
                result[4] = ""[0];  //represents "\0", because "\0" represents in array is {0 0} and "" is {0}.
                return std::string(result);
            }

            int getFrameCount() const {
                CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
                return p->get(CV_CAP_PROP_FRAME_COUNT);
            }

            CvSize getFrameSize() const {
                CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
                int const w = p->get(CV_CAP_PROP_FRAME_WIDTH);
                int const h = p->get(CV_CAP_PROP_FRAME_HEIGHT);
                CvSize const result= cvSize(w, h);
                return result;
            }

            int getPosition(void) const {
                CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
                return p->get(CV_CAP_PROP_POS_FRAMES);
            }
            void setPosition(int p) { this->set(CV_CAP_PROP_POS_FRAMES, p); }

            CvVideoCapture(const std::string &fileName): cv::VideoCapture(fileName) {}
            CvVideoCapture(int n): cv::VideoCapture(n) {}
            CvVideoCapture(): cv::VideoCapture() {}
};

//static CvVideoCapture openVideo(const char *source)
//{
//    std::string filename;
//    std::istringstream sss(source);
//    sss >> filename;
//    if (sss) return CvVideoCapture(filename);
//    return CvVideoCapture(-1);
//}

// draw rectangles imageframe
void DrawRectangleOnFrame(cv::Mat &frame, const vector2d &faces, const cv::Scalar color=cv::Scalar(255, 0, 0)) //const vector2d &prev_rects,
{
    //static const cv::Scalar bluecolor = cv::Scalar(255,   0,   0);
    static const int thickness = 4;
    static const int linekind = 8;
    static const int shift = 0;
    static size_t facecount = faces.size() > 0 ? faces.size():0;

    for (size_t i = 0; i < facecount; ++i)
    {
        const detectarray &rect = faces[i];

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


String trim(const String& str, const String& whitespace = " \t", const char& r = '\r')
{
    String newstr;
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
    {
        return ""; // no content
    }

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    newstr = str.substr(strBegin, strRange);
    if ((!newstr.empty()) && (newstr[newstr.size() - 1] == r))
    {
        newstr.erase(newstr.size() - 1);
    }
    return newstr;
}

// ////////////////////////////////////////////////////////////
// initilize and open video recorders to store detecion results
// ////////////////////////////////////////////////////////////
void InitRecorder(int width, int height)
{
    cv::Size frame_size = cv::Size(width, height);
    fg_output_video.open(fg_output_video_name, v_codecs, kFps, frame_size); // For writing the detection video
    mask_video.open(mask_video_name, v_codecs, kFps, frame_size); // For writing the mask video
    upper_output_video.open(fg_output_video_name, v_codecs, kFps, frame_size);

}

// /////////////////////////////////////////////////////////
// get file names from the indicated folder
// return: listed image file names for the test video stream
// /////////////////////////////////////////////////////////
vector<String> ReadFrameFolder()
{
    String  read_file_name;
    vector<String> file_names;
    vector<String> file_name_array;
    String str;

    glob(input_folder_, file_names); // new function that does the job

    cout << "Processing file " + curr_file_index_str_ + "..." << endl;
    read_file_name = data_folder_str_ + "video-fold/video-fold-" + curr_file_index_str_ + ".txt";
    cout << read_file_name << endl;

    ifstream input_file(read_file_name);
    cout << "OK" +to_string(input_file.is_open()) << endl;

    if(!input_file.is_open())
    {
        cout << "Error" + read_file_name << endl;
        cerr << "Error: " << strerror(errno);
    }

    while (getline(input_file, str))
    {
        file_name_array.push_back(str);
    }

    cout << "OK, framename file Getline is done."<< endl;

    return file_name_array;

}

// ///////////////////////////////////
// destroy imshow windows
// release opened video recorders
// ///////////////////////////////////
void ReleaseRecorder()
{
    destroyAllWindows();
    waitKey(1);

    if (upper_output_video.isOpened())
    {
        upper_output_video.release();
    }
    if (mask_video.isOpened())
    {
        mask_video.release();
    }
    if (final_output_video.isOpened())
    {
        final_output_video.release();
    }
}

// ///////////////////////////////////////////////////////////////////////////////////////
// enlarge input region list (2D vector)
// param: input region list (2D vector)
// param(output): reference region list (2D vector) of enlarged regions
// return: length of the result region list (2D vector)
// ///////////////////////////////////////////////////////////////////////////////////////
int CreatePrevFrameRect(const vector2d &rects_in, vector2d &rects_out)
{
    rects_out = {};
    if (!rects_in.empty())
    {
        int x1, x2, y1, y2;
        int x1new, x2new, y1new, y2new;
        int wrect, hrect;
        int rect_area;
        float area_rate;
        detectarray new_rect;

        for(size_t irect=0; irect<rects_in.size(); ++irect) // for irect in rect_nomove
        {
            const detectarray &rect = rects_in[irect];
            x1 = rect[0];
            y1 = rect[1];
            x2 = rect[2];
            y2 = rect[3];
            wrect = x2-x1;
            hrect = y2-y1;
            y2new = std::min(int(frame_height_),y2+ int(hrect*0.5));
            y1new = std::max(0,y1- int(hrect*0.5));
            x2new = std::min(int(frame_width_),x2+ int(wrect*0.5));
            x1new = std::max(0,x1- int(wrect*0.5));
            rect_area = (y2new-y1new+1)*(x2new-x1new+1);
            area_rate = float(rect_area)/float(frame_area_);
            if(area_rate>0.1) //1/8
            {
                y2new = std::min(int(frame_height_), y2 + int(hrect*0.125));
                y1new = std::max(0, y1 - int(hrect*0.125));
                x2new = std::min(int(frame_width_), x2 + int(wrect*0.125));
                x1new = std::max(0, x1 - int(wrect*0.125));
                rect_area = (y2new-y1new+1)*(x2new-x1new+1);
            }
            new_rect[0] = x1new;    // first rectangle top left x
            new_rect[1] = y1new;    // first rectangle top left y
            new_rect[2] = x2new;  // first rectangle bottom right x
            new_rect[3] = y2new; // first rectangle bottom right y
            new_rect[4] = rect[4];
            new_rect[5] = current_scale_;
            new_rect[6] = rect_area;  //rect_area

            rects_out.push_back(new_rect);
        }
        return rects_out.size();
    }
    else
    {
        cout<< "input vector2d is empty" << endl;
        return 0;
    }

}

int main(void)
{
    //get file names from the indicated folder
    cv::Mat first_frame;
    String  read_image_name;

    vector<String> file_name_input_list;
    String str;
    int first_frame_index = 0; // first frame

    file_name_input_list = ReadFrameFolder();
    int frame_count = file_name_input_list.size();

    //read the first frame
    read_image_name = data_folder_root_str_ +file_name_input_list[first_frame_index]; //.rstrip;
    cout << "OK read first image " + read_image_name  << endl;
    read_image_name = trim(read_image_name);
    cout << "read_image_name =" + read_image_name  << endl;
    first_frame = cv::imread(read_image_name, CV_LOAD_IMAGE_COLOR);

    // check whether the first frame has been loaded from image file
    if(first_frame.empty())
    {
        cout<<"image not found or read!<<endl";
        exit (EXIT_FAILURE);
    }

    //check frame size setting
    cv::Size first_frame_size = first_frame.size();
    cout << "first_frame_size = " << first_frame.size() << endl;
    int first_frame_width = first_frame_size.width;
    int first_frame_height= first_frame_size.height;

    if((first_frame_width!=frame_width_)||(first_frame_height!=frame_height_))
    {
        cout << "frame size is set wrong!: size of first frame is different from the settings." << endl;
        cout << "first_frame_width = "  << first_frame_width << endl;
        cout << "first_frame_height = " << first_frame_height << endl;
        exit (EXIT_FAILURE);
    }


    // initilize 2D vectors to store the intermediate and final dection results
    static vector2d output_rectangles = {};
    static vector2d prev_rectangles = {};
    static vector2d fg_rectangles = {};
    static vector2d upper_rectangles = {};
    static vector2d new_fg_rectangles = {};
    static vector2d new_upper_rectangles = {};   
    static vector2d rect_nomask = {};
    static vector2d rect_nostill = {};
    static vector2d prev_rect_nomove = {};

    // create upperbody detector
    UpperbodyDetector upperdetector(frame_width_, frame_height_, b_upper_rescale_);
    // create foreground detector
    MotionDector fgdetector(frame_width_, frame_height_, fg_threshold_, fg_min_face_size_);

    // initialize the first frame for foreground detector
    fgdetector.InitilizeFirstFrame(first_frame); // no foreground detection at the 1st frame ???

    // initialize video recorders
    InitRecorder(frame_width_, frame_height_);

    // process each frame in the test stream
    for (int iframe=0; iframe<200; ++iframe)  //frame_count
    {
        if(iframe%100==0)
        {
            cout << "Processing image: " + to_string(iframe) << endl; // processing start
        }
        read_image_name  = data_folder_root_str_ +file_name_input_list[iframe];
        read_image_name  = trim(read_image_name);
        // load frame as original image
        curr_frame_org   = imread(read_image_name,CV_LOAD_IMAGE_COLOR);
        // a frame copy for drawing and storing the detection results
        curr_frame_4show = curr_frame_org.clone(); // for cvShowImage

        if(iframe >= 0)
        {
            cout<<"fg_detect start... F#=" <<iframe<<endl;
            fgdetector.Run(curr_frame_org, iframe, fg_rectangles);
            fgdetector.GetFgMask(fgmask); // read out fgmask
            cvtColor(fgmask, fgmask_rgb, CV_GRAY2BGR);
            imwrite( "Binary_Image2.jpg", fgmask_rgb);
            cout<<"upp_detect start... F#=" <<iframe<<endl;
            fgdetector.CleanFgMaskBuf();
            upperdetector.Run(curr_frame_org, iframe, upper_rectangles);

            // select no moving previous detected rectangles that are not overlapped with current detected foreground rectangles
            if (!output_rectangles.empty())
            {
                fgdetector.CheckPrevRectsOnFg(fg_rectangles, output_rectangles, prev_rect_nomove, fgmask, THRESHOLD_RECTOVERLAP, THRESHOLD_INFG);
            }

            //enlarge detected rectangles in the previous frame, where the regions are not heavy overlapped with current foreground
            CreatePrevFrameRect(prev_rect_nomove, prev_rectangles);

            // keep the upperbody detection results that are not heavily overlapped with the existing detected rectangles
            if ((!upper_rectangles.empty())&&(!fg_rectangles.empty()))
            {
                rect_nomask = {};
                upperdetector.CheckPrevFrameRectNoMask(upper_rectangles,fg_rectangles,rect_nomask,THRESHOLDOVERLAP_1);
                if (!rect_nomask.empty())
                {
                    for(size_t irect=0; irect<rect_nomask.size(); ++irect)
                    {
                        const detectarray &upprect = rect_nomask[irect];
                        new_upper_rectangles.push_back(upprect);
                    }
                }
            }
            else if (!upper_rectangles.empty())
            {
                new_upper_rectangles=upper_rectangles;
            }

            // merge new upperbody detection results with the previous frame results
            if (!new_upper_rectangles.empty())
            {
                for (size_t irect=0; irect<new_upper_rectangles.size(); ++irect)
                {                   
                    const detectarray &upprect = new_upper_rectangles[irect];
                    prev_rectangles.push_back(upprect);
                }
            }

            // merge the listed rectangles
            if (prev_rectangles.size()>1)
            {
                // NMS delete large rectangles overlappted with small rectangles
                sort(prev_rectangles.begin(), prev_rectangles.end(),[](const detectarray &a, const detectarray &b) {\
                    return a[6] > b[6]; });  //descending order
                prev_rectangles = upperdetector.MaskNms(prev_rectangles, THRESHOLDNMS_DN_1, frame_size_);
                // NMS delete small rectangles overlappted with large rectangles  
                sort(prev_rectangles.begin(), prev_rectangles.end(),[](const detectarray &a, const detectarray &b) {\
                    return a[6] < b[6]; });  //ascending order
                prev_rectangles = upperdetector.MaskNms(prev_rectangles, THRESHOLDNMS_UP, frame_size_); //rescale issue has been considered in  detection_size_, THRESHOLD_MASKMNS = 0.5
            }

            // display detection results
            if (!prev_rectangles.empty())
            {
                // display nonoverlapped upperbody rectangles
                String str = "frame# = " + to_string(iframe);
                putText(curr_frame_4show, str, cv::Point(25,30), kOutFont, 1, cv::Scalar(0,0,0), 1, LINE_AA);
                putText(fgmask_rgb, str, cv::Point(25,30), kOutFont, 1, Scalar(255,255,255), 1, LINE_AA); // put date on the frame cv::LINE_AA=16
                // draw rectangles imageframe 
                DrawRectangleOnFrame(curr_frame_4show, prev_rectangles);
                upper_output_video.write(curr_frame_4show);
                mask_video.write(fgmask_rgb);
                cv::imshow("Upperbody Image", curr_frame_4show); // use C++ API cv::imshow, instead of using C API cvShowImage
                waitKey(1);
                cv::imshow("FG", fgmask_rgb);
                waitKey(1);
            }

            // add foreground detection
            if ((!fg_rectangles.empty())&&(!prev_rectangles.empty()))
            {
                rect_nostill ={};
                upperdetector.CheckPrevFrameRectNoMask(fg_rectangles,prev_rectangles,rect_nostill,THRESHOLDOVERLAP_2);
                if (!rect_nostill.empty())
                {
                    for(size_t irect=0; irect<rect_nostill.size(); ++irect)
                    {
                        const detectarray &rect = rect_nostill[irect];
                        new_fg_rectangles.push_back(rect);
                    }
                }
            }
            else if(!fg_rectangles.empty())
            {
               new_fg_rectangles = fg_rectangles;
            }

            // merge the listed rectangles,
            // add new foreground detection to the existing rectangle list (prev_rectangles)
            if (!prev_rectangles.empty())
            {
                for (size_t irect=0; irect<prev_rectangles.size(); ++irect)
                {
                    const detectarray &rect = prev_rectangles[irect];
                    new_fg_rectangles.push_back(rect);
                }
            }

            // merge the listed rectangles
            if (new_fg_rectangles.size()>1)
            {
                // NMS delete large rectangles overlappted with small rectangles
                sort(new_fg_rectangles.begin(), new_fg_rectangles.end(),[](const detectarray &a, const detectarray &b) {\
                    return a[6] > b[6]; });  //descending order
                new_fg_rectangles = upperdetector.MaskNms(new_fg_rectangles, THRESHOLDNMS_DN_2, frame_size_);
                // NMS delete small rectangles overlappted with large rectangles
                sort(new_fg_rectangles.begin(), new_fg_rectangles.end(),[](const detectarray &a, const detectarray &b) {\
                    return a[6] < b[6]; });  //ascending order
                new_fg_rectangles = upperdetector.MaskNms(new_fg_rectangles, THRESHOLDNMS_UP, frame_size_);
            }

            if (!new_fg_rectangles.empty())
            {
                String str = "frame# = " + to_string(iframe);
                // draw rectangles imageframe  ????           
                DrawRectangleOnFrame(curr_frame_4show, new_fg_rectangles);
                putText(curr_frame_4show, str, cv::Point(25,30), kOutFont, 1, cv::Scalar(0,0,0), 1, LINE_AA);
                final_output_video.write(curr_frame_4show);
                cv::imshow("Final Image", curr_frame_4show); // cvShowImage
                waitKey(1);
                cv::imshow("FG",  fgmask_rgb);
                const int c = waitKey(1000/10);
                if( c!=-1 ) break;
            }

        }//iframe>=0

        prev_rectangles = {};     // in upgrade version this should not be cleared after processing a frame
        fg_rectangles = {};
        upper_rectangles = {};
        new_fg_rectangles = {};
        new_upper_rectangles = {};

        rect_nomask = {};
        rect_nostill = {};

        fgmask = Mat::zeros(frame_height_, frame_width_, CV_8U);
        fgmask_rgb = Mat::zeros(frame_height_, frame_width_, CV_8UC3);

    }// for (int iframe=0; iframe<frame_count; ++iframe)

    ReleaseRecorder();

    return 0;

}
