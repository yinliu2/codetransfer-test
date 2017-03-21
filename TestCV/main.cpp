//#include <QCoreApplication>
//#include <opencv/ml.h>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <QDebug>
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;

// extend cv::VideoCapture.  The const_cast<>()s
// work around the missing member const on cv::VideoCapture::get().
struct CvVideoCapture: VideoCapture {

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


static void detectCascade(CascadeClassifier &classifier, const Mat &gray, std::vector<Rect>&regions)
{
    static double scaleFactor = 1.2;
    static const int minNeighbors = 3;
    static const Size minSize(40,40);
    static const Size maxSize;
    classifier.detectMultiScale(gray, regions, scaleFactor, minNeighbors,0|CV_HAAR_SCALE_IMAGE, minSize, maxSize);
}

// Draw rectangle r in color c on image i.
//
//static void drawRectangle(CvMat &i, const CvScalar &c, const CvRect &r)
static void drawRectangle(IplImage* image, const CvScalar &c, const CvRect &r)
{
    static const int thickness = 4;
    static const int lineKind = 8;
    static const int shift = 0;
    // draws a rectangle with given coordinates of the upper left
    //	and lower right corners into an image
    CvPoint ul; CvPoint lr;
    ul.x = r.x;
    ul.y = r.y;
    lr.x = r.x + r.width;
    lr.y = r.y + r.height;
    //cvRectangle(image, r, c, thickness, lineKind, shift);
    cvRectangle(image, ul, lr, c, thickness, lineKind, shift);
}


//static void drawBody(CvMat &frame, const CvRect &body,
static void drawBody(IplImage* frame, const std::vector<cv::Rect> &bodies)
{
    static const CvScalar blue = cvScalar(255,   0,   0);
    const size_t bodyCount = bodies.size() > 0 ? bodies.size():0;
    for (size_t b = 0; b < bodyCount; ++b) {
        const CvRect &body = bodies[b];
        drawRectangle(frame, blue, body);
    }

    //const size_t faceCount = faces.size() > 1 ? 1 : faces.size();
    //for (size_t f = 0; f < faceCount; ++f) {
    //    const CvRect &face = faces[f];
    //    drawRectangle(frame, green, face + bodyLocation);
    //    const CvPoint faceLocation = face.tl() + bodyLocation;
    //    const size_t eyeCount = eyes.size() > 2 ? 2 : eyes.size();
    //    for (size_t e = 0; e < eyeCount; ++e) {
    //        drawRectangle(frame, red, eyes[e] + faceLocation);
    //    }
    //}
}

//static void displayBody(Mat &frame,
static void displayBody(IplImage* frame,
                        CascadeClassifier &bodyHaar)
{
    static Mat gray;
    cvtColor(Mat(frame), gray, CV_BGR2GRAY);
    equalizeHist(gray,gray);
    static std::vector<Rect> bodyRects;
    detectCascade(bodyHaar, gray, bodyRects);
    drawBody(frame, bodyRects);
    //for (size_t i=0; i<bodyRects.size(); ++i){
        //const cv::Mat bodyROI = gray(bodyRects[i]);
        //static std::vector<cv::Rect> faces;
        //detectCasecade(faceHaar, bodyROI, faces);
        //drawBody(frame, bodyRects[i], faces);
    //imshow("cascade upper body detection",frame);
    cvShowImage("cascade upper body detection",frame);
}

static CvVideoCapture openVideo(const char *source)
{
    int cameraId = 0;
    std::istringstream iss(source);
    iss >> cameraId;
    if (iss) return CvVideoCapture(cameraId);
    std::string filename;
    std::istringstream sss(source);
    sss >> filename;
    if (sss) return CvVideoCapture(filename);
    return CvVideoCapture(-1);
}

//int main(int argc, char *argv[])
const char *cascadeUpBodyName = "/home/ying/anaconda2/share/OpenCV/haarcascades/haarcascade_mcs_upperbody.xml";
String folder = "/home/ying/surveillance/multi_face_turning/";
// extract Region of Interests: upper body
stringstream outnamestream;
const char* outname_for_opencv;
const char* filename_for_opencv;


int main(void)
{
    //QCoreApplication a(argc, argv);
    //IplImage *src_img = 0, *src_gray = 0;
    vector<String> filenames;
    glob(folder, filenames); // new function that does the job
    //const char *cascadeUpBodyName = "/home/ying/anaconda2/share/openCV/haarcascades/haarcascade_mcs_upperbody.xml";
    qDebug()<<"Use haarcasecade upperbody detection in the video sequence"<<endl<<endl;
    std::cout <<"up body detecion ..." << std::endl;
    //::CascadeClassifier bodyHaar(av[1]);
    CascadeClassifier bodyHaar;
    bodyHaar.load(cascadeUpBodyName);
    const bool detectStatus = !bodyHaar.empty();
    //qDebug()<<!bodyHaar.empty()<<endl<<endl;
    if(detectStatus){
        for(size_t iframe=0; iframe<25; ++iframe ) //unsigned int  filenames.size()
        {
            filename_for_opencv = static_cast<const char*>(filenames[iframe].c_str());
            IplImage* currframe = cvLoadImage( filename_for_opencv, 1 );
            Mat framecopy = Mat(currframe);
            if(!framecopy.empty()){
                displayBody(currframe, bodyHaar);
            }
            const int c = waitKey(1000/10);
            if( c!=-1 ) break;
        }
        return 0;
    }
    //showUsage(av[0]);
    return 1;
}

