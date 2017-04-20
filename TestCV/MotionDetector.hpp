//*********************************************************************************
// class for forground detection
// input: video frames
// output: detection rectangles
//
//*********************************************************************************
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/videoio.hpp>
#include <opencv2/video/video.hpp>
//#include <opencv2/video/background_segm.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <QDebug>
#include <iostream>
#include <fstream>
#include <stdio.h>
using namespace std;
using namespace cv;

