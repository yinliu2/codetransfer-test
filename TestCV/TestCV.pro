QT += core
QT -= gui

CONFIG += c++11

TARGET = TestCV
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    upperbodydetector.cpp \
    motiondetector.cpp

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += /homr/ying/opencv/include/opencv
INCLUDEPATH += /homr/ying/opencv/include/opencv2
#INCLUDEPATH += /homr/ying/anaconda2/include/opencv
#INCLUDEPATH += /homr/ying/anaconda2/include/opencv2
LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_video

#LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui

HEADERS += \
    upperbodydetector.h \
    motiondetector.h \
    facedetection.h
