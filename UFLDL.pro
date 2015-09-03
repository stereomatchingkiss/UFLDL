TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

CONFIG += c++11

msvc:QMAKE_CXXFLAGS_RELEASE += /openmp
#gcc:QMAKE_CXXFLAGS_RELEASE += -fopenmp

DEFINES += OCV_TEST_AUTOENCODER OCV_PRINT_COST
DEFINES += OCV_TEST_SOFTMAX

#msvc:QMAKE_CXXFLAGS_RELEASE += /O2 /openmp /arch:AVX
#gcc:QMAKE_CXXFLAGS_RELEASE += -O3 -march=native -fopenmp -D_GLIBCXX_PARALLEL

include(../pri/eigen.pri)
include(../pri/cv.pri)

INCLUDEPATH += ..

SOURCES += main.cpp \
    softmax_test.cpp \
    mnist_reader.cpp \          
    visualize_autoencoder.cpp \
    autoencoder_test.cpp

include(deployment.pri)
qtcAddDeployment()

HEADERS += \
    softmax_test.hpp \
    mnist_reader.hpp \         
    visualize_autoencoder.hpp \
    autoencoder_test.hpp

