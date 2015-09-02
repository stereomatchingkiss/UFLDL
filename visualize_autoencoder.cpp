#include "visualize_autoencoder.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace{

void copy_to(cv::Mat const &input, cv::Mat &output)
{    
    int index = 0;
    for(int col = 0; col != output.cols; ++col){
        for(int row = 0; row != output.rows; ++row){
            output.at<double>(row, col) = input.at<double>(index++, 0);
        }
    }
}

}

/**
 * @brief visualize the results trained by autoencoder
 * @param input the w1 of auto encoder, dimension of rows are\n
 * same as the neurons of input layer(L1), dimension of cols are\n
 * same as the neurons of hidden layer(L2)
 * @return the features trained by autocoder which could be shown by image
 */
cv::Mat visualize_network(const cv::Mat &input)
{
    cv::Mat input_temp;
    input.convertTo(input_temp, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(input_temp, mean, stddev);
    input_temp -= mean[0];

    int rows = 0, cols = (int)std::ceil(std::sqrt(input_temp.cols));
    if(std::pow(std::floor(std::sqrt(input_temp.cols)), 2) != input_temp.cols){
        while(input_temp.cols % cols != 0 && cols < 1.2*std::sqrt(input_temp.cols)){
            ++cols;
        }
        rows = (int)std::ceil(input_temp.cols/cols);
    }else{
        cols = (int)std::sqrt(input_temp.cols);
        rows = cols;
    }

    int const SquareRows = (int)std::sqrt(input_temp.rows);
    int const Buf = 1;

    int const Offset = SquareRows+Buf;
    cv::Mat array  = cv::Mat::ones(Buf+rows*(Offset),
                                   Buf+cols*(Offset),
                                   input_temp.type());

    for(int k = 0; ;){
        for(int i = 0; i != rows; ++i){
            for(int j = 0; j != cols; ++j){
                if(k >= input_temp.cols){
                    continue;
                }
                double min = 0.0, max = 0.0;
                cv::minMaxLoc(cv::abs(input_temp.col(k)), &min, &max);
                cv::Mat reshape_mat(SquareRows, SquareRows, input_temp.type());
                copy_to(input_temp.col(k), reshape_mat);
                if(max != 0.0){
                    reshape_mat /= max;
                }
                reshape_mat.copyTo(array({j*(Offset), i*(Offset),
                                          reshape_mat.cols, reshape_mat.rows}));
                ++k;
            }
        }
    }

    cv::normalize(array, array, 0, 1,
                  cv::NORM_MINMAX);
    array *= 255.0;
    array.convertTo(array, CV_8U);

    return array;
}
