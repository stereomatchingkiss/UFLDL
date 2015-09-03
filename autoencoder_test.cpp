#include "autoencoder_test.hpp"
#include "visualize_autoencoder.hpp"

#include <ocv_libs/eigen/eigen.hpp>
#include <ocv_libs/ml/deep_learning/autoencoder.hpp>
#include <ocv_libs/ml/deep_learning/softmax.hpp>
#include <ocv_libs/ml/deep_learning/propagation.hpp>
#include <ocv_libs/profile/measure.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <memory>

namespace{

using AutoEncoder = ocv::ml::autoencoder<>;
using EMat = AutoEncoder::EigenMat;

EMat read_img(std::string const &file_name,
              size_t rows, size_t cols)
{
    std::ifstream in(file_name,
                     std::ios::out | std::ios::binary);

    EMat mat(rows, cols);
    for(size_t col = 0; col != mat.cols(); ++col){
        for(size_t row = 0; row != mat.rows(); ++row){
            in>>mat(row, col);
        }
    }

    return mat;
}

}

void autoencoder_test()
{
    cv::AutoBuffer<int> hidden_size(1);
    hidden_size[0] = 25;

    AutoEncoder ae(hidden_size);
    ae.read("autoencoder_test.xml");
    /*ae.set_reuse_layer(true);
    ae.set_batch_size(10000);
    ae.set_beta(3.0);
    ae.set_lambda(0.0001);
    ae.set_learning_rate(4);
    ae.set_max_iter(20000);
    ae.set_sparse(0.01);
    ae.set_epsillon(1e-10);

    ae.train(read_img("image.txt", 64, 10000));
    ae.write("autoencoder_test.xml");//*/

    auto &layer = ae.get_layer()[0];
    EMat const W1 = layer.w1_.transpose();
    cv::Mat img = ocv::eigen::eigen2cv_ref(W1);
    auto img_vision = visualize_network(img);
    cv::resize(img_vision, img_vision, {}, 4, 4);
    cv::imshow("", img_vision);
    cv::waitKey();//*/
}
