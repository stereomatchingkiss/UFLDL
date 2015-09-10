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
                     std::ios::in | std::ios::binary);

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

    //in this case, mini-batch is hard to reach ideal
    //minimum, if the results are not good, train it again
    //by reuse the trained layer until the results are
    //reasonable. Or train the example with full batch size
    AutoEncoder ae(hidden_size);
    /*{
        std::ifstream in("autoencoder_test_2.xml");
        if(in.is_open()){
            ae.read("autoencoder_test_2.xml");
        }
    }*/
    ae.set_batch_size(400);
    ae.set_beta(3.0);
    ae.set_lambda(0.0001);
    ae.set_learning_rate(4);
    ae.set_max_iter(1000);
    ae.set_sparse(0.01);
    ae.set_epsillon(1e-9);
    {
        auto const Train = read_img("image.txt", 64, 10000);
        ae.train(Train);
        ae.set_reuse_layer(true);
        ae.set_batch_size(10000);
        ae.set_learning_rate(3.8);
        ae.set_max_iter(7000);
        ae.train(Train);
    }
    ae.write("autoencoder_test_2.xml");

    for(size_t i = 0; i != hidden_size.size(); ++i){
        auto &layer = ae.get_layer()[i];
        EMat const W1 = layer.w1_.transpose();
        cv::Mat img = ocv::eigen::eigen2cv_ref(W1);
        auto img_vision = visualize_network(img);
        cv::resize(img_vision, img_vision, {}, 4, 4);
        cv::imshow("", img_vision);
        cv::imwrite("encoder_result.jpg", img_vision);
        cv::waitKey();
    }
}
