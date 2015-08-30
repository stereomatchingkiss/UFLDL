#include "softmax_test.hpp"

#include <ocv_libs/ml/deep_learning/softmax.hpp>

#include <fstream>

using namespace ocv::ml;

namespace{

using EMat = ocv::ml::softmax<>::EigenMat;

EMat read_data(std::string const &file)
{
    std::ifstream in(file);
    std::cout<<"read file\n";
    if(in.is_open()){
        std::cout<<"is open\n";
        EMat output(30, 284);
        for(size_t col = 0; col != output.cols(); ++col){
            for(size_t row = 0; row != output.rows(); ++row){
                in>>output(row, col);
            }
        }

        return output;
    }else{
        std::cout<<"cannot open file : "<<file<<"\n";
    }

    return softmax<>::EigenMat();
}

std::vector<int> read_label(std::string const &file)
{
    std::ifstream in(file);
    std::vector<double> output;
    //not the most efficient way, but easier to write
    std::copy(std::istream_iterator<double>(in),
              std::istream_iterator<double>(),
              std::back_inserter(output));

    return std::vector<int>(output.begin(),
                            output.end());
}

}

void softmax_test()
{
    ocv::ml::softmax<> sm;    
    sm.set_lambda(0.0);
    sm.train(read_data("softmax_train_data.txt"),
             read_label("softmax_train_label.txt"));
    auto const TestData =
            read_data("softmax_test_data.txt");
    auto const TestLabel =
            read_label("softmax_test_label.txt");
    double correct = 0;
    for(size_t i = 0; i != TestLabel.size(); ++i){
        auto const Result =
                sm.predict(TestData.block(0, i, TestData.rows(),
                                          1));
        if(Result == TestLabel[i]){
            ++correct;
        }        
    }

    std::cout<<"results : "<<
               (correct/TestLabel.size())<<"\n";
}
