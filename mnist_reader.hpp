#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <fstream>
#include <functional>
#include <string>
#include <vector>

template<typename EigenMat>
struct mnist_to_eigen
{
public:
    void operator()(std::ifstream &in, int number_of_images,
                    int n_rows, int n_cols);

    EigenMat mat_;
};

template<typename EigenMat>
void mnist_to_eigen<EigenMat>::
operator()(std::ifstream &in, int number_of_images,
           int n_rows, int n_cols)
{
    using Scalar = typename EigenMat::Scalar;    
    mat_.resize(n_rows * n_cols, number_of_images);    
    for(int i = 0; i < number_of_images; ++i){
        for(int r = 0; r < n_rows; ++r){
            int const Offset = r * n_cols;
            for(int c = 0; c < n_cols; ++c){
                unsigned char temp = 0;
                in.read((char*) &temp, sizeof(temp));
                mat_(Offset + c, i) = Scalar(temp);
            }
        }
    }
}

bool
read_mnist(std::string const &file_name,
           std::function<void(std::ifstream&, int, int, int)> func);

std::vector<int> read_mnist_label(std::string const &file_name);

#endif // MNIST_READER_HPP

