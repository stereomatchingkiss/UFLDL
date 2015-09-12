#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <shark/Data/Dataset.h>

#include <fstream>
#include <functional>
#include <string>

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

template<typename T = shark::RealVector>
struct mnist_to_shark_vector
{
    void operator()(std::ifstream &in, int number_of_images,
                    int n_rows, int n_cols);

    std::vector<T> mat_;
};

template<typename T>
void mnist_to_shark_vector<T>::
operator()(std::ifstream &in, int number_of_images,
                int n_rows, int n_cols)
{
    using ScalarType = typename T::scalar_type;
    for(int i = 0; i < number_of_images; ++i){
        T sample(n_rows * n_cols);
        for(int r = 0; r < n_rows; ++r){
            int const Offset = r * n_cols;
            for(int c = 0; c < n_cols; ++c){
                unsigned char temp = 0;
                in.read((char*) &temp, sizeof(temp));
                sample(Offset + c) = float(temp);
            }
        }
        mat_.emplace_back(std::move(sample));
    }
}

bool
read_mnist(std::string const &file_name,
           std::function<void(std::ifstream&, int, int, int)> func);

std::vector<int> read_mnist_label(std::string const &file_name);

#endif // MNIST_READER_HPP

