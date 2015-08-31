#include "mnist_reader.hpp"

namespace{

int reverse_int(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) +
            ((int)ch3 << 8) + ch4;
}

}

bool read_mnist(const std::string &file_name,
                std::function<void(std::ifstream &, int, int, int)> func)
{
    std::ifstream in(file_name, std::ios::binary);
    if (in.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        in.read((char*) &magic_number,
                sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        in.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        in.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        in.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);

        func(in, number_of_images, n_rows, n_cols);

        return true;
    }

    return false;
}


std::vector<int> read_mnist_label(const std::string &file_name)
{
    std::ifstream file(file_name, std::ios::binary);
    std::vector<int> labels;
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            labels.emplace_back(temp);
        }
    }

    return labels;
}
