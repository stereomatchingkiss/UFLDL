#include <mlpack/core.hpp>

#include "autoencoder_test.hpp"
#include "softmax_test.hpp"
#include "stacked_autoencoder.hpp"

#include <iostream>

int main()
{            
    /*arma::mat test_input = arma::randu<arma::mat>(3, 2);
    test_struct test(test_input);
    std::cout<<test.data<<"\n\n";
    test_input(0, 0) = 100;
    std::cout<<test.data<<"\n\n";*/

    //autoencoder_test();
    //self_taught_learning();
    stacked_autoencoder();
    //softmax_test();    

    return 0;
}
