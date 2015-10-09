#include "self_taught_learning_mlpack.hpp"

#include <ocv_libs/profile/measure.hpp>

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>
#include <mlpack/methods/sparse_autoencoder/sparse_autoencoder.hpp>
#include <mlpack/methods/sparse_autoencoder/sparse_autoencoder_function.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression_function.hpp>

#include <mlpack/methods/finetune/finetune.hpp>
#include <mlpack/methods/finetune/softmax_finetune.hpp>

#include "mnist_reader.hpp"

namespace{

using M2Arma = mnist_to_arma<arma::mat>;

using SAEF = mlpack::nn::SparseAutoencoderFunction<>;
using SAE = mlpack::nn::SparseAutoencoder<>;
using SM = mlpack::regression::SoftmaxRegression<>;
using SMF = mlpack::regression::SoftmaxRegressionFunction;

using SAEFOptimizer = mlpack::optimization::L_BFGS<SAEF>;
using SMOptimizer = mlpack::optimization::L_BFGS<SMF>;

std::unique_ptr<SAE> sae1;
std::unique_ptr<SAE> sae2;
std::unique_ptr<SM> softmax;

std::unique_ptr<SAEF> saef1;
std::unique_ptr<SAEF> saef2;
std::unique_ptr<SMF> smf;

M2Arma test_data;
M2Arma train_data;

arma::mat sae1_features;
arma::mat sae2_features;
arma::Row<size_t> train_labels(10000);

bool read_mnist_data(std::string const &data,
                     std::string const &label,
                     M2Arma &me,
                     std::vector<int> &labels)
{
    if(read_mnist(data, std::ref(me))){
        labels = read_mnist_label(label);
        if(labels.empty()){
            std::cerr<<"cannot read label\n";
            return false;
        }
        return true;
    }else{
        std::cerr<<"cannot read data\n";
    }

    return false;
}

size_t train_autoencoder(arma::mat &input,
                         arma::mat &features,
                         std::unique_ptr<SAEF> &saef,
                         std::unique_ptr<SAE> &sae)
{
    size_t const visibleSize = input.n_rows;
    size_t const hiddenSize = 200;
    double const lambda = 0.003;
    double const beta = 3;
    double const rho = 0.1;
    saef = std::make_unique<SAEF>(input, visibleSize,
                                  hiddenSize, lambda, beta, rho);
    size_t numBasis = 5;
    size_t numIteration = 350;
    SAEFOptimizer optimizer(*saef, numBasis, numIteration);
    size_t const Duration = ocv::time::measure<>::execution([&]()
    {
        sae = std::make_unique<SAE>(optimizer);
    });
    sae->GetNewFeatures(input, features);

    return Duration;
}

void pretrain()
{
    std::vector<int> training_labels;
    if(read_mnist_data("mnist/train-images.idx3-ubyte",
                       "mnist/train-labels.idx1-ubyte",
                       train_data, training_labels)){

        train_data.mat_ = arma::resize(train_data.mat_, 28*28, 10000);
        training_labels.resize(train_data.mat_.n_cols);
        const size_t numClasses = 10;
        train_labels.set_size(training_labels.size());
        for(size_t i = 0; i != train_labels.n_elem; ++i){
            train_labels(i) = training_labels[i];
        }

        bool load_data = true;
        if(!load_data){
            size_t const Duration1 =
                    train_autoencoder(train_data.mat_, sae1_features, saef1, sae1);
            std::cout<<"Duration1 : "<<Duration1/1000<<"\n";
            size_t const Duration2 =
                    train_autoencoder(sae1_features, sae2_features, saef2, sae2);
            std::cout<<"Duration2 : "<<Duration2/1000<<"\n";
        }else{
            sae1 = std::make_unique<SAE>(train_data.mat_.n_rows, 200);
            sae2 = std::make_unique<SAE>(200, 200);
            mlpack::data::Load("sae1.txt", "sae", *sae1, true);
            mlpack::data::Load("sae2.txt", "sae", *sae2, true);
            sae1->GetNewFeatures(train_data.mat_, sae1_features);
            sae2->GetNewFeatures(sae1_features, sae2_features);
        }

        const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
        const size_t numIterations = 400; // Maximum number of iterations.

        // Use an instantiated optimizer for the training.
        using SRF = mlpack::regression::SoftmaxRegressionFunction;

        //srf_ = std::make_unique<SRF>(train_data_2_, labels, numClasses);
        smf = std::make_unique<SMF>(sae2_features, train_labels, numClasses);
        SMOptimizer optimizer(*smf, numBasis, numIterations);
        size_t const Duration3 = ocv::time::measure<>::execution([&]()
        {
            softmax = std::make_unique<SM>(optimizer);
        });
        std::cout<<"softmax train duration : "<<Duration3/1000<<"\n";

        if(!load_data){
            mlpack::data::Save("sae1.txt", "sae", *sae1, true);
            mlpack::data::Save("sae2.txt", "sae", *sae2, true);
            //mlpack::data::Save("sm.txt", "sm", *softmax, true);
        }
    }
}

void predicts()
{
    std::vector<int> testing_labels;
    if(read_mnist_data("mnist/t10k-images.idx3-ubyte",
                       "mnist/t10k-labels.idx1-ubyte",
                       test_data, testing_labels)){
        arma::mat inputs;
        sae1->GetNewFeatures(test_data.mat_, inputs);
        sae2->GetNewFeatures(inputs, inputs);
        arma::vec predicts;
        softmax->Predict(inputs, predicts);
        double bingo = 0;
        for(size_t i = 0; i != predicts.n_elem; ++i){
            if(std::abs(predicts[i] - testing_labels[i]) < 1e-2){
                ++bingo;
            }
        }
        std::cout<<"accuracy : "<<(bingo)/double(predicts.n_elem)<<"\n";
    }
}

void finetune_train()
{
    std::vector<arma::mat*> inputs{&train_data.mat_, &sae1_features,
                &sae2_features};
    std::vector<arma::mat*> params{&sae1->Parameters(),
                &sae2->Parameters(),
                &softmax->Parameters()};
    std::cout<<arma::size(*params[0])<<"\n";
    std::cout<<arma::size(*params[1])<<"\n";
    std::cout<<arma::size(*params[2])<<"\n";

    using FineTune = mlpack::nn::FineTuneFunction<SMF, mlpack::nn::SoftmaxFineTune>;
    std::cout<<"create finetune\n";
    FineTune finetune(inputs, params, *smf);    
    size_t numBasis = 5;
    size_t numIteration = 200;
    mlpack::optimization::L_BFGS<FineTune> optimizer(finetune, numBasis,
                                                     numIteration);

    size_t const Duration = ocv::time::measure<>::execution([&]()
    {
        auto parameters = finetune.GetInitialPoint();
        optimizer.Optimize(parameters);
        finetune.UpdateParameters(parameters);
    });
    std::cout<<"fine tune duration : "<<Duration<<"\n";//*/
}

}

void self_taught_learning_mlpack()
{
    pretrain();
    std::cout<<"-----------before finetune-----------------\n";
    predicts();
    std::cout<<"-----------after finetune-----------------\n";
    finetune_train();
    predicts();
}
