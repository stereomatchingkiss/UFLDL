#include "self_taught_learning.hpp"

#include "mnist_reader.hpp"

#include <shark/Data/Pgm.h> //for exporting the learned filters
#include <shark/Data/Statistics.h> //for normalization
#include <shark/ObjectiveFunctions/SparseAutoencoderError.h>//the error function performing the regularisation of the hidden neurons
#include <shark/Algorithms/GradientDescent/LBFGS.h>// the L-BFGS optimization algorithm
#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/Algorithms/Trainers/RFTrainer.h> //the random forest trainer
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h> // squared loss used for regression
#include <shark/ObjectiveFunctions/Regularizer.h> //L2 regulariziation

#include <shark/Core/Timer.h> //measures elapsed time
#include <shark/Models/Softmax.h>

#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //zero one loss for evaluation

#include <boost/archive/text_oarchive.hpp>

namespace{

using MTSU = mnist_to_shark_vector<>;

using Autoencoder1 =
shark::Autoencoder<shark::LogisticNeuron, shark::LogisticNeuron>;

using Autoencoder2 =
shark::TiedAutoencoder<shark::LogisticNeuron, shark::LogisticNeuron>;

using Autoencoder3 =
shark::Autoencoder<
shark::DropoutNeuron<shark::LogisticNeuron>,
shark::LogisticNeuron
>;

using Autoencoder4 =
shark::TiedAutoencoder<
shark::DropoutNeuron<shark::LogisticNeuron>,
shark::LogisticNeuron
>;

bool read_mnist_data(std::string const &data,
                     std::string const &label,
                     MTSU *me,
                     std::vector<int> *labels)
{
    if(read_mnist(data, std::ref(*me))){
        *labels = read_mnist_label(label);
        if(labels->empty()){
            std::cerr<<"cannot read label\n";
            return false;
        }
        return true;
    }else{
        std::cerr<<"cannot read data\n";
    }

    return false;
}

template<typename Model>
void initialize_ffnet(Model *model)
{
    // Set the starting point for the optimizer. This is 0 for all bias
    // weights and in the interval [-r, r] for non-bias weights.
    double const R = std::sqrt(6.0) /
            std::sqrt(model->numberOfHiddenNeurons() +
                      model->inputSize() + 1.0);
    shark::RealVector params(model->numberOfParameters(), 0);
    //we use  here, that the weights of the layers are the first in the vectors
    //std::cout<<model->inputSize()<<", "<<model->outputSize()<<"\n";
    std::size_t hiddenWeights = model->inputSize() + model->outputSize();
    hiddenWeights *= model->numberOfHiddenNeurons();
    //std::cout<<model->numberOfParameters()<<", "<<hiddenWeights<<"\n";
    for(std::size_t i = 0; i != hiddenWeights; ++i){
        params(i) = shark::Rng::uni(-R, R);
    }
    model->setParameterVector(params);
}

template<typename T>
void split_data(std::vector<T> const &data,
                std::vector<int> const &labels,
                std::vector<T> *split_data,
                std::vector<unsigned int> *split_label)
{
    split_data->clear();
    split_label->clear();
    for(size_t i = 0; i != data.size(); ++i){
        if(labels[i] >= 5){
            split_data->emplace_back(data[i] / 255.0);
            split_label->emplace_back(labels[i]);
        }
    }
}

template<typename Model>
void output_model_state(Model const &model)
{
    std::cout << "Model has: " << model.numberOfParameters() << " params." <<"\n";
    std::cout << "Model has: " << model.numberOfHiddenNeurons() << " hidden neurons." <<"\n";
    std::cout << "Model has: " << model.inputSize() << " inputs." <<"\n";
    std::cout << "Model has: " << model.outputSize() << " outputs." <<"\n";
}

template<typename Optimizer, typename Error, typename Model>
void optimize_params(std::string const &encoder_name,
                     size_t iterate,
                     Optimizer *optimizer,
                     Error *error,
                     Model *model)
{
    using namespace shark;

    Timer timer;
    for (size_t i = 0; i < iterate; ++i) {
        optimizer->step(*error);
        std::cout<<i<<" Error: "<<optimizer->solution().value <<"\n";
    }
    std::cout<<"Elapsed time: " <<timer.stop()<<"\n";
    std::cout<<"Function evaluations: "<<error->evaluationCounter()<<"\n";

    exportFiltersToPGMGrid(encoder_name, model->encoderMatrix(), 28, 28);
    std::ofstream out(encoder_name);
    boost::archive::polymorphic_text_oarchive oa(out);
    model->write(oa);
}

template<typename Model>
void train_autoencoder(std::vector<shark::RealVector> const &unlabel_data,
                       std::string const &encoder_name,
                       Model *model)
{
    using namespace shark;

    model->setStructure(unlabel_data[0].size(), 200);

    //Do not know which part is not thread safe, so I lock all of them
    //except of the time consuming training loop
    initRandomUniform(*model, -0.1*std::sqrt(1.0/unlabel_data[0].size()),
            0.1*std::sqrt(1.0/unlabel_data[0].size()));


    SquaredLoss<RealVector> loss;
    UnlabeledData<RealVector> const Samples = createDataFromRange(unlabel_data);
    RegressionDataset data(Samples, Samples);

    ErrorFunction error(data, model, &loss);
    // Add weight regularization
    const double lambda = 0.01; // Weight decay paramater
    TwoNormRegularizer regularizer(error.numberOfVariables());
    error.setRegularizer(lambda, &regularizer);

    //output some info of model, like number of params, input size etc
    //output_model_state(*model);

    IRpropPlusFull optimizer;
    optimizer.init(error);
    optimize_params(encoder_name, 200, &optimizer, &error, model);
}

template<typename Model>
void train_sparse_autoencoder(std::vector<shark::RealVector> const &unlabel_data,
                              std::string const &encoder_name,
                              Model *model)
{
    using namespace shark;

    //std::cout<<"set structure\n";
    model->setStructure(unlabel_data[0].size(), 200);

    //Do not know which part is not thread safe, so I lock all of them
    //except of the time consuming training loop
    if(std::is_same<Model, Autoencoder2>::value ||
            std::is_same<Model, Autoencoder4>::value){
        //std::cout<<"init ffnet by random uniform\n";
        initRandomUniform(*model, -0.1*std::sqrt(1.0/unlabel_data[0].size()),
                0.1*std::sqrt(1.0/unlabel_data[0].size()));
    }else{
        //std::cout<<"init ffnet\n";
        initialize_ffnet(model);
    }

    //std::cout<<"create data from range\n";
    SquaredLoss<RealVector> loss;
    UnlabeledData<RealVector> const Samples = createDataFromRange(unlabel_data);
    //std::cout<<"gen regression dataset\n";
    RegressionDataset data(Samples, Samples);

    const double Rho = 0.01; // Sparsity parameter
    const double Beta = 6.0; // Regularization parameter
    //std::cout<<"set up error\n";
    SparseAutoencoderError error(data, model, &loss, Rho, Beta);
    // Add weight regularization
    const double lambda = 0.01; // Weight decay paramater
    //std::cout<<"setup regularizer\n";
    TwoNormRegularizer regularizer(error.numberOfVariables());
    error.setRegularizer(lambda, &regularizer);

    //output some info of model, like number of params, input size etc
    //output_model_state(*model);

    //std::cout<<"init optimizer\n";
    LBFGS optimizer;
    optimizer.lineSearch().lineSearchType() = LineSearch::WolfeCubic;
    optimizer.init(error);
    //std::cout<<"run optimizer loop\n";
    optimize_params(encoder_name, 400, &optimizer, &error, model);
}

template<typename Model>
void prediction(std::string const &encoder_file,
                std::string const &rtree_file,
                std::vector<shark::RealVector> const &train_data,
                std::vector<unsigned int> const &train_label,
                Model *model,
                bool reuse_rtree = false)
{
    using namespace shark;
    {
        std::ifstream in(encoder_file);
        boost::archive::polymorphic_text_iarchive ia(in);
        model->read(ia);
    }
    ClassificationDataset train =
            createLabeledDataFromRange(train_data, train_label);
    train.inputs() = model->encode(train.inputs());

    RFClassifier rf_model;
    if(reuse_rtree){
        std::ifstream in2(rtree_file);
        boost::archive::polymorphic_text_iarchive ia2(in2);
        std::cout<<"begin to read "<<rtree_file<<"\n";
        rf_model.read(ia2);
    }else{
        RFTrainer trainer;
        trainer.train(rf_model, train);

        std::ofstream out(rtree_file);
        boost::archive::polymorphic_text_oarchive oa(out);
        rf_model.write(oa);
    }

    std::cout<<"begin to predict\n";
    ZeroOneLoss<unsigned int, RealVector> loss;
    Data<RealVector> prediction = rf_model(train.inputs());
    std::cout<<"Random Forest on training set accuracy: "
            <<1. - loss.eval(train.labels(), prediction)<<"\n";

    MTSU me;
    std::vector<int> test_labels;
    if(read_mnist_data("mnist/t10k-images.idx3-ubyte",
                       "mnist/t10k-labels.idx1-ubyte",
                       &me,
                       &test_labels)){
        std::vector<RealVector> test_data;
        std::vector<unsigned int> utest_labels;
        split_data(me.mat_, test_labels,
                   &test_data, &utest_labels);
        me.mat_.swap(std::vector<RealVector>());
        ClassificationDataset test_data_set =
                createLabeledDataFromRange(test_data, utest_labels);
        test_data_set.inputs() = model->encode(test_data_set.inputs());
        prediction = rf_model(test_data_set.inputs());
        std::cout<<"Random Forest on test set accuracy: "
                <<1. - loss.eval(test_data_set.labels(), prediction) <<"\n";
    }else{
        std::cout<<"cannot read test data\n";
    }
    std::cout<<"\n";
}

void autoencoder_prediction(std::vector<shark::RealVector> const &train_data,
                            std::vector<unsigned int> const &train_label)
{
    {
        Autoencoder1 model;
        train_autoencoder(train_data, "ls_ls.txt", &model);
        prediction("ls_ls.txt", "ls_ls_rtree.txt", train_data,
                   train_label, &model);
    }

    {
        Autoencoder2 model;
        train_autoencoder(train_data, "tied_ls_ls.txt", &model);
        prediction("tied_ls_ls.txt", "tied_ls_ls_rtree.txt", train_data,
                   train_label, &model);
    }

    //Autoencoder3 has bug, the prediction will stuck and cannot complete
    //Do not not it is cause by Shark3.0 beta or my fault
    /*
    {
        Autoencoder3 model;
        train_autoencoder(train_data, "dropls_ls.txt", &model);
        prediction("dropls_ls.txt", "dropls_ls_rtree.txt", train_data,
                   train_label, &model);
    });//*/

    {
        Autoencoder4 model;
        train_autoencoder(train_data, "tied_dropls_ls.txt", &model);
        prediction("tied_dropls_ls.txt", "tied_dropls_ls_rtree.txt", train_data,
                   train_label, &model);
    }
}

void sparse_autoencoder_prediction(std::vector<shark::RealVector> const &train_data,
                                   std::vector<unsigned int> const &train_label)
{
    {
        Autoencoder1 model;
        train_sparse_autoencoder(train_data, "sparse_ls_ls.txt", &model);
        prediction("sparse_ls_ls.txt", "sparse_ls_ls_rtree.txt", train_data,
                   train_label, &model);
    }

    {
        Autoencoder2 model;
        train_sparse_autoencoder(train_data, "sparse_tied_ls_ls.txt", &model);
        prediction("sparse_tied_ls_ls.txt", "sparse_tied_ls_ls_rtree.txt", train_data,
                   train_label, &model);
    }

    {
        Autoencoder3 model;
        train_sparse_autoencoder(train_data, "sparse_dropls_ls.txt", &model);
        prediction("sparse_dropls_ls.txt", "sparse_dropls_ls_rtree.txt", train_data,
                   train_label, &model);
    }


    {
        Autoencoder4 model;
        train_sparse_autoencoder(train_data, "sparse_tied_dropls_ls.txt", &model);
        prediction("sparse_tied_dropls_ls.txt", "sparse_tied_dropls_ls_rtree.txt",
                   train_data, train_label, &model);
    }
}

}

void self_taught_learning()
{
    // Random needs a seed
    shark::Rng::seed(42);

    MTSU me;
    std::vector<int> labels;
    if(read_mnist_data("mnist/train-images.idx3-ubyte",
                       "mnist/train-labels.idx1-ubyte",
                       &me,
                       &labels)){
        using namespace shark;

        std::vector<RealVector> unlabel_data;
        std::vector<unsigned int> unlabel_data_label;
        //pick the digits from 5~9
        split_data(me.mat_, labels,
                   &unlabel_data, &unlabel_data_label);
        //release the memory asap since this test will eat up lot of rams
        me.mat_.swap(std::vector<RealVector>());

        //autoencoder_prediction(unlabel_data, unlabel_data_label);
        sparse_autoencoder_prediction(unlabel_data, unlabel_data_label);
    }
}
