#include "speckle_gen.h"

/**
 * @brief generate Random disk center coordinates
 * 
 * @param output output array
 * @param kappa output coverage ratio
 * @param theta output expected perimeter
 * @param myParamSpeckle speckle parameters
 * @param dims image dimensions
 * @param number toatal number of generated disks
 * @param seed PRNG seed

 * @return void
*/
void generate_random_radius(float output[], float *kappa, float *theta, paramSpeckle<float> myParamSpeckle, vec2D<int> dims, int number, unsigned int seed)
{

    std::mt19937_64 generator(seed);
    float mu = myParamSpeckle.mu;
    float sigmaR = myParamSpeckle.sigmaR;
    float lambda = myParamSpeckle.lambda;
    switch (myParamSpeckle.distribR)
    {
    case 'E':
    {
        printf("Exponential distribution for radii\n");
        std::exponential_distribution<float> distribution(1 / mu); // lambda of the exponential distribution = 1/mu (mu is the mean)
        for (int i = 0; i < number; ++i)
        {
            output[i] = (float)distribution(generator);
        }
        (*kappa) = (float)(1 - exp(-2 * pi * lambda / (float)(dims.x * dims.y) * mu * mu));
        (*theta) = (float)(2 * pi * lambda * mu * exp(-2 * pi * lambda / (float)(dims.x * dims.y) * mu * mu));
        break;
    }
    case 'U':
    {
        printf("Uniform distribution for radii\n");
        std::uniform_real_distribution<float> distribution;
        for (int i = 0; i < number; ++i)
        {
            output[i] = (float)2 * mu * distribution(generator);
        }
        (*kappa) = (float)(1 - exp(-4 * pi * lambda / (float)(dims.x * dims.y) * mu * mu / 3));
        (*theta) = (float)(2 * pi * lambda * mu * exp(-4 * pi * lambda / (float)(dims.x * dims.y) * mu * mu / 3));
        break;
    }
    case 'P':
    {
        printf("Poisson distribution for radii\n");
        std::poisson_distribution<int> distribution(mu); // by default, th type is "int"
        for (int i = 0; i < number; ++i)
        {
            output[i] = (float)distribution(generator);
        }
        (*kappa) = (float)(1 - exp(-2 * pi * lambda / (float)(dims.x * dims.y) * (mu + mu * mu)));
        (*theta) = (float)(2 * pi * lambda * mu * exp(-2 * pi * lambda / (float)(dims.x * dims.y) * (mu + mu * mu)));
        break;
    }
    case 'L':
    {
        printf("Log-normal distribution for radii\n");
        float meanlogR = log(mu * mu / sqrt(sigmaR * sigmaR + mu * mu));
        float stdlogR = log(mu * mu / sqrt(sigmaR * sigmaR + mu * mu));
        std::lognormal_distribution<float> distribution(meanlogR, stdlogR);
        for (int i = 0; i < number; ++i)
        {
            output[i] = (float)distribution(generator);
        }
        (*kappa) = (float)(1 - exp(-pi * lambda / (float)(dims.x * dims.y) * (sigmaR * sigmaR + mu * mu)));
        (*theta) = (float)(2 * pi * lambda * mu * exp(-pi * lambda / (float)(dims.x * dims.y) * (sigmaR * sigmaR + mu * mu)));
        break;
    }
    }
}



/*
Normal inverse cumulative distribution function (see Algorithm in https://fr.mathworks.com/help/stats/norminv.html#mw_06d38ca9-8aa3-44b8-bb8e-549e1c9b4fde)
Using Inverse ERandom_radiusor Function (erfc_inv) from the Boost library ( for details check https://www.boost.org/doc/libs/1_68_0/libs/math/doc/html/math_toolkit/sf_erf/eRandom_radiusor_inv.html)

+ TO install Boost lib on linux ==> sudo apt-get install libboost-all-dev 
*/

float icdf(float p)
{
    return (float)(-sqrt(2) * boost::math::erfc_inv(2 * p));
}


/**
 * @brief generate Boolean model and RBound
 * 
 * @param Random_centers output generated disk centers from Boolean model
 * @param Random_radius output generated disk radii from Boolean model
 * @param RBound output RBound array
 * @param myParamSpeckle speckle parameters
 * @param myParamAlgo Algorithm parameters
 * @param myParamSensor Sensor parameters
 * @param number toatal number of generated disks
 * @param seed PRNG seed

 * @return void
*/

void boolean_model(float *Random_centers, float *Random_radius, float *RBound, paramSpeckle<float> myParamSpeckle, paramAlgo<float> myParamAlgo, paramSensor<float> myParamSensor, int number, unsigned int seed)
{
    // init local variables
    vec2D<int> dims = myParamSensor.dims;
    float alpha = myParamAlgo.alpha;
    int nbit = myParamSensor.nbit;
    float gamma = myParamSpeckle.gamma;
    float sigma = myParamSensor.sigma;

    // translate "Random_centers = repmat(dims,[number,1]).*rand(number,2,prec);"
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> distribution;
    for (int i = 0; i < number; ++i)
    {
        Random_centers[2 * i] = (float)dims.x * (float)distribution(generator);     // x of RC
        Random_centers[2 * i + 1] = (float)dims.y * (float)distribution(generator); // y of RC
    }

    float kappa = 0, theta = 0;
    // generate random radii
    generate_random_radius(Random_radius, &kappa, &theta, myParamSpeckle, dims, number, seed);

    printf("coverage ratio: %f\nexpected perimeter: %f\n", kappa, theta);
    printf("largest Monte Carlo sample size: %d\n", (int)floor((float)((float)1 / pi / 2 * gamma * gamma * pow(2, (2 * nbit)) / (alpha * alpha))));

    // estimate delta
    float delta = estimate_delta(dims);
    printf("Estimated delta : %E\n", delta);

    // estimate B
    float tmp_icdf = icdf(1 - pi * alpha / sqrt(2) / pow(2, nbit) / 10);
    printf("Estimated B : %.4f * sigma\n", tmp_icdf); // This should print 3.7547 for alpha=0.1 and nbit=8 (the same result as Matlab function)

    for (int i = 0; i < number; ++i)
    {
        RBound[i] = pow(Random_radius[i] + (1 + delta) * tmp_icdf * sigma, 2);
    }
}


/**
 * @brief generate Boolean model and RBound
 * 
 * @param Random_centers output generated disk centers from Boolean model
 * @param Random_radius output generated disk radii from Boolean model
 * @param RBound output RBound array
 * @param myParamSpeckle speckle parameters
 * @param myParamAlgo Algorithm parameters
 * @param myParamSensor Sensor parameters
 * @param number toatal number of generated disks
 * @param seed PRNG seed
 * @param disp_map_x x-displacement map
 * @param disp_map_y y-displacement map

 * @return void
*/
void boolean_model_mesh(float *Random_centers, float *Random_radius, float *RBound, paramSpeckle<float> myParamSpeckle, paramAlgo<float> myParamAlgo, paramSensor<float> myParamSensor, int number, unsigned int seed, float *disp_map_x, float *disp_map_y)
{
    // init local variables
    vec2D<int> dims = myParamSensor.dims;
    float alpha = myParamAlgo.alpha;
    int nbit = myParamSensor.nbit;
    float gamma = myParamSpeckle.gamma;
    float sigma = myParamSensor.sigma;

    // translate "Random_centers = repmat(dims,[number,1]).*rand(number,2,prec);"
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> distribution;
    for (int i = 0; i < number; ++i)
    {
        Random_centers[2 * i] = (float)dims.x * (float)distribution(generator);     // x of RC
        Random_centers[2 * i + 1] = (float)dims.y * (float)distribution(generator); // y of RC
    }

    float kappa = 0, theta = 0;
    // generate random radii
    generate_random_radius(Random_radius, &kappa, &theta, myParamSpeckle, dims, number, seed);

    printf("coverage ratio: %f\nexpected perimeter: %f\n", kappa, theta);
    printf("largest Monte Carlo sample size: %d\n", (int)floor((float)((float)1 / pi / 2 * gamma * gamma * pow(2, (2 * nbit)) / (alpha * alpha))));

    // estimate delta
    float delta = estimate_delta_mesh(dims, disp_map_x, disp_map_y);
    printf("Estimated delta : %E\n", delta);

    // estimate B
    float tmp_icdf = icdf(1 - pi * alpha / sqrt(2) / pow(2, nbit) / 10);
    printf("Estimated B : %.4f * sigma\n", tmp_icdf); // This should print 3.7547 for alpha=0.1 and nbit=8 (the same result as Matlab function)

    for (int i = 0; i < number; ++i)
    {
        RBound[i] = pow(Random_radius[i] + (1 + delta) * tmp_icdf * sigma, 2);
    }
}


/**
 * @brief quantization of the image matrix over nbit
 * 
 * @param img_out output quantized image
 * @param speckle_matrix input speckle matrix
 * @param img_size image size (width x height)
 * @param nbit quantization depth

 * @return void
*/
void quantization(float *img_out, float *speckle_matrix, int img_size, int nbit)
{
    for (int i = 0; i < img_size; ++i)
    {
        img_out[i] = round(speckle_matrix[i]);

        if (img_out[i] < 0)
        {
            img_out[i] = 0;
        }

        if (img_out[i] > pow(2, nbit) - 1)
        {
            img_out[i] = pow(2, nbit) - 1;
        }
    }
}