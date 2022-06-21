
#include <string>
#include <random>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include "io_png.h"
#include "boolean_model.h"

#include <omp.h>

//////// MUST have the SAME declaration as in "MC_estimation_cuda.cu" ////////////////////////// 

/*
mapping(displacement/deformation) function
*/
template <typename T>
void func(T x_in, T y_in, T *x_out, T *y_out)
{
    // identity function
    *x_out = x_in;
    *y_out = y_in;
}

//////////////////////////////////////////////////////////////////////////////


template <typename T>
void generate_random_radius(T output[], T *kappa, T *theta, paramSpeckle<T> myParamSpeckle, vec2D<int> dims, int number, unsigned int seed)
{

    std::mt19937_64 generator(seed);
    T mu = myParamSpeckle.mu;
    T sigmaR = myParamSpeckle.sigmaR;
    T lambda = myParamSpeckle.lambda;
    switch (myParamSpeckle.distribR)
    {
    case 'E':
    {
        printf("Exponential distribution for radii\n");
        std::exponential_distribution<T> distribution(1 / mu); // lambda of the exponential distribution = 1/mu (mu is the mean)
        for (int i = 0; i < number; ++i)
        {
            output[i] = (T)distribution(generator);
        }
        (*kappa) = (T)(1 - exp(-2 * pi * lambda / (T)(dims.x * dims.y) * mu * mu));
        (*theta) = (T)(2 * pi * lambda * mu * exp(-2 * pi * lambda / (T)(dims.x * dims.y) * mu * mu));
        break;
    }
    case 'U':
    {
        printf("Uniform distribution for radii\n");
        std::uniform_real_distribution<T> distribution;
        for (int i = 0; i < number; ++i)
        {
            output[i] = (T)2 * mu * distribution(generator);
        }
        (*kappa) = (T)(1 - exp(-4 * pi * lambda / (T)(dims.x * dims.y) * mu * mu / 3));
        (*theta) = (T)(2 * pi * lambda * mu * exp(-4 * pi * lambda / (T)(dims.x * dims.y) * mu * mu / 3));
        break;
    }
    case 'P':
    {
        printf("Poisson distribution for radii\n");
        std::poisson_distribution<int> distribution(mu); // by default, th type is "int"
        for (int i = 0; i < number; ++i)
        {
            output[i] = (T)distribution(generator);
        }
        (*kappa) = (T)(1 - exp(-2 * pi * lambda / (T)(dims.x * dims.y) * (mu + mu * mu)));
        (*theta) = (T)(2 * pi * lambda * mu * exp(-2 * pi * lambda / (T)(dims.x * dims.y) * (mu + mu * mu)));
        break;
    }
    case 'L':
    {
        printf("Log-normal distribution for radii\n");
        T meanlogR = log(mu * mu / sqrt(sigmaR * sigmaR + mu * mu));
        T stdlogR = log(mu * mu / sqrt(sigmaR * sigmaR + mu * mu));
        std::lognormal_distribution<T> distribution(meanlogR, stdlogR);
        for (int i = 0; i < number; ++i)
        {
            output[i] = (T)distribution(generator);
        }
        (*kappa) = (T)(1 - exp(-pi * lambda / (T)(dims.x * dims.y) * (sigmaR * sigmaR + mu * mu)));
        (*theta) = (T)(2 * pi * lambda * mu * exp(-pi * lambda / (T)(dims.x * dims.y) * (sigmaR * sigmaR + mu * mu)));
        break;
    }
    }
}

/*
delta estimation function
@fun : mapping function (should be declared before. For usage as a parameter in the following function, declare it as "fun<T>" not as "fun")
@dims : image dimensions
*/
template <typename T, typename F>
T estimate_delta(vec2D<int> dims, F fun)
{
    const int width = dims.x;
    const int height = dims.y;
    const int size = width * height;
    T *dX, *dY;
    dX = (T *)malloc(size * sizeof(T));
    dY = (T *)malloc(size * sizeof(T));

    for (int row = 0; row < height; ++row)
    {
        T y = row + 1; // fix the same "y" value in each row
        for (int column = 0; column < width; ++column)
        {
            T x = column + 1; // fix the same "x" value in each column
            T fx = 0, fy = 0; // outputs of "fun"
            fun(x, y, &fx, &fy);
            dX[column + row * width] = fx - x;
            dY[column + row * width] = fy - y;
        }
    }

    T *sqF;
    sqF = (T *)malloc((width - 1) * (height - 1) * sizeof(T));
    T max = -INFINITY;
    T sqF_r_c = 0;
    // Matlab equivalent ==>  diff(dX(:,2:end),1,1).^2 + diff(dY(:,2:end),1,1).^2 + diff(dX(2:end,:),1,2).^2 + diff(dY(2:end,:),1,2).^2;
    for (int row = 0; row < height - 1; ++row)
    {
        for (int column = 0; column < width - 1; ++column)
        {
            sqF_r_c = pow(dX[(column + 1) + row * width] - dX[(column + 1) + (row + 1) * width], 2);
            sqF_r_c += pow(dY[(column + 1) + row * width] - dY[(column + 1) + (row + 1) * width], 2);
            sqF_r_c += pow(dX[column + (row + 1) * width] - dX[(column + 1) + (row + 1) * width], 2);
            sqF_r_c += pow(dY[column + (row + 1) * width] - dY[(column + 1) + (row + 1) * width], 2);
            sqF[column + row * (width - 1)] = sqF_r_c;
            max = (sqF[column + row * (width - 1)] > max) ? sqF[column + row * (width - 1)] : max; // calculate max
        }
    }

    free(dX);
    free(dY);
    free(sqF);

    return (T)(sqrt(max));
}

/*
Normal inverse cumulative distribution function (see Algorithm in https://fr.mathworks.com/help/stats/norminv.html#mw_06d38ca9-8aa3-44b8-bb8e-549e1c9b4fde)
Using Inverse ERandom_radiusor Function (erfc_inv) from the Boost library ( for details check https://www.boost.org/doc/libs/1_68_0/libs/math/doc/html/math_toolkit/sf_erf/eRandom_radiusor_inv.html)

+ TO install Boost lib on linux ==> sudo apt-get install libboost-all-dev 
*/
template <typename T>
T icdf(T p)
{
    return (T)(-sqrt(2) * boost::math::erfc_inv(2 * p));
}


/*

@Random_radius: output
*/

template <typename T>
void boolean_model(T *Random_centers, T *Random_radius, T *RBound, paramSpeckle<T> myParamSpeckle, paramAlgo<T> myParamAlgo, paramSensor<T> myParamSensor, int number, std::function<void(T, T, T *, T *)> fun, unsigned int seed)
{
    // init local variables
    vec2D<int> dims = myParamSensor.dims;
    T alpha = myParamAlgo.alpha;
    int nbit = myParamSensor.nbit;
    T gamma = myParamSpeckle.gamma;
    T sigma = myParamSensor.sigma;

    // translate "Random_centers = repmat(dims,[number,1]).*rand(number,2,prec);"
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<T> distribution;
    for (int i = 0; i < number; ++i)
    {
        Random_centers[2 * i] = (T)dims.x * (T)distribution(generator);     // x of RC
        Random_centers[2 * i + 1] = (T)dims.y * (T)distribution(generator); // y of RC
    }

    T kappa = 0, theta = 0;
    // generate random radii
    generate_random_radius<T>(Random_radius, &kappa, &theta, myParamSpeckle, dims, number, seed);

    printf("coverage ratio: %f\nexpected perimeter: %f\n", kappa, theta);
    printf("largest Monte Carlo sample size: %d\n", (int)floor((T)((T)1 / pi / 2 * gamma * gamma * pow(2, (2 * nbit)) / (alpha * alpha))));

    // estimate delta
    T delta = estimate_delta<T>(dims, fun);
    printf("Estimated delta : %E\n", delta);

    // estimate B
    T tmp_icdf = icdf<T>(1 - pi * alpha / sqrt(2) / pow(2, nbit) / 10);
    printf("Estimated B : %.4f * sigma\n", tmp_icdf); // This should print 3.7547 for alpha=0.1 and nbit=8 (the same result as Matlab function)

    for (int i = 0; i < number; ++i)
    {
        RBound[i] = pow(Random_radius[i] + (1 + delta) * tmp_icdf * sigma, 2);
    }

#ifdef DEBUG
    printf("A few samples from Random_radius : ");
    for (int i = 0; i < 10; ++i)
    {
        printf("%f* ", Random_radius[i]);
        printf("%f, ", Random_centers[i]);
        printf("%f| ", RBound[i]);
    }
    printf("\n");
#endif
}

template <typename T>
void quantization(T *img_out, T *speckle_matrix, int img_size, int nbit)
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

///////////////////////////////////////////////////////

static void show_help();

/// help on usage of inpainting code
static void show_help()
{
    std::cerr << "\nSpeckle Generator.\n"
              << "Usage: "
              << " speckle_generator_main imgNameOut.png [options]\n\n"
              << "Options (default values in parentheses)\n"
              << "-prec : data precision (0: float(default), 1:double)\n"
              << "-distribR : probability distribution of the radii ('E' for exponential, 'U' for uniform, 'P' for Poisson, 'L' for log-normal)\n"
              << "-gamma : standard deviation of the radius\n"
              << "-lambda : average number of disks per image\n"
              << "-sigmaR : standard deviation of radius (for 'l' only)\n"
              << "-mu : average radius of disks\n"
              << "-alpha : quantization error probability\n"
              << "-N0 : sample size to estimate s^2\n"
              << "-NMCmax : size of the largest MC sample \n"
              << "-width : output image width\n"
              << "-height : output image height\n"
              << "-nbit : bit depth\n"
              << "-sigmaG : standard deviation of the Gaussian PSF\n"
              << "-seed : seed of the random generator (default: 2020, shuffle: 0)\n"
              << std::endl;
}

/**
 * 
 */
/**
* @brief Find the command option named option
*/
char *getCmdOption(char **begin, char **end, const std::string &option)
{
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

/**
 * 
 */
/**
* @brief Check for input parameter
*
* @param beginning of input command chain
* @param end of input command chain
* @return whether the parameter exists or not
*/
bool cmdOptionExists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

/**
 * 
 */
/**
* @brief Get file exension of file name
*
* @param File name
* @return File extension
*/
std::string getFileExt(const std::string &s)
{
    size_t i = s.rfind('.', s.length());
    if (i != std::string::npos)
    {
        return (s.substr(i + 1, s.length() - i));
    }
    else
        return ("");
}

/**
 * 
 */
/**
* @brief Get file name, without extension
*
* @param Input file name
* @return File name without extension
*/
std::string getFileName(const std::string &s)
{
    size_t i = s.rfind('.', s.length());
    if (i != std::string::npos)
    {
        return (s.substr(0, i));
    }
    else
        return (s);
}

/**
 * 
 */
/**
* @brief Get current directory
*
* @return Current directory name
*/
std::string get_curr_dir()
{
    size_t maxBufSize = 1024;
    char buf[maxBufSize];
    char *charTemp = getcwd(buf, maxBufSize);
    std::string currDir(charTemp);
    return (currDir);
}

/**
 * 
 */
/**
* @brief Write the output to a .tiff or .png image.
*
* @param imgOut output image to write
* @param fileNameOut output file name
* @return 0 if write success, -1 if failure
*/
template <typename T>
int write_output_image(float *imgOut, const std::string fileNameOut,
                       paramSpeckle<T> myParamSpeckle, paramAlgo<T> myParamAlgo, paramSensor<T> myParamSensor)
{
    std::string outputExtension(getFileExt(fileNameOut));

    //write output image
    std::string fileNameOutFull = (char *)((getFileName(fileNameOut) + "." + outputExtension).c_str());
    std::cout << "output file name : " << fileNameOutFull << std::endl;

    if (strcmp((const char *)(outputExtension.c_str()), "png") == 0 ||
        strcmp((const char *)(outputExtension.c_str()), "") == 0) //png files
    {
        if (0 != io_png_write_f32(fileNameOutFull.c_str(), imgOut,
                                  myParamSensor.dims.x, myParamSensor.dims.y, 1)) // 1 channel is used
        {
            std::cout << "Error, could not write the image file." << std::endl;
            return (-1);
        }
    }
    else
    {
        std::cout << "Error, unknown output file extension." << std::endl;
        return (-1);
    }

    return (0);
}

/**
 * @brief speckle generator main function
 * 
 * @tparam T 
 * @param argc 
 * @param argv 
 * @param prec data precision
 * @return int 
 */
template <typename T>
int generator_main(int argc, char *argv[], const char *prec)
{

    //get file name
    std::string fileNameOut(argv[1]);

    T mu, sigma, sigmaR, gamma, alpha, lambda;
    int width, height, nbit, N0, NMCmax;
    unsigned int seed;
    char distribR;

    // init mapping function
    std::function<void(T, T, T *, T *)> fun;
    fun = func<T>;

    //show help
    if (cmdOptionExists(argv, argv + argc, "-h"))
    {
        show_help();
        return (-1);
    }

    /**************************************************/
    /*************   GET INPUT OPTIONS   **************/
    /**************************************************/

    //algorithm params

    if (cmdOptionExists(argv, argv + argc, "-alpha"))
    {
        alpha = (T)atof(getCmdOption(argv, argv + argc, "-alpha"));
    }
    else
        alpha = 0.1;

    if (cmdOptionExists(argv, argv + argc, "-N0"))
    {
        N0 = (int)atoi(getCmdOption(argv, argv + argc, "-N0"));
    }
    else
        N0 = 1001;

    if (cmdOptionExists(argv, argv + argc, "-NMCmax"))
    {
        NMCmax = (int)atoi(getCmdOption(argv, argv + argc, "-NMCmax"));
    }
    else
        NMCmax = (int)pow(2, 24);

    //sensor params
    if (cmdOptionExists(argv, argv + argc, "-width"))
    {
        width = (int)atoi(getCmdOption(argv, argv + argc, "-width"));
    }
    else
        width = 500;

    if (cmdOptionExists(argv, argv + argc, "-height"))
        height = (int)atoi(getCmdOption(argv, argv + argc, "-height"));
    else
        height = 500;

    if (cmdOptionExists(argv, argv + argc, "-nbit"))
        nbit = (int)atoi(getCmdOption(argv, argv + argc, "-nbit"));
    else
        nbit = 8;

    if (cmdOptionExists(argv, argv + argc, "-sigmaG"))
    {
        sigma = (T)atof(getCmdOption(argv, argv + argc, "-sigmaG"));
    }
    else
        sigma = 1;

    if (cmdOptionExists(argv, argv + argc, "-seed"))
    {
        int seed_choice = atoi(getCmdOption(argv, argv + argc, "-seed"));
        if (seed_choice == 0)
        {
            // generate a random seed
            std::random_device rd;
            seed = (unsigned int)rd();
        }
        else
        {
            seed = (unsigned int)seed_choice;
        }
    }
    else
        seed = 2020u;

    //speckle params
    if (cmdOptionExists(argv, argv + argc, "-distribR"))
    {
        distribR = getCmdOption(argv, argv + argc, "-distribR")[0];
    }
    else
        distribR = 'E';

    if (cmdOptionExists(argv, argv + argc, "-gamma"))
    {
        gamma = (T)atof(getCmdOption(argv, argv + argc, "-gamma"));
    }
    else
        gamma = 0.6;

    if (cmdOptionExists(argv, argv + argc, "-lambda"))
    {
        lambda = (T)atof(getCmdOption(argv, argv + argc, "-lambda"));
    }
    else
        lambda = round(5 / 9.0 * width * height); //500000;

    if (cmdOptionExists(argv, argv + argc, "-sigmaR"))
    {
        sigmaR = (T)atof(getCmdOption(argv, argv + argc, "-sigmaR"));
    }
    else
        sigmaR = 0;

    if (cmdOptionExists(argv, argv + argc, "-mu"))
    {
        mu = (T)atof(getCmdOption(argv, argv + argc, "-mu"));
    }
    else
        mu = 0.5;

    /////// init parameters
    paramSpeckle<T> myParamSpeckle;
    paramAlgo<T> myParamAlgo;
    paramSensor<T> myParamSensor;
    vec2D<int> myDims;

    myDims.x = width;
    myDims.y = height;
    myParamSpeckle.distribR = distribR;
    myParamSpeckle.gamma = gamma;
    myParamSpeckle.lambda = lambda;
    myParamSpeckle.mu = mu;
    myParamSpeckle.sigmaR = sigmaR;
    myParamAlgo.alpha = alpha;
    myParamAlgo.N0 = N0;
    myParamAlgo.NMCmax = NMCmax;
    myParamSensor.dims = myDims;
    myParamSensor.nbit = nbit;
    myParamSensor.sigma = sigma;

    //display parameters
    std::cout << "------- Sensor params -------" << std::endl;
    std::cout << "Image size : " << width << " x " << height << std::endl;
    std::cout << "nbit : " << nbit << std::endl;
    std::cout << "sigma : " << sigma << std::endl;
    std::cout << "------- Speckle params -------" << std::endl;
    std::cout << "distribR : " << distribR << std::endl;
    std::cout << "gamma : " << gamma << std::endl;
    std::cout << "lambda : " << lambda << std::endl;
    std::cout << "mu : " << mu << std::endl;
    std::cout << "sigmaR : " << sigmaR << std::endl;
    std::cout << "------- Algorithm params -------" << std::endl;
    std::cout << "alpha : " << alpha << std::endl;
    std::cout << "N0 : " << N0 << std::endl;
    std::cout << "NMCmax : " << NMCmax << std::endl;
    std::cout << "precision : " << prec << std::endl;
    std::cout << "------- Seed -------" << std::endl;
    std::cout << "seed : " << seed << std::endl;

    /**************************************************/
    /*****  TIME AND CARRY OUT GRAIN RENDERING   ******/
    /**************************************************/

    struct timeval start, end;
    gettimeofday(&start, NULL);

    //execute the speckle generation
    std::cout << "***************************" << std::endl;

    T *speckle_matrix;
    speckle_matrix = (T *)malloc(width * height * sizeof(T));

    // Draw the number of disks, following a Poisson distribution of intensity "lambda"
    std::mt19937_64 generator(seed);
    std::poisson_distribution<int> distrib(myParamSpeckle.lambda);
    int number = distrib(generator);
#ifdef DEBUG
    printf("number of disks = %d\n", number);
#endif

    // allocate memory for the generated Random_radius
    T *Random_radius;
    Random_radius = (T *)malloc(number * sizeof(T));
    // generate RBound
    T *RBound;
    RBound = (T *)malloc(number * sizeof(T)); // same size as Random_radius
    // generate RC
    T *Random_centers;
    Random_centers = (T *)malloc(2 * number * sizeof(T)); // this is a vector with x (pair indexes)and y (impair indexes) for each RC

    boolean_model<T>(Random_centers, Random_radius, RBound, myParamSpeckle, myParamAlgo, myParamSensor, number, fun, seed);

    // monte_carlo_estimation<T>(speckle_matrix, Random_centers, Random_radius, RBound, myParamSpeckle, myParamAlgo, myParamSensor, number, fun, seed);
    monte_carlo_estimation_cuda<T>(speckle_matrix, Random_centers, Random_radius, RBound, number, seed, width, height, alpha, nbit, gamma, N0);

    //quantization ==> output in range [0, 2^nbit-1]
    T *qt_out = NULL;
    qt_out = (T *)malloc(width * height * sizeof(T));
    quantization<T>(qt_out, speckle_matrix, width * height, nbit);

    //create output float image
    float *imgOut = NULL;
    imgOut = (float *)malloc(width * height * 1 * sizeof(float)); // number of channels is 3 <=> RGB

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; j++)
        {
            imgOut[j + i * width] = qt_out[j + i * width]; //red
        }
    }
    write_output_image(imgOut, fileNameOut, myParamSpeckle, myParamAlgo, myParamSensor);

    // free the allocated memory
    free(imgOut);
    free(qt_out);
    free(speckle_matrix);
    free(Random_radius);
    free(RBound);
    free(Random_centers);

    // display elapsed time
    gettimeofday(&end, NULL);
    double elapsedTime = (end.tv_sec - start.tv_sec) +
                         (end.tv_usec - start.tv_usec) / 1.e6;
    std::cout << "time elapsed : " << elapsedTime << std::endl;
    std::cout << "***************************" << std::endl;

    return (0);
}

/**
* @brief main function call
*/
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        show_help();
        return -1;
    }

    if (cmdOptionExists(argv, argv + argc, "-prec"))
    {
        int prec = atoi(getCmdOption(argv, argv + argc, "-prec"));
        if (prec == 1)
            generator_main<double>(argc, argv, "double");
        else
            generator_main<float>(argc, argv, "float");
    }
    else
        generator_main<float>(argc, argv, "float");
}