#include <iostream>
#include <random>
#include <cmath>
#include <functional>
#include <ctime>
#include "paramalgo.h"
#include <boost/math/special_functions/erf.hpp>
#include "mapping_functions.h"

const float pi = 3.14159265358979323846f;

template <typename T>
void generate_random_radius(T output[], T *kappa, T *theta, paramSpeckle<T> myParamSpeckle, vec2D<int> dims, int number, unsigned int seed);

template <typename T, typename F>
T estimate_delta(vec2D<int> dims, F fun);

template <typename T>
void boolean_model(T *Random_radius, T *RBound, paramSpeckle<T> myParamSpeckle, paramAlgo<T> myParamAlgo, paramSensor<T> myParamSensor, int number, std::function<void(T, T, T *, T *)> fun, unsigned int seed);

template <typename T>
void monte_carlo_estimation_cuda(T *speckle_matrix, T *Random_centers, T *Random_radius, T *RBound, int number, unsigned int seed, int width, int height, T alpha, int nbit, T gamma, int N0);

template <typename T>
void quantization(T *img_out, T *speckle_matrix, int img_size, int nbit);