#include <cmath>
#define my_pi 3.14159265358979323846f


/*
modsinewave2 mapping function
*/
template <typename T>
void ripple(T x_in, T y_in, T *x_out, T *y_out);

/*
translation01 mapping function
*/
template <typename T>
void translation01(T x_in, T y_in, T *x_out, T *y_out);


/*
modsinewave2 mapping function
*/
template <typename T>
void modsinewave2(T x_in, T y_in, T *x_out, T *y_out);


/*
identity function (mapping function)
*/
template <typename T>
void identity(T x_in, T y_in, T *x_out, T *y_out);

template <typename T>
void sinewave(T x_in, T y_in, T *x_out, T *y_out);