#include "mapping_functions.h"


/*
modsinewave2 mapping function
*/
template <typename T>
void ripple(T x_in, T y_in, T *x_out, T *y_out)
{
    // Ripple effect function
    // TODO : make this function independant from image size

    T tmp_x = x_in;
    T tmp_y = y_in;

    unsigned int n = 500; // this equal to width or height ==> for image size = 100 X 100

    unsigned int nb_rings = 6; // nb of rings to output 
    T periode = (T)n/(2*nb_rings);
    T step = 2*my_pi/periode;
    tmp_x = (tmp_x-1)*step;
    tmp_y = (tmp_y-1)*step;

    T p_x = (T)(n-2)/2*step;
    T p_y = (T)(n-2)/2*step;

    tmp_x = tmp_x - p_x;
    tmp_y = tmp_y - p_y;
    T d = tmp_x*tmp_x + tmp_y*tmp_y;
    T amp = 0.5;


    *x_out = x_in + amp*cos(sqrt(d)); 
    *y_out = y_in;

}

/*
translation01 mapping function
*/
template <typename T>
void translation01(T x_in, T y_in, T *x_out, T *y_out)
{
    // translation function (translation01)
    T t = 0.5;
    *x_out = x_in + t;
    *y_out = y_in;
}

/*
modsinewave2 mapping function
*/
template <typename T>
void modsinewave2(T x_in, T y_in, T *x_out, T *y_out)
{
    // sinwave function (modsinewave2)
    T A = 0.5;
    T decalage_origine = 250;
    T periode_mini = 4;
    T periode_maxi = 32;
    T maxy = 500;
    T periode = y_in * (periode_maxi - periode_mini) / maxy + periode_mini;
    *x_out = x_in + A * (T)cos(2 * my_pi / periode * (x_in - decalage_origine));
    *y_out = y_in;
}

template <typename T>
void sinewave(T x_in, T y_in, T *x_out, T *y_out)
{
    // sinwave function (modsinewave2)
    T A = 0.5;
    T decalage_origine = 50;
    T periode = 8;
    *x_out = x_in + A * (T)cos(2 * my_pi / periode * (x_in - decalage_origine));
    *y_out = y_in;
}

/*
identity function (mapping function)
*/
template <typename T>
void identity(T x_in, T y_in, T *x_out, T *y_out)
{
    *x_out = x_in;
    *y_out = y_in;
}

// template Instantiations
template void ripple<float>(float, float, float*, float*);
template void ripple<double>(double, double, double*, double*);
template void translation01<float>(float, float, float*, float*);
template void translation01<double>(double, double, double*, double*);
template void modsinewave2<float>(float, float, float*, float*);
template void modsinewave2<double>(double, double, double*, double*);
template void sinewave<float>(float, float, float*, float*);
template void sinewave<double>(double, double, double*, double*);
template void identity<float>(float, float, float*, float*);
template void identity<double>(double, double, double*, double*);