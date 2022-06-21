

template <typename T>
struct vec2D {
    T x;
    T y;
};


template <typename T>
struct paramAlgo {
    int N0;
    T alpha;
    typedef T dataType; // this means that dataType and T are the same type
    int NMCmax;
};

template <typename T>
struct paramSensor {
    int nbit;
    T sigma;
    vec2D<int> dims;
};

template <typename T>
struct paramSpeckle {
    char distribR;
    T mu;
    T lambda;
    T gamma;
    T sigmaR=0;
};
