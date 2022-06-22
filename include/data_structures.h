#ifndef    PARAMALGO_H
#define    PARAMALGO_H




template <typename T>
struct vec2D {
    T x;
    T y;
};

////////// parameters structures for user's inputs ///////////////

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


///////// structures for MC estimation /////////
struct Random_disk {
    float x;
    float y;
    float r;
};

struct Boolean_model_disk {
    Random_disk disk;
    float rbound;
};



#endif