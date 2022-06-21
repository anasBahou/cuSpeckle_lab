
// headers of the utils 

#include<string>
#include<math.h>
#include <stdbool.h>

#include </home/lotfi/cuProject/cuSpeckle/include/paramalgo.h>


void quantization(float  *img_out,
                  float  *speckle_matrix,
                  int img_size, int nbit);


static void show_help();


char *getCmdOption(char **begin, char **end, const std::string &option);

bool cmdOptionExists(char **begin, char **end, const std::string &option);

std::string getFileExt(const std::string &s);

std::string getFileName(const std::string &s);

std::string get_curr_dir();

template <typename T>
int write_output_image(float *imgOut, const std::string fileNameOut,
                       paramSpeckle<T> myParamSpeckle,
                       paramAlgo<T> myParamAlgo,
                       paramSensor<T> myParamSensor)



