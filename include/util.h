#include <iostream>
#include <string>
#include <unistd.h>
#include <stdlib.h>
#include <algorithm>
#include <random>
#include <cstring>

#include "paramalgo.h"

void show_help();

char *getCmdOption(char **begin, char **end, const std::string &option);

bool cmdOptionExists(char **begin, char **end, const std::string &option);

std::string getFileExt(const std::string &s);

std::string getFileName(const std::string &s);

std::string get_curr_dir();

int write_output_image(float *imgOut, const std::string fileNameOut,
                       paramSpeckle<float> myParamSpeckle, paramAlgo<float> myParamAlgo, paramSensor<float> myParamSensor);


