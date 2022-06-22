#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip> 
#include <vector>
#include <string>
#include <unistd.h>
#include <stdlib.h>
#include <algorithm>
#include <cstring>

#include "data_structures.h"

void show_help();

char *getCmdOption(char **begin, char **end, const std::string &option);

bool cmdOptionExists(char **begin, char **end, const std::string &option);

std::string getFileExt(const std::string &s);

std::string getFileName(const std::string &s);

std::string get_curr_dir();

int write_output_image(float *imgOut, const std::string fileNameOut,
                       paramSpeckle<float> myParamSpeckle, paramAlgo<float> myParamAlgo, paramSensor<float> myParamSensor);

template <typename T>
std::vector<T> read_csv_matrix(std::string filename, int *size_x, int *size_y);
