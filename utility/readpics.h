/*Please refer to the LICENSE.txt file in this project's root
  for copyright informations!*/
#ifndef READPICS_H
#define READPICS_H

#include <string>

#include "generic.h"


//Some utility functions for string parsing, file reading, etc...
inline std::string convertInt(int i);
inline std::string zeropad(std::string s, int len);
dmatrix3 readfile(std::string filename);

//Behold, the holder of unlabeled data!
//Predictions can be dumped to file.
class UnlabeledData
{
public:
    dmatrix4 Data;
    ivec Predictions;
    
    UnlabeledData(int);
    void dump();
};

class LabeledData
{
public:
    dmatrix4 Training;
    dmatrix2 TLabels;
    dmatrix4 Validation;
    dmatrix2 VLabels;
    dmatrix3 Pic;

    LabeledData(int, int);
    
private:
    void parsedirs(dmatrix4&, dmatrix2&, int n);
    void setmembers(dmatrix4&, dmatrix2&, int n, int val);

};


#endif /* READPICS_H */

