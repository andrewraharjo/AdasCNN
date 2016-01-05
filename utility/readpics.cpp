#include <iostream>
#include <fstream>
#include <assert.h>
#include <string>
#include <sstream>
#include <chrono>

#include "readpics.h"
#include "generic.h"
#include "matops.h"

#define RELATIVE_PATH_OF_TRAINING_DIRECTORY "train-52x52/"
#define RELATIVE_PATH_OF_TESTING_DIRECTORY "test-52x52/"

using namespace std;

//MinGW's to_sting() is buggy :(
inline string convertInt(int i)
{
    stringstream ss;
    ss << i;
    return ss.str();
}

inline string zeropad(string s, int len)
{
    string infix = "";
    for(int x=0;x<len-s.size();x++)
        infix += "0";
    return infix + s;
}

dmatrix3 readfile(string filename)
{
    const char *cstr = filename.c_str();
    FILE* f = fopen(cstr, "rb");
        
    unsigned char info[54]; // the header is 54 bytes long!
    fread(info, sizeof(unsigned char), 54, f);

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];
 
    // just to be sure you guys are not triing anything funny...
    //cout << filename << " H: " << height << " W: " << width << endl;
    
    dmatrix3 output(3, dmatrix2(52, dvec(52)));

    // This is only needed in 24 bit color BMP files.
    int row_padded = (width*3 + 3) & (~3);
    unsigned char* data = new unsigned char[row_padded];

    int x=0;
    for(int i=0; i<height; i++, x++)
    {
        int y=0;
        fread(data, sizeof(unsigned char), row_padded, f);
        for(int j=0; j<width*3; j+=3, y++)
        {
            output[0][y][x] = (double)(int)data[j];   // B
            output[1][y][x] = (double)(int)data[j+1]; // G
            output[2][y][x] = (double)(int)data[j+2]; // R
        }
    }
    fclose(f);
    return feature_scale(output);
}

UnlabeledData::UnlabeledData(int n) :
        Data(n, dmatrix3(3, dmatrix2(52, dvec(52)))),
        Predictions(n)
{
    string prefix = RELATIVE_PATH_OF_TESTING_DIRECTORY;
    string suffix = ".bmp";
    for(int x=1;x<=n;x++) {
        string infix = convertInt(x);
        Data[x-1] = readfile(prefix + zeropad(infix, 8) + suffix);
    }
}

void UnlabeledData::dump()
{
    ofstream log ("log.txt");
    
    for(int x=0;x<Data.size();x++) {
        if (log.is_open()) {
            log << "Pic " << x << " -> " << Predictions[x] << endl;
        }
        else cout << "Can't open log file...";
    }
    cout << "Made the predictions! Logfile dumped to log.txt in cwd." << endl;
    log.close();
}

LabeledData::LabeledData(int n, int val) :
        Training(n*12-val, dmatrix3(3, dmatrix2(52, dvec(52)))),
        TLabels(n*12-val, dvec(12)),
        Validation(val, dmatrix3(3, dmatrix2(52, dvec(52)))),
        VLabels(val, dvec(12))
{
    dmatrix4 data(n*12, dmatrix3(3, dmatrix2(52, dvec(52))));
    dmatrix2 labels(n*12, dvec(12));
    
    parsedirs(data, labels, n); //reads bmp data into the matrices above
    //Two independently stored vectors (data and labels) needs to be randomized
    //together, so a helper array is used, which contains the shuffled indices
    //of the two vectors.
    setmembers(data, labels, n, val);
}

void LabeledData::setmembers(dmatrix4 &data, dmatrix2 &labels, int n, int val)
{
    int from_chaos_came_randomness[n*12];
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    
    for(int x=0;x<n*12;x++)
        from_chaos_came_randomness[x] = x;
    
    shuffle(from_chaos_came_randomness,
            from_chaos_came_randomness+data.size(),
            std::default_random_engine(seed));
    
    for(int x=0;x<n*12;x++) {
        if(x<val) {
            Validation[x]  = data[from_chaos_came_randomness[x]];
            VLabels[x]     = labels[from_chaos_came_randomness[x]];
        }
        else {
            Training[x-val] = data[from_chaos_came_randomness[x]];
            TLabels[x-val]  = labels[from_chaos_came_randomness[x]];
        }
    }
}

void LabeledData::parsedirs(dmatrix4 &data, dmatrix2& labels, int n)
{
    string   dirname;
    string   srepr;
    string   postfix = ".bmp";
    string   prefix;
    string   infix;
    int      srpl;

    
    for(int dirnum=1;dirnum<=12;dirnum++) {
        //Set the 12-vector representing the target of the outputs
        dvec vlabel(12, 0.0);
        vlabel[dirnum-1] = 1.0;
        
        //Building the first part of the path string
        prefix = RELATIVE_PATH_OF_TRAINING_DIRECTORY;
        dirname = convertInt(dirnum);
        prefix += dirname + "/" + dirname + "_";
        
        for(int i=0;i<n;i++) {
            //Building the middle part of the path string
            infix = "";
            srepr = convertInt(i);
            
            //Padding with zeros
            srpl = srepr.size();
            for(int x=0;x<4-srpl;x++) infix += "0";

            infix += srepr; // Add the file number after the padding
            
            //Read the pixel data into this 4D matrix
            data[((dirnum-1)*n)+i] = readfile(prefix + infix + postfix);
            //Store the labels in this 2D matrix
            labels[((dirnum-1)*n)+i] = vlabel;
        }
    }
}