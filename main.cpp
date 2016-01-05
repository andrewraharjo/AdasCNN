/* Convolutional Neural Network demonstration
 * Copyright (c) 2015 Csaba Gor <csxeba@gmail.com> 
 * 

 This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
  
 
 * Artificial Neural Network implementation for roadsign classification.
 * Various layer definitions can be found in the layerdefs directory.
 * Concrete network definition is implemented as a class with layer objects
 * as members. I didn't create it to be flexible.
 * Since the use of 3db party libs was not allowed, I wrote my own matrix
 * operations, which can be found in the utility/matops files. They are not
 * fast, I didn't paralellize them.
 * I tried to be as elegant as possible, but I'm no C++ programmer, and most
 * of my time and energy was spent learning the tits and tats of the language.
 * This was a fun project though. Real challange.
 * 3-4 hours of learning ahead with this one...
 */

#include <iostream>
#include <fstream>
#include <vector>

#include "utility/generic.h"
#include "utility/matops.h"
#include "utility/readpics.h"
#include "netdef/ConvNet.h"

using namespace std;

void introduction()
{
    cout << "Image classification by a Convolutional Neural Network" << endl <<
            "   Copyright (c) 2016 Csaba Gor <csabagor@gmail.com>" << endl <<
            "   This is a free software licensed under the GNU GPL 3.0 license." <<
            endl << "   You should have received a copy of the license along" <<
            endl << "   with the source code." << endl << endl <<
            "Now let the fun begin..." << endl << endl;
    
    cout << "I assume the compiled and linked executable is located at the" <<
            endl << "root of the directory, which holds the unzipped training" <<
            endl << "and testing datasets, which I further assume are" <<
            endl << "named exactly the same as they were named in the original" <<
            endl << "zip files :)" << endl << endl;
}
    
void training_round(ConvNet &cn, LabeledData &LData)
{
    // Teach the pictures to the network
    cout << "Training..." << endl;
    cn.learn(LData.Training, LData.TLabels);
    cout << "Training done!" << endl;
    
    // Validate against these "unseen" pictures
    cout << "Validation..." << endl;
    double right = cn.validation(LData.Validation, LData.VLabels);
    cout << "Right answers: " << right * 100 << "%" << endl;
}

void testing_round(ConvNet &cn, UnlabeledData &ULData)
{
    int choice;
    
    cout << endl << "This is an interactive testing mode." << endl <<
            "Just type in the number corresponding to an unlabeled" << endl <<
            "picture in the testing dataset and the" << endl <<
            "network will make a prediction." << endl <<
            "There are " << ULData.Data.size() << " testing pictures" << 
            " to chose from." << endl;
    
    cout << "Press 0 to quit testing mode!";
    do {
        cout << endl << "Testing > ";
        cin >> choice;
        
        if (choice) {
        int prediction = cn.predict(ULData.Data[choice-1]);
        ULData.Predictions[choice-1] = prediction;
        cout << endl << "Picture #" << choice << " is of category #" << 
                prediction+1 << endl;
        }
        else return; //Freezes for some weird reason O.o
    }
    while (choice);
}

void predict_all(ConvNet &cn, UnlabeledData &ULData)
{
    //Predicting every test image and dumping the predictions to a log file
    for(int x=0;x<ULData.Data.size();x++) {
        ULData.Predictions[x] = cn.predict(ULData.Data[x]);
        if ((x+1)%10 == 0) cout << "Done " << x << "!" << endl;
    }
    ULData.dump();
}

int main()
{
    //Instanciate the classes holding the data    
    int training_pics = 5;
    int testing_pics = 20;
    int validation_pics = 10;
    cout << "Please specify the number of training pics to be" << endl <<
            "read into arrays per directory ( < 5000)" << endl <<
            "> ";
    cin >> training_pics;
    
    cout << endl;
    
    cout << "Please specify the number of testing pics to be" << endl <<
            "read into arrays ( < 6000)" << endl <<
            "> ";
    cin >> testing_pics;
    
    cout << "Please specify the number of validation pics to be" << endl <<
            "used to validate the network ( < " <<training_pics*12 << ")" << endl
            << "> ";
    cin >> validation_pics;
    
    LabeledData LData(training_pics, validation_pics);
    UnlabeledData ULData(testing_pics);
    cout << endl;
    
    //Definition of some CNN hyperparameters
    int filters = 5; //the number of filters
    int stride = 1; //zero-padding not implemented, so carefully change stride!
    int hidNeurons = 100; // The fully connected hidden layer's neurons
    int outNeurons = LData.TLabels[0].size();    
    ivec inshape(shapeof(LData.Training[0]));
    ivec convfilter({23, 23}); //convolutional filter size
    double eta = 1.5; //this is the learning rate
    
    // Instanciate the Network
    ConvNet cn(filters, inshape, convfilter, stride,
               hidNeurons, outNeurons, eta);    
        
    training_round(cn, LData);
    testing_round(cn, ULData);
    predict_all(cn, ULData);
    
    return 0;
}