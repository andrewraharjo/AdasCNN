/*Please refer to the LICENSE.txt file in this project's root
  for copyright informations!*/
#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <vector>

#include "../utility/generic.h"

class ConvNet;

class ConvLayer
{
public:
    
    dmatrix3 Excitations;
    dmatrix3 Activations;
    dmatrix3 Errors;
    ivec InShape;
    ivec OutShape;    
    dmatrix4 Filters; // 4D! The dream!!!
    
    ConvLayer(int filters, ivec inshape, ivec fshape, int stride, ConvNet*);
    
    dmatrix3 think(dmatrix3 mat);
    dmatrix3 backpropagation() const;
    void weight_update(dmatrix3* inputs);

    dmatrix3* Inputs;    
    ivec Fshape;
    imatrix2 Steps;
    int Stride;

    ConvNet* Brain;
};



#endif /* CONVLAYER_H */

