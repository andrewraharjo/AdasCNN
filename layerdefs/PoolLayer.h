/*Please refer to the LICENSE.txt file in this project's root
  for copyright informations!*/
#ifndef LAYERS_H
#define LAYERS_H

#include <vector>

#include "../utility/generic.h"

class ConvNet; //Forward declare this baby

struct ThoughtBubble
{
    ivec    Excitation;
    double  Activation;
};

class PoolLayer
{
public:
    imatrix3 Excitations;
    dmatrix3 Activations;
    dmatrix3 Errors;
    ivec     OutShape;
    ivec     InShape;
    
    PoolLayer(ivec, ConvNet*);
    dmatrix3 think(dmatrix3&);
    dmatrix3 backpropagation() const;
    
    int      Stride;
    ivec     Fshape; //filter shape
    imatrix2 Steps;
    
    ThoughtBubble max_pool(dmatrix2) const;
    void reset();
    void apply_exc(imatrix2&, ivec, int);
    
    ConvNet* Brain;
};

#endif	// LAYERS_H