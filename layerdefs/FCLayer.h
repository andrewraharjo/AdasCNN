/*Please refer to the LICENSE.txt file in this project's root
  for copyright informations!*/
#ifndef FCLAYER_H
#define FCLAYER_H

#include <vector>

#include "../utility/generic.h"

class ConvNet; //Forward declare this baby

class FCLayer
{
public:
    FCLayer(ivec inshape, int neurons, ConvNet*);
    dvec think(dvec&);
    dvec think(dmatrix2&);
    dvec think(dmatrix3&);
    
    dvec backpropagation();
    void weight_update(dvec);
    
    void reset();

    dvec     Activations;
    dvec     Excitations;
    dvec     Errors;
    ivec     OutShape;
    
    dmatrix2 Weights;
    dvec     Biases;
    
    ConvNet* Brain;
};

#endif /* FCLAYER_H */

