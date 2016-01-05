/*Please refer to the LICENSE.txt file in this project's root
  for copyright informations!*/
#ifndef CONVNET_H
#define CONVNET_H

#include <vector>

#include "../utility/generic.h"
#include "../layerdefs/ConvLayer.h"
#include "../layerdefs/PoolLayer.h"
#include "../layerdefs/FCLayer.h"

class ConvNet
{
public:
    real Eta;
    ConvNet(int filters, ivec inshape, ivec convfilter, int stride,
            int hidNeurons, int outNeurons, double eta);
    dvec feedforward(dmatrix3 &stimulus);
    void learn(dmatrix3 &stimulus, dvec &target);
    void learn(dmatrix4 &stimulus, dmatrix2 &target, int lessons=0);
    real validation(dmatrix4 &lessons, dmatrix2 &targets);
    int predict(dmatrix3&);
    
private:
    ConvLayer   L1;
    PoolLayer   L2;
    FCLayer     L3;
    FCLayer     L4;
    dmatrix3*   Inputs;
};

#endif /* CONVNET_H */

