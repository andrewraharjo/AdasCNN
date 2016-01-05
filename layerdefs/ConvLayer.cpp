#include <vector>
#include <assert.h>
#include <iostream>

#include "ConvLayer.h"
#include "../utility/matops.h"
#include "../utility/activations.h"

ConvLayer::ConvLayer(int filters, ivec inshape, ivec fshape, int stride,
                     ConvNet* net)
{
    InShape  = inshape;
    Stride   = stride;
    Fshape   = fshape;
    OutShape = outshape(InShape, Fshape, Stride, filters);
    Steps    = calcsteps(InShape, Fshape, Stride, filters);
            
    dmatrix3 refE(OutShape[0], dmatrix2(OutShape[1], dvec(OutShape[2], 0.0)));
    refE.swap(Excitations);
    dmatrix3 refA(OutShape[0], dmatrix2(OutShape[1], dvec(OutShape[2], 0.0)));
    refA.swap(Activations);
    dmatrix3 refErr(OutShape[0],dmatrix2(OutShape[1],dvec(OutShape[2], 0.0)));
    refErr.swap(Errors);
    dmatrix4 flt(filters,dmatrix3(InShape[0],
                 dmatrix2(Fshape[0],dvec(Fshape[1], 0.5))));
    flt.swap(Filters);
    
    Brain = net;
}

dmatrix3 ConvLayer::think(dmatrix3 mat)
{
    dmatrix3 slab(mat.size(), dmatrix2(Fshape[1], dvec(Fshape[0])));
    ivec step(4);
    dvec exc(OutShape[1]*OutShape[2]);
    dvec act(OutShape[1]*OutShape[2]);
       
    ivec foldshape(2);
    foldshape[0] = OutShape[1];
    foldshape[1] = OutShape[2];
    
    Inputs = &mat;
    
    for(int f=0;f<Filters.size();f++) {
        dmatrix3 filt = Filters[f];
        for(int i=0;i<Steps.size();i++) {
            step = Steps[i];
            slab = invert<real>(slice<real>(invert<real>(mat), step));
            exc[i] = frobenius(slab, filt); // This is the "convolve" step
            act[i] = sigmoid(exc[exc.size()-1]);
        }
        Excitations[f] = fold2<real>(exc, foldshape);
        Activations[f] = fold2<real>(act, foldshape);
    }
    return Activations;
}

dmatrix3 ConvLayer::backpropagation() const
{
    dmatrix3 outputs(Excitations.size(), dmatrix2
                    (Excitations[0].size(), dvec
                    (Excitations[0][0].size(), 0.0)));
    ivec step;
    step.reserve(4);
    
    int index;
    
    for(int z=0;z<Errors.size();z++) {
        index = 0;
        for(int y=0;y<Errors[0].size();y++) {
            for(int x=0;x<Errors[0][0].size();x++, index++) {
                step = Steps[index];
                for(int i=step[0];i<step[1];i++) {
                    for(int j=step[2];j<step[3];j++) {
                        outputs[z][i][j] += sigmoid_p(
                                Excitations[z][i][j] *
                                Errors[z][y][x]);
                    }
                }
            }
        }
    }
    return outputs;
}

void ConvLayer::weight_update(dmatrix3* inputs)
{
    assert(Errors.size()==Filters.size());
    
    std::size_t sh[4] = {Filters.size(),Filters[0].size(),
                 Filters[0][0].size(),Filters[0][0][0].size()};
    ivec step(4);
    dmatrix2 sheet;
    
    for(int f=0;f<sh[0];f++) {
        for(int z=0;z<sh[1];z++) {
            for(int y=0;y<sh[2];y++) {
                for(int x=0;x<sh[3];x++) {
                    step[0] = x; step[1] = x + OutShape[2];
                    step[2] = y; step[3] = y + OutShape[1];
                    
                    sheet = slice<real>(inputs->at(z), step);
                    
                    Filters[f][z][y][x] += frobenius(Errors[f], sheet);
                }
            }
        }
    }
}