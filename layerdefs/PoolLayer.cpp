#include <vector>
#include <iostream>
#include <algorithm>

#include "PoolLayer.h"
#include "../utility/matops.h"

// class constructor
PoolLayer::PoolLayer(ivec inshape, ConvNet* net) :
    Fshape({2, 2}),
    InShape(inshape),
    Excitations(inshape[0], imatrix2(inshape[1], ivec(inshape[2])))
{   
    Stride   = 2;
    
    OutShape = outshape(InShape, Fshape, Stride, 0);
    Steps    = calcsteps(InShape, Fshape, Stride, 0);
    
    dmatrix3(OutShape[0], dmatrix2
            (OutShape[1], dvec
            (OutShape[2]))).swap(Activations);
    dmatrix3(OutShape[0], dmatrix2
            (OutShape[1], dvec
            (OutShape[2]))).swap(Errors);
    Brain = net;
}

// Sets attributes to zero before a new phase of learning
inline void PoolLayer::reset()
{
    imatrix3 nullMatrix(InShape[0], imatrix2(InShape[1], ivec(InShape[2])));
    nullMatrix.swap(Excitations);
}

// Feedforward
dmatrix3 PoolLayer::think(dmatrix3 &mat)
{
    
    ThoughtBubble bubble;
    int index=0;
    dvec activationVec(OutShape[0]*OutShape[1]*OutShape[2]);
    ivec step(Fshape[0]*Fshape[1]);
    
    reset();
    
    for(int z=0;z<mat.size();z++) {
        for(int i=0;i<Steps.size();i++, index++) {
            imatrix2 exc(Fshape[0], ivec(Fshape[1]));

            step.assign(Steps[i].begin(), Steps[i].end());            
            bubble = max_pool(slice<real>(mat[z], step));            
            activationVec[index] = bubble.Activation;
            exc = fold2<int>(bubble.Excitation, Fshape);
            apply_exc(exc, step, z);
        }
    }
    Activations = fold3<real>(activationVec, OutShape);
    return Activations;
}

void PoolLayer::apply_exc(imatrix2 &exc, ivec step, int z)
{
    int i=0;
    for(int ey=step[2];ey<step[3];ey++, i++) {
        int j=0;
        for(int ex=step[0];ex<step[1];ex++, j++) {
            Excitations[z][ey][ex] += exc[i][j];
        }
    }
}

ThoughtBubble PoolLayer::max_pool(dmatrix2 slice) const
{
    dvec fslice(4);
    ivec bslice(4, 0);
    double maxi;
    ThoughtBubble output;
    
    fslice = flatten<real>(slice);
    maxi = *std::max_element(fslice.begin(), fslice.end());
    bslice[*std::find(fslice.begin(), fslice.end(), maxi)] = 1;
    
    output.Excitation = bslice;
    output.Activation = maxi;
    return output;
}

dmatrix3 PoolLayer::backpropagation() const
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
                        outputs[z][i][j] +=
                                Excitations[z][i][j] *
                                Errors[z][y][x];
                    }
                }
            }
        }
    }
    return outputs;
}