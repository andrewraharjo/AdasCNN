#include <vector>
#include <numeric>
#include <assert.h>

#include "FCLayer.h"
#include "../netdef/ConvNet.h"
#include "../utility/matops.h"
#include "../utility/activations.h"

FCLayer::FCLayer(ivec inshape, int neurons, ConvNet* net) :
        OutShape(1, neurons),
        Biases(neurons, 0.5),
        Errors(neurons),
        Excitations(neurons),
        Activations(neurons)
{
    int inputs=1;
    for(int i=0;i<inshape.size();i++) {
        inputs *= inshape[i];
    }

    dmatrix2(neurons, dvec(inputs, 0.5)).swap(Weights);
    Brain = net;
}

dvec FCLayer::think(dvec &stimuli)
{
    assert(stimuli.size() == Weights[0].size());
    
    for(int y=0;y<Weights.size();y++) {
        Excitations[y] = dot<real>(Weights[y], stimuli) + Biases[y];
        Activations[y] = sigmoid(Excitations[y]);
    }
    return Activations;
}

dvec FCLayer::think(dmatrix2 &stimuli)
{
    dvec vstim = flatten(stimuli);
    return think(vstim);
}

dvec FCLayer::think(dmatrix3 &stimuli)
{
    dvec vstim = flatten(stimuli);
    return think(vstim);
}

void FCLayer::reset(void)
{
    for(int y; y<Weights.size(); y++) {
        for(int x; x<Weights[0].size(); x++) {
            Weights[y][x] = 0.5;
        }
    }
}

dvec FCLayer::backpropagation()
{
    //Calculating the errors of the previous layer!
    dmatrix2 weightT = transpose<real>(Weights);
    dvec output(weightT.size());

    for(int x=0;x<weightT.size();x++) {
        output[x] = (sigmoid_p(Excitations[x]) *
                    (dot<real>(weightT[x], Errors)));
    }
    return output;
    printm(output);
}

void FCLayer::weight_update(dvec prevOutput)
{
    assert(prevOutput.size()==Weights[0].size());
    for(int i=0;i<Weights.size();i++){
        for(int j=0;j<Weights[0].size();j++) {
            // I might be wrong about this! I only did this in 2D.
            Weights[i][j] += Brain->Eta * Errors[i] * prevOutput[j];
            Biases[i] += Brain->Eta * Errors[i];
            // This means that the matrix of deltas is actually the
            // Outer product of the Error and previous output vectors.
        }
    }
}
