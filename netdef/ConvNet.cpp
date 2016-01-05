#include <iostream>
#include <vector>
#include <algorithm>

#include "ConvNet.h"
#include "../utility/matops.h"
#include "../utility/activations.h"

ConvNet::ConvNet(int filters, ivec inshape, ivec convfilter,
        int stride, int hidNeurons, int outNeurons, double eta) :
    L1(filters, inshape, convfilter, stride, this),
    L2(L1.OutShape, this),
    L3(L2.OutShape, hidNeurons, this),
    L4(L3.OutShape, outNeurons, this)
{
    std::cout << "Created ConvNet with architecture:" << std::endl;
    std::cout << "ConvLayer fltshape: ";
    printdims<real>(L1.Filters);
    std::cout << "ConvLayer outshape: ";
    printm<int>(L1.OutShape);
    std::cout << "PoolLayer outshape: ";
    printm<int>(L2.OutShape);
    std::cout << "FCLayer1  weights : ";
    printdims<real>(L3.Weights);
    std::cout << "FCLayer1  outshape: ";
    printm<int>(L3.OutShape);
    std::cout << "FCLayer2  weights : ";
    printdims<real>(L4.Weights);
    std::cout << "FCLayer2  outshape: ";
    printm<int>(L4.OutShape);
    Eta = eta;
    std::cout << std::endl;
}

dvec ConvNet::feedforward(dmatrix3 &stimulus)
{
    Inputs = &stimulus;

    dmatrix3 stage1;// = L1.think(stimulus);
    dmatrix3 stage2;// = L2.think(stage1);
    dvec stage3;// = L3.think(stage2);
    dvec result;// = L4.think(stage3);
    
    stage1 = L1.think(stimulus);
    stage2 = L2.think(stage1);
    stage3 = L3.think(stage2);
    result = L4.think(stage3);
        
    return result;
}

void ConvNet::learn(dmatrix3 &stimulus, dvec &target)
{
    dvec result(target.size());
    
    result = feedforward(stimulus);
    real er = 0;
    for(int x=0;x<target.size();x++) {
        L4.Errors[x] = sigmoid_p(target[x]) * (target[x] - result[x]);
        er += L4.Errors[x];
    }
    
    std::cout << "Output error: " << er << std::endl;
    
    L3.Errors = L4.backpropagation();
    dvec l2er = L3.backpropagation();
    L2.Errors = fold3(l2er, L2.OutShape);
    L1.Errors = L2.backpropagation();
    
    L4.weight_update(L3.Activations);
    L3.weight_update(flatten(L2.Activations));
    //L2 is a PoolLayer, those doesn't have any weights.
    L1.weight_update(Inputs); // Warning: Input is a pointer!!!
}

void ConvNet::learn(dmatrix4 &stimulus, dmatrix2 &target, int lessons)
{
    assert(stimulus.size()==target.size());
    
    if(lessons==0) lessons=stimulus.size();
    
    for(int n=0;n<lessons;n++) {
        if (n%10 == 0) std::cout << "Learning lesson #" << n+1 << std::endl;
        learn(stimulus[n], target[n]);
    }
}

real ConvNet::validation(dmatrix4 &lessons, dmatrix2 &targets)
{
    assert(lessons.size()==targets.size());
    
    double rounds = (double)lessons.size();
    int result; // Holds the networks' prediction of the category
    int answer; // Holds the actual number of the right category
    int right = 0; // Holds the number of right answers
        
    for(int n=0;n<rounds;n++) {
        result = predict(lessons[n]);
        answer = *std::find(targets[n].begin(),
                           targets[n].end(), 1.0); //index of 1.0 in targets
        if (result==answer) right++;
    }
    return ((double)right/(double)lessons.size());
}

int ConvNet::predict(dmatrix3 &mat)
{
    dvec thought(12);
    real maxi;
    int result;
    
    thought = feedforward(mat);
    
    //We determine the index of the maximum element in the output vector.
    maxi = *std::max_element(thought.begin(), thought.end());
    result= *std::find(thought.begin(), thought.end(), maxi);
    
    return result;
}