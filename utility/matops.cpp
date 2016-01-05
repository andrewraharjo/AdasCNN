#include <iostream>
#include <vector>
#include <assert.h>

#include "matops.h"

ivec outshape(ivec inshape, ivec filtersize, int stride, int filters)
{
    ivec out(3);
    int x;
    
    out[0] = (filters == 0) ? inshape[0] : filters;
    
    for(int i=2;i>0;i--)
    {
        x = inshape[i]-filtersize[i-1];
        assert(x % stride==0);
        out[i] = (x / stride) + 1;
    }
   
    return out;
}

imatrix2 calcsteps(ivec inshape, ivec filtersize, int stride, int filters)
{
    ivec oshape;
    ivec startxes;
    ivec startys;
    ivec endxes;
    ivec endys;
    imatrix2 steps;
    
    oshape = outshape(inshape, filtersize, stride, filters);
    
    for(int i=0;i<oshape[2];i++) {
        startxes.push_back(i*stride);
        endxes.push_back(startxes[i]+filtersize[0]);
    }
    for(int i=0;i<oshape[1];i++) {
        startys.push_back(i*stride);
        endys.push_back(startys[i]+filtersize[0]);
    }
    for(int i=0;i<startxes.size();i++) {
        for(int j=0;j<startys.size();j++) {
            int s[4] = {startxes[i], endxes[i], startys[j], endys[j]};
            std::vector<int> step(s, s+4);
            steps.push_back(step);
        }
    }
    return steps;
}