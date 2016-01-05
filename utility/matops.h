/*Please refer to the LICENSE.txt file in this project's root
  for copyright informations!*/
#ifndef MATOPS_H
#define MATOPS_H

//Heard this wasn't very ethical.
#define Tvec std::vector<T>
#define Tmatrix2 std::vector<std::vector<T> >
#define Tmatrix3 std::vector<std::vector<std::vector<T> > >
#define Tmatrix4 std::vector<std::vector<std::vector<std::vector<T> > > >
//Well, I'm lazy.

#include <assert.h>
#include <numeric>
#include <iostream>
#include <algorithm>

#include "generic.h"

  //-----------------------  //
 // Definition of functions //
//  -----------------------//

//Calculation of step coordinates
//This is defined in the .cpp
imatrix2 calcsteps(ivec inshape, ivec filtersize, int stride, int filters);
ivec outshape(ivec inshape, ivec filtersize, int stride, int filters);

//Used by many, must be at the top!
template <typename T>
inline bool dimsmatch(Tvec &v1, Tvec &v2) {
    return v1.size()==v2.size();
}

template <typename T>
inline bool dimsmatch(Tmatrix2 &m1, Tmatrix2 &m2) {
    return (dimsmatch(m1[0], m2[0]) and
            (m1.size() == m2.size()));
}

template <typename T>
inline bool dimsmatch(Tmatrix3 &m1, Tmatrix3 &m2) {
    return (dimsmatch(m1[0], m2[0]) and
            (m1.size() == m2.size()));
}

//in-place functions
namespace inplace {
    
    template <typename T>
    void elementwise_mult(Tmatrix2 *mat, real r)
    {
        for(int y=0;y<mat->size();y++) {
            for(int x=0;x<mat->at(0).size();x++) {
                mat->at(y)[x] *= r;
            }
        }
    }

    template <typename T>
    void elementwise_add(Tvec *vec, real r)
    {
        for(int x=0;x<vec->size();x++) {
            vec->at(x) += r;
        }
    }
    
    template <typename T>
    void elementwise_add(Tmatrix2 *mat, real r)
    {
        for(int y=0;y<mat->size();y++) {
            inplace::elementwise_add(&mat->at(y), r);
        }
    }
    
    template <typename T>
    void elementwise_add(Tmatrix3 *mat, real r)
    {
        for(int z=0;z<mat->size();z++) {
            inplace::elementwise_add(&mat->at(z), r);
        }
    }

    template <typename T>
    void elementwise_add(Tvec *v1, Tvec &v2)
    {
        assert(dimsmatch(*v1, v2));
        for(int x=0;x<v1->size();x++) {
                v1->at(x) += v2[x];
            }
        }
    
    template <typename T>
    void elementwise_add(Tmatrix2 *mat1, Tmatrix2 &mat2)
    {
        assert(dimsmatch(*mat1, mat2));
        for(int y=0;y<mat1->size();y++) {
            inplace::elementwise_add(&mat1->at(y), mat2);
            }
    }

    template <typename T>
    void elementwise_add(Tmatrix3 *mat1, Tmatrix3 &mat2)
    {
        assert(dimsmatch(*mat1, mat2));
        for(int z=0;z<mat1->size();z++) {
            inplace::elementwise_add(&mat1->at(z), mat2);
        }
    }
} // namespace inplace    //
 // -------------------  //
//  function  templates //

template <typename T>
ivec shapeof(Tvec &vec)
{
    size_t s[1] = {vec.size()};
    ivec output(s, s+1);
    return output;
}

template <typename T>
ivec shapeof(Tmatrix2 &mat)
{
    size_t s[2] = {mat.size(), mat[0].size()};
    ivec output(s, s+2);
    return output;
}

template <typename T>
ivec shapeof(Tmatrix3 &mat)
{
    size_t s[3] = {mat.size(), mat[0].size(), mat[0][0].size()};
    ivec output(s, s+3);
    return output;
}

template <typename T>
ivec shapeof(Tmatrix4 &mat)
{
    size_t s[3] = {mat.size(), mat[0].size(), mat[0][0].size(),
                   mat[0][0][0].size()};
    ivec output(s, s+4);
    return output;
}

template <typename T>
void printm(Tvec &vec)
{
    for(int i=0;i<vec.size();i++) {
        std::cout << vec[i] << ", ";
    }
    std::cout << std::endl;
}

template <typename T>
void printm(Tmatrix2 &mat)
{
    for(int i=0;i<mat.size();i++) {
        printm<T>(mat[i]);
    }
}

template <typename T>
void printm(Tmatrix3 &mat)
{
    for(int i=0;i<mat.size();i++) {
        printm<T>(mat[i]);
        std::cout << std::endl << std::endl;
    }
}

template <typename T = real>
void printdims(Tmatrix2 &mat)
{
    std::cout << mat.size() << ", " << mat[0].size() << std::endl;
}

template <typename T>
void printdims(Tmatrix3 &mat)
{
    std::cout << mat.size() << ", " << mat[0].size() << ", "
              << mat[0][0].size() << std::endl;
}

template <typename T>
void printdims(Tmatrix4 &mat)
{
    std::cout << mat.size() << ", " << mat[0].size() << ", "
              << mat[0][0].size() << ", " << mat[0][0][0].size() << std::endl;
}

template <typename T> Tvec
flatten(Tmatrix2 &mat)
{
    int i=0;
    Tvec output(mat.size()*mat[0].size());
    
    for(int y=0;y<mat.size();y++) {
        for(int x=0;x<mat[y].size();x++, i++) {
            output[i] = mat[y][x];
        }
    }
    return output;
}

template <typename T> Tvec
flatten(Tmatrix3 &mat)
{
    std::size_t dims[3] = {mat.size(), mat[0].size(), mat[0][0].size()};
    Tvec output(dims[0]*dims[1]*dims[2]);
    int i = 0;
    
    for(int z=0;z<dims[0];z++) {
        for(int y=0;y<dims[1];y++) {
            for(int x=0;x<dims[2];x++, i++) {
                output[i] = mat[z][y][x];
            }
        }
    }
    return output;
}

template <typename T> Tvec
flatten(Tmatrix4 &mat)
{
    std::size_t dims[4] = {mat.size(), mat[0].size(),
                            mat[0][0].size(), mat[0][0][0].size()};
    Tvec output(dims[0]*dims[1]*dims[2]*dims[3]);

    int index = 0;
    for(int z=0;z<dims[0];z++) {
        for(int y=0;y<dims[1];y++) {
            for(int x=0;x<dims[2];x++) {
                for(int i=0;i<dims[3];i++) {
                    output[index] = mat[z][y][x][i];
                }
            }
        }
    }
    return output;
}


template <typename T> Tmatrix2
fold2(Tvec &vec, ivec &shape)
{
    Tmatrix2
    output(shape[0], Tvec
          (shape[1]));
    
    assert(vec.size()== shape[0]*shape[1]);
    
    int index=0;
    for(int y=0;y<shape[0];y++) {
        for(int x=0;x<shape[1];x++, index++) {
            output[y][x] = vec[index];
        }
    }
    return output;
}

template <typename T> Tmatrix3
fold3(Tvec vec, ivec shape)
{
    Tmatrix3
    output(shape[0], Tmatrix2
          (shape[1], Tvec
          (shape[2])));
    
    assert(vec.size()== shape[0] * shape[1] * shape[2]);
    
    int index = 0;
    for(int z=0;z<shape[0];z++) {
        for(int y=0;y<shape[1];y++) {
            for(int x=0;x<shape[1];x++, index++) {
                output[z][y][x] = vec[index];
        
            }
        }
    }
    return output;
}

template <typename T> Tmatrix4
fold4(Tvec &vec, ivec &shape)
{
    Tmatrix4
    output(shape[0], Tmatrix3
          (shape[1], Tmatrix2
          (shape[2], Tvec
          (shape[3]))));
    
    assert(vec.size()== shape[0]*shape[1]*shape[2]*shape[3]);
    
    int index = 0;
    for(int z=0;z<shape[0];z++) {
        for(int y=0;y<shape[1];y++) {
            for(int x=0;x<shape[2];x++) {
                for(int i=0;i<shape[3];i++, index++) {
                    output[z][y][x][i] = vec[index];
                }
            }
        }
    }
    return output;
}

template <typename T> Tmatrix2
reshape(Tmatrix2 &mat, ivec &shape)
{
    Tmatrix2 output;
    dvec flatmat;
    
    assert(mat.size()*mat[0].size()==shape[0]*shape[1]);
    
    flatmat = flatten<T>(mat);
    output = fold2<T>(flatmat, shape);
    
    return output;
}

template <typename T> Tmatrix3
invert(Tmatrix3 mat)
{
    Tmatrix3 output
        (mat[0][0].size(), Tmatrix2(mat[0].size(), Tvec(mat.size())));
    
    for(int z=0;z<mat.size();z++) {
        for(int y=0;y<mat[0].size();y++) {
            for(int x=0;x<mat[0][0].size();x++) {
                output[x][y][z] = mat[z][y][x];
            }
        }
    }
    return output;
}

template <typename T> Tmatrix2
transpose(Tmatrix2 &mat)
{
    Tmatrix2 output(mat[0].size(), Tvec(mat.size()));
    for(int y=0;y<mat.size();y++) {
        for(int x=0;x<mat[0].size();x++) {
            output[x][y] = mat[y][x];
        }
    }
    return output;
}

template <typename T> Tmatrix2
slice(Tmatrix2 mat, ivec indices)
{
    Tmatrix2 output(indices[1]-indices[0], Tvec(indices[3]-indices[2]));
    int i=0;
    
    for(int y=indices[0];y<indices[1];y++, i++) {
        int j=0;
        for(int x=indices[2];x<indices[3];x++, j++) {
            output[i][j] = mat[y][x]; //one element gets passed!
        }
    }
    return output;
}

template <typename T> Tmatrix3
slice(Tmatrix3 mat, ivec indices)
{
    //Warning! Assuming inverted matrix!
    Tmatrix3 output(indices[1]-indices[0], Tmatrix2
                   (indices[3]-indices[2], Tvec
                   (mat[0][0].size())));
    int i=0;
    for(int x=indices[0];x<indices[1];x++, i++) {
        int j=0;
        for(int y=indices[2];y<indices[3];y++, j++) {
            output[i][j] = mat[x][y]; //vector gets passed!
        }
    }
    return output;
}

template <typename T>
T dot(Tvec &vec1, Tvec &vec2)
{   
    assert(vec1.size()==vec2.size());
    return std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0);
} 

template <typename T> Tmatrix2
dot(Tmatrix2 &mat1, Tmatrix2 &mat2)
{
    assert(mat1[0].size()==mat2[0].size());
    
    double outshape[2] = {mat1.size(), mat2.size()};
    Tmatrix2 output(outshape[0], Tvec(outshape[1], 0));
    
    for(int i=0;i<outshape[0];i++) {
        for(int j=0;j<outshape[1];j++) {
            output[i][j] = dot(mat1[i], mat2[j]);
        }
    }
    
    return output;
}

template <typename T>
T frobenius(Tmatrix2 &mat1, Tmatrix2 &mat2)
{
    assert(dimsmatch<T>(mat1, mat2));
    
    Tvec v1 = flatten<T>(mat1);
    Tvec v2 = flatten<T>(mat2);
    
    return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0);
}

template <typename T>
T frobenius(Tmatrix3 &mat1, Tmatrix3 &mat2)
{
    assert(dimsmatch<T>(mat1, mat2));
    
    Tvec v1 = flatten<T>(mat1);
    Tvec v2 = flatten<T>(mat2);
    
    return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0);
}

template <typename T> Tvec
feature_scale(Tvec &vec)
{
    Tvec output(vec.size());
    real max = *std::max_element(vec.begin(), vec.end());
    real min = *std::min_element(vec.begin(), vec.end());
    for(int x=0;x<vec.size();x++) {
        output[x] = (vec[x] - min) / (max - min);
    }
    return output;
}

template <typename T> Tmatrix2
feature_scale(Tmatrix2 &mat)
{
    ivec dims(shapeof(mat));
    Tvec vec(flatten(mat));
    Tvec outvec(dims[0]*dims[1]);
    
    real max = *std::max_element(vec.begin(), vec.end());
    real min = *std::min_element(vec.begin(), vec.end());
    for(int x=0;x<vec.size();x++) {
        outvec[x] = (vec[x] - min) / (max - min);
    }
    return fold2(outvec, dims);
}

template <typename T> Tmatrix3
feature_scale(Tmatrix3 &mat)
{
    ivec dims(shapeof(mat));
    Tvec vec(flatten(mat));
    Tvec outvec(dims[0]*dims[1]*dims[2]);
    
    real max = *std::max_element(vec.begin(), vec.end());
    real min = *std::min_element(vec.begin(), vec.end());
    for(int x=0;x<vec.size();x++) {
        outvec[x] = (vec[x] - min) / (max - min);
    }
    return fold3(outvec, dims);
}

template <typename T> Tmatrix2
elementwise_mult(Tmatrix2 &mat, real r)
{
    Tmatrix2 output(mat.size(), dvec(mat[0].size()));
    
    for(int y=0;y<mat.size();y++) {
        for(int x=0;x<mat[0].size();x++) {
            output[y][x] = mat[y][x] * r;
        }
    }
    return output;
}

template <typename T> Tmatrix2
elementwise_add(Tmatrix2 &mat, real r)
{
    Tmatrix2 output(mat.size(), dvec(mat[0].size()));
    
    for(int y=0;y<mat.size();y++) {
        for(int x=0;x<mat[0].size();x++) {
            output[y][x] = mat[y][x] + r;
        }
    }
    return output;
}
#endif /* MATOPS_H */