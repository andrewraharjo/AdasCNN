/*Please refer to the LICENSE.txt file in this project's root
  for copyright informations!*/
#ifndef TYPES_H
#define TYPES_H

#include <vector>

typedef double real;

typedef std::vector<real> dvec;
typedef std::vector<int> ivec;

typedef std::vector<dvec> dmatrix2;
typedef std::vector<ivec> imatrix2;

typedef std::vector<dmatrix2> dmatrix3;
typedef std::vector<imatrix2> imatrix3;

typedef std::vector<dmatrix3> dmatrix4;

/* Will do...
template<typename T>
using Tvec = std::vector<T>;

template<typename T>
using Tmatrix2 = std::vector<Tvec>;

template<typename T>
using Tmatrix3 = std::vector<Tmatrix2>;
*/

#endif /* TYPES_H */

