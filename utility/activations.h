/*Please refer to the LICENSE.txt file in this project's root
  for copyright informations!*/
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cmath>

#include "generic.h"

inline real sigmoid(real z)
{
    return 1 / (1 + (std::exp(-z)));
}

inline real sigmoid_p(real x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

#endif /* ACTIVATIONS_H */

