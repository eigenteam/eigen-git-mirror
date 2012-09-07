// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define SCALAR        float
#define SCALAR_SUFFIX s
#define SCALAR_SUFFIX_UP "S"
#define ISCOMPLEX     0

#include "level1_impl.h"
#include "level1_real_impl.h"
#include "level2_impl.h"
#include "level2_real_impl.h"
#include "level3_impl.h"

float BLASFUNC(sdsdot)(int* n, float* alpha, float* x, int* incx, float* y, int* incy)
{
  float res = *alpha;

  if(*n>0) {
    if(*incx==1 && *incy==1)    res += (vector(x,*n).cwiseProduct(vector(y,*n))).sum();
    else if(*incx>0 && *incy>0) res += (vector(x,*n,*incx).cwiseProduct(vector(y,*n,*incy))).sum();
    else if(*incx<0 && *incy>0) res += (vector(x,*n,-*incx).reverse().cwiseProduct(vector(y,*n,*incy))).sum();
    else if(*incx>0 && *incy<0) res += (vector(x,*n,*incx).cwiseProduct(vector(y,*n,-*incy).reverse())).sum();
    else if(*incx<0 && *incy<0) res += (vector(x,*n,-*incx).reverse().cwiseProduct(vector(y,*n,-*incy).reverse())).sum();
  }
  return res;
}
