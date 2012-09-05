// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define SCALAR        double
#define SCALAR_SUFFIX d
#define SCALAR_SUFFIX_UP "D"
#define ISCOMPLEX     0

#include "level1_impl.h"
#include "level1_real_impl.h"
#include "level2_impl.h"
#include "level2_real_impl.h"
#include "level3_impl.h"

// currently used by DSDOT only
double* cast_vector_to_double(float* x, int n, int incx)
{
  double* ret = new double[n];
  if(incx<0) vector(ret,n) = vector(x,n,-incx).reverse().cast<double>();
  else       vector(ret,n) = vector(x,n, incx).cast<double>();
  return ret;
}

double BLASFUNC(dsdot)(int* n, float* px, int* incx, float* py, int* incy)
{
  if(*n <= 0) return 0;
  double* x = cast_vector_to_double(px, *n, *incx);
  double* y = cast_vector_to_double(py, *n, *incy);
  return vector(x,*n).cwiseProduct(vector(y,*n)).sum();
}

