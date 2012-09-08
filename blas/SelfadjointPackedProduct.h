// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SELFADJOINT_PACKED_PRODUCT_H
#define EIGEN_SELFADJOINT_PACKED_PRODUCT_H

namespace internal {

/* Optimized matrix += alpha * uv'
 * The matrix is in packed form.
 *
 * FIXME I always fail tests for complex self-adjoint matrices.
 *
 ******* FATAL ERROR - PARAMETER NUMBER  6 WAS CHANGED INCORRECTLY *******
 ******* xHPR   FAILED ON CALL NUMBER:
      2: xHPR  ('U',  1, 0.0, X, 1, AP)
 */
template<typename Scalar, typename Index, int UpLo>
struct selfadjoint_packed_rank1_update
{
  static void run(Index size, Scalar* mat, const Scalar* vec, Scalar alpha)
  {
    typedef Map<const Matrix<Scalar,Dynamic,1> > OtherMap;
    Index offset = 0;

    for (Index i=0; i<size; ++i)
    {
      Map<Matrix<Scalar,Dynamic,1> >(mat+offset, UpLo==Lower ? size-i : (i+1))
	  += alpha * conj(vec[i]) * OtherMap(vec+(UpLo==Lower ? i : 0), UpLo==Lower ? size-i : (i+1));
      //FIXME This should be handled outside.
      mat[offset+(UpLo==Lower ? 0 : i)] = real(mat[offset+(UpLo==Lower ? 0 : i)]);
      offset += UpLo==Lower ? size-i : (i+1);
    }
  }
};

//TODO struct selfadjoint_packed_product_selector

} // end namespace internal

#endif // EIGEN_SELFADJOINT_PACKED_PRODUCT_H
