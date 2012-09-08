// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_RANK2UPDATE_H
#define EIGEN_RANK2UPDATE_H

namespace internal {

/* Optimized selfadjoint matrix += alpha * uv' + conj(alpha)*vu'
 * This is the low-level version of SelfadjointRank2Update.h
 */
template<typename Scalar, typename Index, int UpLo>
struct rank2_update_selector;

template<typename Scalar, typename Index>
struct rank2_update_selector<Scalar,Index,Upper>
{
  static void run(Index size, Scalar* mat, Index stride, const Scalar* _u, const Scalar* _v, Scalar alpha)
  {
    typedef Matrix<Scalar,Dynamic,1> PlainVector;
    Map<const PlainVector> u(_u, size), v(_v, size);

    for (Index i=0; i<size; ++i)
    {
      Map<PlainVector>(mat+stride*i, i+1) += conj(alpha) * conj(_u[i]) * v.head(i+1)
					  +  alpha * conj(_v[i]) * u.head(i+1);
    }
  }
};

template<typename Scalar, typename Index>
struct rank2_update_selector<Scalar,Index,Lower>
{
  static void run(Index size, Scalar* mat, Index stride, const Scalar* _u, const Scalar* _v, Scalar alpha)
  {
    typedef Matrix<Scalar,Dynamic,1> PlainVector;
    Map<const PlainVector> u(_u, size), v(_v, size);

    for (Index i=0; i<size; ++i)
    {
      Map<PlainVector>(mat+(stride+1)*i, size-i) += conj(alpha) * conj(_u[i]) * v.tail(size-i)
						 +  alpha * conj(_v[i]) * u.tail(size-i);
    }
  }
};

/* Optimized selfadjoint matrix += alpha * uv' + conj(alpha)*vu'
 * The matrix is in packed form.
 */
template<typename Scalar, typename Index, int UpLo>
struct packed_rank2_update_selector;

template<typename Scalar, typename Index>
struct packed_rank2_update_selector<Scalar,Index,Upper>
{
  static void run(Index size, Scalar* mat, const Scalar* _u, const Scalar* _v, Scalar alpha)
  {
    typedef Matrix<Scalar,Dynamic,1> PlainVector;
    Map<const PlainVector> u(_u, size), v(_v, size);
    Index offset = 0;

    for (Index i=0; i<size; ++i)
    {
      offset += i;
      Map<PlainVector>(mat+offset, i+1) += conj(alpha) * conj(_u[i]) * v.head(i+1)
					+  alpha * conj(_v[i]) * u.head(i+1);
      mat[offset+i] = real(mat[offset+i]);
    }
  }
};

template<typename Scalar, typename Index>
struct packed_rank2_update_selector<Scalar,Index,Lower>
{
  static void run(Index size, Scalar* mat, const Scalar* _u, const Scalar* _v, Scalar alpha)
  {
    typedef Matrix<Scalar,Dynamic,1> PlainVector;
    Map<const PlainVector> u(_u, size), v(_v, size);
    Index offset = 0;

    for (Index i=0; i<size; ++i)
    {
      Map<PlainVector>(mat+offset, size-i) += conj(alpha) * conj(_u[i]) * v.tail(size-i)
					   +  alpha * conj(_v[i]) * u.tail(size-i);
      mat[offset] = real(mat[offset]);
      offset += size-i;
    }
  }
};

} // end namespace internal

#endif // EIGEN_RANK2UPDATE_H
