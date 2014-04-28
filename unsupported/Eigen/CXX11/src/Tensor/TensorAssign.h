// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_ASSIGN_H
#define EIGEN_CXX11_TENSOR_TENSOR_ASSIGN_H


namespace Eigen {

/** \class TensorAssign
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor assignment class.
  *
  * This class is responsible for triggering the evaluation of the expressions
  * used on the lhs and rhs of an assignment operator and copy the result of
  * the evaluation of the rhs expression at the address computed during the
  * evaluation lhs expression.
  *
  * TODO: vectorization. For now the code only uses scalars
  * TODO: parallelisation using multithreading on cpu, or kernels on gpu.
  */
namespace internal {

template<typename Derived1, typename Derived2>
struct TensorAssign
{
  typedef typename Derived1::Index Index;
  EIGEN_DEVICE_FUNC
  static inline void run(Derived1& dst, const Derived2& src)
  {
    TensorEvaluator<Derived1> evalDst(dst);
    TensorEvaluator<Derived2> evalSrc(src);
    const Index size = dst.size();
    for(Index i = 0; i < size; ++i) {
      evalDst.coeffRef(i) = evalSrc.coeff(i);
    }
  }
};


} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_ASSIGN_H
