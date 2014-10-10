// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_IO_H
#define EIGEN_CXX11_TENSOR_TENSOR_IO_H

namespace Eigen {

template <typename T>
std::ostream& operator << (std::ostream& os, const TensorBase<T, ReadOnlyAccessors>& expr) {
  // Evaluate the expression if needed
  TensorForcedEvalOp<const T> eval = expr.eval();
  TensorEvaluator<const TensorForcedEvalOp<const T>, DefaultDevice> tensor(eval, DefaultDevice());
  tensor.evalSubExprsIfNeeded(NULL);

  typedef typename T::Scalar Scalar;
  typedef typename T::Index Index;
  typedef typename TensorEvaluator<const TensorForcedEvalOp<const T>, DefaultDevice>::Dimensions Dimensions;
  const Index total_size = internal::array_prod(tensor.dimensions());

  // Print the tensor as a 1d vector or a 2d matrix.
  if (internal::array_size<Dimensions>::value == 1) {
    Map<Array<Scalar, Dynamic, 1> > array(tensor.data(), total_size);
    os << array;
  } else {
    const Index first_dim = tensor.dimensions()[0];
    Map<Array<Scalar, Dynamic, Dynamic> > matrix(tensor.data(), first_dim, total_size/first_dim);
    os << matrix;
  }

  // Cleanup.
  tensor.cleanup();
  return os;
}

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_IO_H
