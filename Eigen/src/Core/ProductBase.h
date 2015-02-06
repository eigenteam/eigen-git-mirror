// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PRODUCTBASE_H
#define EIGEN_PRODUCTBASE_H

namespace Eigen { 

/** \internal
  * Overloaded to perform an efficient C = (A*B).lazy() */
template<typename Derived>
template<typename ProductDerived, typename Lhs, typename Rhs>
Derived& MatrixBase<Derived>::lazyAssign(const ProductBase<ProductDerived, Lhs,Rhs>& other)
{
  other.derived().evalTo(derived());
  return derived();
}

} // end namespace Eigen

#endif // EIGEN_PRODUCTBASE_H
