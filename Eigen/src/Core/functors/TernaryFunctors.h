// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Eugene Brevdo <ebrevdo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TERNARY_FUNCTORS_H
#define EIGEN_TERNARY_FUNCTORS_H

namespace Eigen {

namespace internal {

//---------- associative ternary functors ----------

/** \internal
  * \brief Template functor to compute the incomplete beta integral betainc(a, b, x)
  *
  */
template<typename Scalar> struct scalar_betainc_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_betainc_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& x, const Scalar& a, const Scalar& b) const {
    using numext::betainc; return betainc(x, a, b);
  }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& x, const Packet& a, const Packet& b) const
  {
    return internal::pbetainc(x, a, b);
  }
};
template<typename Scalar>
struct functor_traits<scalar_betainc_op<Scalar> > {
  enum {
    // Guesstimate
    Cost = 400 * NumTraits<Scalar>::MulCost + 400 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBetaInc
  };
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_TERNARY_FUNCTORS_H
