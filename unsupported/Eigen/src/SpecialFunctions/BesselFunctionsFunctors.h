// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Eugene Brevdo <ebrevdo@gmail.com>
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BESSELFUNCTIONS_FUNCTORS_H
#define EIGEN_BESSELFUNCTIONS_FUNCTORS_H

namespace Eigen {

namespace internal {

/** \internal
 * \brief Template functor to compute the modified Bessel function of the first
 * kind of order zero.
 * \sa class CwiseUnaryOp, Cwise::i0()
 */
template <typename Scalar>
struct scalar_bessel_i0_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_bessel_i0_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x) const {
    using numext::i0;
    return i0(x);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return internal::pi0(x);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bessel_i0_op<Scalar> > {
  enum {
    // On average, a Chebyshev polynomial of order N=20 is computed.
    // The cost is N multiplications and 2N additions. We also add
    // the cost of an additional exp over i0e.
    Cost = 28 * NumTraits<Scalar>::MulCost + 48 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBessel
  };
};

/** \internal
 * \brief Template functor to compute the exponentially scaled modified Bessel
 * function of the first kind of order zero
 * \sa class CwiseUnaryOp, Cwise::i0e()
 */
template <typename Scalar>
struct scalar_bessel_i0e_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_bessel_i0e_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x) const {
    using numext::i0e;
    return i0e(x);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return internal::pi0e(x);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bessel_i0e_op<Scalar> > {
  enum {
    // On average, a Chebyshev polynomial of order N=20 is computed.
    // The cost is N multiplications and 2N additions.
    Cost = 20 * NumTraits<Scalar>::MulCost + 40 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBessel
  };
};

/** \internal
 * \brief Template functor to compute the modified Bessel function of the first
 * kind of order one
 * \sa class CwiseUnaryOp, Cwise::i1()
 */
template <typename Scalar>
struct scalar_bessel_i1_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_bessel_i1_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x) const {
    using numext::i1;
    return i1(x);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return internal::pi1(x);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bessel_i1_op<Scalar> > {
  enum {
    // On average, a Chebyshev polynomial of order N=20 is computed.
    // The cost is N multiplications and 2N additions. We also add
    // the cost of an additional exp over i1e.
    Cost = 28 * NumTraits<Scalar>::MulCost + 48 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBessel
  };
};

/** \internal
 * \brief Template functor to compute the exponentially scaled modified Bessel
 * function of the first kind of order zero
 * \sa class CwiseUnaryOp, Cwise::i1e()
 */
template <typename Scalar>
struct scalar_bessel_i1e_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_bessel_i1e_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x) const {
    using numext::i1e;
    return i1e(x);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return internal::pi1e(x);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bessel_i1e_op<Scalar> > {
  enum {
    // On average, a Chebyshev polynomial of order N=20 is computed.
    // The cost is N multiplications and 2N additions.
    Cost = 20 * NumTraits<Scalar>::MulCost + 40 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBessel
  };
};

/** \internal
 * \brief Template functor to compute the Bessel function of the second kind of
 * order zero
 * \sa class CwiseUnaryOp, Cwise::j0()
 */
template <typename Scalar>
struct scalar_bessel_j0_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_bessel_j0_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x) const {
    using numext::j0;
    return j0(x);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return internal::pj0(x);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bessel_j0_op<Scalar> > {
  enum {
    // 6 polynomial of order ~N=8 is computed.
    // The cost is N multiplications and N additions each, along with a
    // sine, cosine and rsqrt cost.
    Cost = 63 * NumTraits<Scalar>::MulCost + 48 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBessel
  };
};

/** \internal
 * \brief Template functor to compute the Bessel function of the second kind of
 * order zero
 * \sa class CwiseUnaryOp, Cwise::y0()
 */
template <typename Scalar>
struct scalar_bessel_y0_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_bessel_y0_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x) const {
    using numext::y0;
    return y0(x);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return internal::py0(x);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bessel_y0_op<Scalar> > {
  enum {
    // 6 polynomial of order ~N=8 is computed.
    // The cost is N multiplications and N additions each, along with a
    // sine, cosine, rsqrt and j0 cost.
    Cost = 126 * NumTraits<Scalar>::MulCost + 96 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBessel
  };
};

/** \internal
 * \brief Template functor to compute the Bessel function of the first kind of
 * order one
 * \sa class CwiseUnaryOp, Cwise::j1()
 */
template <typename Scalar>
struct scalar_bessel_j1_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_bessel_j1_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x) const {
    using numext::j1;
    return j1(x);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return internal::pj1(x);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bessel_j1_op<Scalar> > {
  enum {
    // 6 polynomial of order ~N=8 is computed.
    // The cost is N multiplications and N additions each, along with a
    // sine, cosine and rsqrt cost.
    Cost = 63 * NumTraits<Scalar>::MulCost + 48 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBessel
  };
};

/** \internal
 * \brief Template functor to compute the Bessel function of the second kind of
 * order one
 * \sa class CwiseUnaryOp, Cwise::j1e()
 */
template <typename Scalar>
struct scalar_bessel_y1_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_bessel_y1_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x) const {
    using numext::y1;
    return y1(x);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return internal::py1(x);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bessel_y1_op<Scalar> > {
  enum {
    // 6 polynomial of order ~N=8 is computed.
    // The cost is N multiplications and N additions each, along with a
    // sine, cosine, rsqrt and j1 cost.
    Cost = 126 * NumTraits<Scalar>::MulCost + 96 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBessel
  };
};

/** \internal
 * \brief Template functor to compute the modified Bessel function of the second
 * kind of order zero
 * \sa class CwiseUnaryOp, Cwise::k0()
 */
template <typename Scalar>
struct scalar_bessel_k0_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_bessel_k0_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x) const {
    using numext::k0;
    return k0(x);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return internal::pk0(x);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bessel_k0_op<Scalar> > {
  enum {
    // On average, a Chebyshev polynomial of order N=10 is computed.
    // The cost is N multiplications and 2N additions. In addition we compute
    // i0, a log, exp and prsqrt and sin and cos.
    Cost = 68 * NumTraits<Scalar>::MulCost + 88 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBessel
  };
};

/** \internal
 * \brief Template functor to compute the exponentially scaled modified Bessel
 * function of the second kind of order zero
 * \sa class CwiseUnaryOp, Cwise::k0e()
 */
template <typename Scalar>
struct scalar_bessel_k0e_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_bessel_k0e_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x) const {
    using numext::k0e;
    return k0e(x);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return internal::pk0e(x);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bessel_k0e_op<Scalar> > {
  enum {
    // On average, a Chebyshev polynomial of order N=10 is computed.
    // The cost is N multiplications and 2N additions. In addition we compute
    // i0, a log, exp and prsqrt and sin and cos.
    Cost = 68 * NumTraits<Scalar>::MulCost + 88 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBessel
  };
};

/** \internal
 * \brief Template functor to compute the modified Bessel function of the
 * second kind of order one
 * \sa class CwiseUnaryOp, Cwise::k1()
 */
template <typename Scalar>
struct scalar_bessel_k1_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_bessel_k1_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x) const {
    using numext::k1;
    return k1(x);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return internal::pk1(x);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bessel_k1_op<Scalar> > {
  enum {
    // On average, a Chebyshev polynomial of order N=10 is computed.
    // The cost is N multiplications and 2N additions. In addition we compute
    // i1, a log, exp and prsqrt and sin and cos.
    Cost = 68 * NumTraits<Scalar>::MulCost + 88 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBessel
  };
};

/** \internal
 * \brief Template functor to compute the exponentially scaled modified Bessel
 * function of the second kind of order one
 * \sa class CwiseUnaryOp, Cwise::k1e()
 */
template <typename Scalar>
struct scalar_bessel_k1e_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_bessel_k1e_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& x) const {
    using numext::k1e;
    return k1e(x);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    return internal::pk1e(x);
  }
};
template <typename Scalar>
struct functor_traits<scalar_bessel_k1e_op<Scalar> > {
  enum {
    // On average, a Chebyshev polynomial of order N=10 is computed.
    // The cost is N multiplications and 2N additions. In addition we compute
    // i1, a log, exp and prsqrt and sin and cos.
    Cost = 68 * NumTraits<Scalar>::MulCost + 88 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBessel
  };
};


} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_BESSELFUNCTIONS_FUNCTORS_H
