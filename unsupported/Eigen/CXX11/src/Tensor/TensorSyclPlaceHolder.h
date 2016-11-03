// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*****************************************************************
 * TensorSyclPlaceHolder.h
 *
 * \brief:
 *  The PlaceHolder expression are nothing but a container preserving
 *  the order of actual data in the tuple of sycl buffer.
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_PLACEHOLDER_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_PLACEHOLDER_HPP

namespace Eigen {
namespace internal {
/// \struct PlaceHolder
/// \brief PlaceHolder is used to replace the \ref TensorMap in the expression
/// tree.
/// PlaceHolder contains the order of the leaf node in the expression tree.
template <typename Scalar, size_t N>
struct PlaceHolder {
  static constexpr size_t I = N;
  typedef Scalar Type;
};

/// \brief specialisation of the PlaceHolder node for const TensorMap
#define TENSORMAPPLACEHOLDER(CVQual)\
template <typename PlainObjectType, int Options_, template <class> class MakePointer_, size_t N>\
struct PlaceHolder<CVQual TensorMap<PlainObjectType, Options_, MakePointer_>, N> {\
  static const size_t I = N;\
  typedef CVQual TensorMap<PlainObjectType, Options_, MakePointer_> Type;\
  typedef typename Type::Self Self;\
  typedef typename Type::Base Base;\
  typedef typename Type::Nested Nested;\
  typedef typename Type::StorageKind StorageKind;\
  typedef typename Type::Index Index;\
  typedef typename Type::Scalar Scalar;\
  typedef typename Type::RealScalar RealScalar;\
  typedef typename Type::CoeffReturnType CoeffReturnType;\
};

TENSORMAPPLACEHOLDER(const)
TENSORMAPPLACEHOLDER()
#undef TENSORMAPPLACEHOLDER

/// \brief specialisation of the PlaceHolder node for TensorForcedEvalOp. The
/// TensorForcedEvalOp acts as a leaf node for its parent node.
#define TENSORFORCEDEVALPLACEHOLDER(CVQual)\
template <typename Expression, size_t N>\
struct PlaceHolder<CVQual TensorForcedEvalOp<Expression>, N> {\
  static const size_t I = N;\
  typedef CVQual  TensorForcedEvalOp<Expression> Type;\
  typedef typename Type::Nested Nested;\
  typedef typename Type::StorageKind StorageKind;\
  typedef typename Type::Index Index;\
  typedef typename Type::Scalar Scalar;\
  typedef typename Type::Packet Packet;\
  typedef typename Type::RealScalar RealScalar;\
  typedef typename Type::CoeffReturnType CoeffReturnType;\
  typedef typename Type::PacketReturnType PacketReturnType;\
};

TENSORFORCEDEVALPLACEHOLDER(const)
TENSORFORCEDEVALPLACEHOLDER()
#undef TENSORFORCEDEVALPLACEHOLDER

template <typename PlainObjectType, int Options_, template <class> class Makepointer_, size_t N>
struct traits<PlaceHolder<const TensorMap<PlainObjectType, Options_, Makepointer_>, N> >: public traits<PlainObjectType> {
  typedef traits<PlainObjectType> BaseTraits;
  typedef typename BaseTraits::Scalar Scalar;
  typedef typename BaseTraits::StorageKind StorageKind;
  typedef typename BaseTraits::Index Index;
  static const int NumDimensions = BaseTraits::NumDimensions;
  static const int Layout = BaseTraits::Layout;
  enum {
    Options = Options_,
    Flags = BaseTraits::Flags,
  };
};

template <typename PlainObjectType, int Options_, template <class> class Makepointer_, size_t N>
struct traits<PlaceHolder<TensorMap<PlainObjectType, Options_, Makepointer_>, N> >
: traits<PlaceHolder<const TensorMap<PlainObjectType, Options_, Makepointer_>, N> > {};

}  // end namespace internal
}  // end namespoace Eigen

#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCL_PLACEHOLDER_HPP
