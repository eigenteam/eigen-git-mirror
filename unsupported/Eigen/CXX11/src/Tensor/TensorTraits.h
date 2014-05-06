// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_TRAITS_H
#define EIGEN_CXX11_TENSOR_TENSOR_TRAITS_H

namespace Eigen {
namespace internal {


template<typename Scalar, int Options>
class compute_tensor_flags
{
  enum {
    is_dynamic_size_storage = 1,

    aligned_bit =
    (
        ((Options&DontAlign)==0) && (
#if EIGEN_ALIGN_STATICALLY
            (!is_dynamic_size_storage)
#else
            0
#endif
            ||
#if EIGEN_ALIGN
            is_dynamic_size_storage
#else
            0
#endif
      )
    ) ? AlignedBit : 0,
    packet_access_bit = packet_traits<Scalar>::Vectorizable && aligned_bit ? PacketAccessBit : 0
  };

  public:
    enum { ret = packet_access_bit | aligned_bit};
};


template<typename Scalar_, std::size_t NumIndices_, int Options_>
struct traits<Tensor<Scalar_, NumIndices_, Options_> >
{
  typedef Scalar_ Scalar;
  typedef Dense StorageKind;
  typedef DenseIndex Index;
  enum {
    Options = Options_,
    Flags = compute_tensor_flags<Scalar_, Options_>::ret,
  };
};


template<typename Scalar_, typename Dimensions, int Options_>
struct traits<TensorFixedSize<Scalar_, Dimensions, Options_> >
{
  typedef Scalar_ Scalar;
  typedef Dense StorageKind;
  typedef DenseIndex Index;
};


template<typename PlainObjectType>
struct traits<TensorMap<PlainObjectType> >
  : public traits<PlainObjectType>
{
  typedef traits<PlainObjectType> BaseTraits;
  typedef typename BaseTraits::Scalar Scalar;
  typedef typename BaseTraits::StorageKind StorageKind;
  typedef typename BaseTraits::Index Index;
};


template<typename _Scalar, std::size_t NumIndices_, int Options>
struct eval<Tensor<_Scalar, NumIndices_, Options>, Eigen::Dense>
{
  typedef const Tensor<_Scalar, NumIndices_, Options>& type;
};

template<typename _Scalar, std::size_t NumIndices_, int Options>
struct eval<const Tensor<_Scalar, NumIndices_, Options>, Eigen::Dense>
{
  typedef const Tensor<_Scalar, NumIndices_, Options>& type;
};

template<typename Scalar_, typename Dimensions, int Options>
struct eval<TensorFixedSize<Scalar_, Dimensions, Options>, Eigen::Dense>
{
  typedef const TensorFixedSize<Scalar_, Dimensions, Options>& type;
};

template<typename Scalar_, typename Dimensions, int Options>
struct eval<const TensorFixedSize<Scalar_, Dimensions, Options>, Eigen::Dense>
{
  typedef const TensorFixedSize<Scalar_, Dimensions, Options>& type;
};

template<typename PlainObjectType>
struct eval<TensorMap<PlainObjectType>, Eigen::Dense>
{
  typedef const TensorMap<PlainObjectType>& type;
};

template<typename PlainObjectType>
struct eval<const TensorMap<PlainObjectType>, Eigen::Dense>
{
  typedef const TensorMap<PlainObjectType>& type;
};

template <typename Scalar_, std::size_t NumIndices_, int Options_>
struct nested<Tensor<Scalar_, NumIndices_, Options_>, 1, typename eval<Tensor<Scalar_, NumIndices_, Options_> >::type>
{
  typedef const Tensor<Scalar_, NumIndices_, Options_>& type;
};

template <typename Scalar_, std::size_t NumIndices_, int Options_>
struct nested<const Tensor<Scalar_, NumIndices_, Options_>, 1, typename eval<const Tensor<Scalar_, NumIndices_, Options_> >::type>
{
  typedef const Tensor<Scalar_, NumIndices_, Options_>& type;
};

template <typename Scalar_, typename Dimensions, int Options>
struct nested<TensorFixedSize<Scalar_, Dimensions, Options>, 1, typename eval<TensorFixedSize<Scalar_, Dimensions, Options> >::type>
{
  typedef const TensorFixedSize<Scalar_, Dimensions, Options>& type;
};

template <typename Scalar_, typename Dimensions, int Options>
struct nested<const TensorFixedSize<Scalar_, Dimensions, Options>, 1, typename eval<const TensorFixedSize<Scalar_, Dimensions, Options> >::type>
{
  typedef const TensorFixedSize<Scalar_, Dimensions, Options>& type;
};

template <typename PlainObjectType>
struct nested<TensorMap<PlainObjectType>, 1, typename eval<TensorMap<PlainObjectType> >::type>
{
  typedef const TensorMap<PlainObjectType>& type;
};

template <typename PlainObjectType>
struct nested<const TensorMap<PlainObjectType>, 1, typename eval<TensorMap<PlainObjectType> >::type>
{
  typedef const TensorMap<PlainObjectType>& type;
};

}  // end namespace internal
}  // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_TRAITS_H
