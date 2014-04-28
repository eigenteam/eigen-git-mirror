// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_MAP_H
#define EIGEN_CXX11_TENSOR_TENSOR_MAP_H

namespace Eigen {

template<int InnerStrideAtCompileTime, int OuterStrideAtCompileTime> class Stride;


/** \class TensorMap
  * \ingroup CXX11_Tensor_Module
  *
  * \brief A tensor expression mapping an existing array of data.
  *
  */

template<typename PlainObjectType> class TensorMap : public TensorBase<TensorMap<PlainObjectType> >
{
  public:
    typedef TensorMap<PlainObjectType> Self;
    typedef typename PlainObjectType::Base Base;
    typedef typename Eigen::internal::nested<Self>::type Nested;
    typedef typename internal::traits<PlainObjectType>::StorageKind StorageKind;
    typedef typename internal::traits<PlainObjectType>::Index Index;
    typedef typename internal::traits<PlainObjectType>::Scalar Scalar;
    typedef typename internal::packet_traits<Scalar>::type PacketScalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef typename Base::CoeffReturnType CoeffReturnType;

  /*    typedef typename internal::conditional<
                         bool(internal::is_lvalue<PlainObjectType>::value),
                         Scalar *,
                         const Scalar *>::type
                     PointerType;*/
    typedef Scalar* PointerType;
    typedef PointerType PointerArgType;

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorMap(PointerArgType dataPtr, Index firstDimension) : m_data(dataPtr), m_dimensions({{firstDimension}}) {
      // The number of dimensions used to construct a tensor must be equal to the rank of the tensor.
      EIGEN_STATIC_ASSERT(1 == PlainObjectType::NumIndices, YOU_MADE_A_PROGRAMMING_MISTAKE)
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE TensorMap(PointerArgType dataPtr, Index firstDimension, IndexTypes... otherDimensions) : m_data(dataPtr), m_dimensions({{firstDimension, otherDimensions...}}) {
      // The number of dimensions used to construct a tensor must be equal to the rank of the tensor.
      EIGEN_STATIC_ASSERT(sizeof...(otherDimensions) + 1 == PlainObjectType::NumIndices, YOU_MADE_A_PROGRAMMING_MISTAKE)
    }
#endif

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Index dimension(Index n) const { return m_dimensions[n]; }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Index size() const { return internal::array_prod(m_dimensions); }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Scalar* data() { return m_data; }
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Scalar* data() const { return m_data; }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const Scalar& operator()(Index index) const
    {
      eigen_internal_assert(index >= 0 && index < size());
      return m_data[index];
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes> EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Scalar& operator()(Index firstIndex, IndexTypes... otherIndices)
    {
      static_assert(sizeof...(otherIndices) + 1 == PlainObjectType::NumIndices, "Number of indices used to access a tensor coefficient must be equal to the rank of the tensor.");
      const Index index = internal::tensor_index_linearization_helper<Index, PlainObjectType::NumIndices, PlainObjectType::NumIndices - 1, PlainObjectType::Options&RowMajor>::run(array<Index, PlainObjectType::NumIndices>{{firstIndex, otherIndices...}}, m_dimensions);
      return m_data[index];
    }
#endif

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    Self& operator=(const OtherDerived& other)
    {
      internal::TensorAssign<Self, const OtherDerived>::run(*this, other);
      return *this;
    }

  private:
    typename PlainObjectType::Scalar* m_data;
    array<DenseIndex, PlainObjectType::NumIndices> m_dimensions;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_MAP_H
