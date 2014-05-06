// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_DIMENSIONS_H
#define EIGEN_CXX11_TENSOR_TENSOR_DIMENSIONS_H


namespace Eigen {

/** \internal
  *
  * \class TensorDimensions
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Set of classes used to encode and store the dimensions of a Tensor.
  *
  * The Sizes class encodes as part of the type the number of dimensions and the
  * sizes corresponding to each dimension. It uses no storage space since it is
  * entirely known at compile time.
  * The DSizes class is its dynamic sibling: the number of dimensions is known
  * at compile time but the sizes are set during execution.
  *
  * \sa Tensor
  */



// Boiler plate code
namespace internal {

template<std::size_t n, typename Dimension> struct dget {
  static const std::size_t value = internal::get<n, typename Dimension::Base>::value;
  };


template<typename Index, std::size_t NumIndices, std::size_t n, bool RowMajor>
struct fixed_size_tensor_index_linearization_helper
{
  template <typename Dimensions>
  static inline Index run(array<Index, NumIndices> const& indices,
                          const Dimensions& dimensions)
  {
    return array_get<RowMajor ? n : (NumIndices - n - 1)>(indices) +
        dget<RowMajor ? n : (NumIndices - n - 1), Dimensions>::value *
        fixed_size_tensor_index_linearization_helper<Index, NumIndices, n - 1, RowMajor>::run(indices, dimensions);
  }
};

template<typename Index, std::size_t NumIndices, bool RowMajor>
struct fixed_size_tensor_index_linearization_helper<Index, NumIndices, 0, RowMajor>
{
  template <typename Dimensions>
  static inline Index run(array<Index, NumIndices> const& indices,
                          const Dimensions&)
  {
    return array_get<RowMajor ? 0 : NumIndices - 1>(indices);
  }
};

}  // end namespace internal


// Fixed size
#ifndef EIGEN_EMULATE_CXX11_META_H
template <typename std::size_t... Indices>
struct Sizes : internal::numeric_list<std::size_t, Indices...> {
  typedef internal::numeric_list<std::size_t, Indices...> Base;
  static const std::size_t total_size = internal::arg_prod(Indices...);

  static std::size_t TotalSize() {
    return internal::arg_prod(Indices...);
  }

  Sizes() { }
  template <typename DenseIndex>
  explicit Sizes(const array<DenseIndex, Base::count>&/* indices*/) {
    // todo: add assertion
  }
#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
  explicit Sizes(std::initializer_list<std::size_t>/* l*/) {
    // todo: add assertion
  }
#endif

  template <typename T> Sizes& operator = (const T&/* other*/) {
    // add assertion failure if the size of other is different
    return *this;
  }

  template <typename DenseIndex>
  size_t IndexOfColMajor(const array<DenseIndex, Base::count>& indices) const {
    return internal::fixed_size_tensor_index_linearization_helper<DenseIndex, Base::count, Base::count - 1, false>::run(indices, *static_cast<const Base*>(this));
  }
  template <typename DenseIndex>
  size_t IndexOfRowMajor(const array<DenseIndex, Base::count>& indices) const {
    return internal::fixed_size_tensor_index_linearization_helper<DenseIndex, Base::count, Base::count - 1, true>::run(indices, *static_cast<const Base*>(this));
  }
};

#else

template <std::size_t n>
struct non_zero_size {
  typedef internal::type2val<std::size_t, n> type;
};
template <>
struct non_zero_size<0> {
  typedef internal::null_type type;
};

template <std::size_t V1=0, std::size_t V2=0, std::size_t V3=0, std::size_t V4=0, std::size_t V5=0> struct Sizes {
  typedef typename internal::make_type_list<typename non_zero_size<V1>::type, typename non_zero_size<V2>::type, typename non_zero_size<V3>::type, typename non_zero_size<V4>::type, typename non_zero_size<V5>::type >::type Base;
  static const size_t count = Base::count;
  static const std::size_t total_size = internal::arg_prod<Base>::value;

  static const size_t TotalSize() {
    return internal::arg_prod<Base>::value;
  }

  Sizes() { }
  template <typename DenseIndex>
  explicit Sizes(const array<DenseIndex, Base::count>& indices) {
    // todo: add assertion
  }
#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
  explicit Sizes(std::initializer_list<std::size_t> l) {
    // todo: add assertion
  }
#endif

  template <typename T> Sizes& operator = (const T& other) {
    // to do: check the size of other
    return *this;
  }

  template <typename DenseIndex>
  size_t IndexOfColMajor(const array<DenseIndex, Base::count>& indices) const {
    return internal::fixed_size_tensor_index_linearization_helper<DenseIndex, Base::count, Base::count - 1, false>::run(indices, *this);
  }
  template <typename DenseIndex>
  size_t IndexOfRowMajor(const array<DenseIndex, Base::count>& indices) const {
    return internal::fixed_size_tensor_index_linearization_helper<DenseIndex, Base::count, Base::count - 1, true>::run(indices, *this);
  }
};

#endif

// Boiler plate
namespace internal {
template<typename Index, std::size_t NumIndices, std::size_t n, bool RowMajor>
struct tensor_index_linearization_helper
{
  static inline Index run(array<Index, NumIndices> const& indices, array<Index, NumIndices> const& dimensions)
  {
    return array_get<RowMajor ? n : (NumIndices - n - 1)>(indices) +
      array_get<RowMajor ? n : (NumIndices - n - 1)>(dimensions) *
        tensor_index_linearization_helper<Index, NumIndices, n - 1, RowMajor>::run(indices, dimensions);
  }
};

template<typename Index, std::size_t NumIndices, bool RowMajor>
struct tensor_index_linearization_helper<Index, NumIndices, 0, RowMajor>
{
  static inline Index run(array<Index, NumIndices> const& indices, array<Index, NumIndices> const&)
  {
    return array_get<RowMajor ? 0 : NumIndices - 1>(indices);
  }
};
}  // end namespace internal



// Dynamic size
template <typename DenseIndex, std::size_t NumDims>
struct DSizes : array<DenseIndex, NumDims> {
  typedef array<DenseIndex, NumDims> Base;

  size_t TotalSize() const {
    return internal::array_prod(*static_cast<const Base*>(this));
  }

  DSizes() { }
#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
  //  explicit DSizes(std::initializer_list<DenseIndex> l) : Base(l) { }
#endif
  explicit DSizes(const array<DenseIndex, NumDims>& a) : Base(a) { }

  DSizes& operator = (const array<DenseIndex, NumDims>& other) {
    *static_cast<Base*>(this) = other;
    return *this;
  }

  // A constexpr would be so much better here
  size_t IndexOfColMajor(const array<DenseIndex, NumDims>& indices) const {
    return internal::tensor_index_linearization_helper<DenseIndex, NumDims, NumDims - 1, false>::run(indices, *static_cast<const Base*>(this));
  }
  size_t IndexOfRowMajor(const array<DenseIndex, NumDims>& indices) const {
    return internal::tensor_index_linearization_helper<DenseIndex, NumDims, NumDims - 1, true>::run(indices, *static_cast<const Base*>(this));
  }

};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_DIMENSIONS_H
