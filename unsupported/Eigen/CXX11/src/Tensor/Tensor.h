// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Christian Seiler <christian@iwakd.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_H
#define EIGEN_CXX11_TENSOR_TENSOR_H

namespace Eigen {

/** \class Tensor
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor class.
  *
  * The %Tensor class is the work-horse for all \em dense tensors within Eigen.
  *
  * The %Tensor class encompasses only dynamic-size objects so far.
  *
  * The first two template parameters are required:
  * \tparam Scalar_ \anchor tensor_tparam_scalar Numeric type, e.g. float, double, int or std::complex<float>.
  *                 User defined scalar types are supported as well (see \ref user_defined_scalars "here").
  * \tparam NumIndices_ Number of indices (i.e. rank of the tensor)
  *
  * The remaining template parameters are optional -- in most cases you don't have to worry about them.
  * \tparam Options_ \anchor tensor_tparam_options A combination of either \b #RowMajor or \b #ColMajor, and of either
  *                 \b #AutoAlign or \b #DontAlign.
  *                 The former controls \ref TopicStorageOrders "storage order", and defaults to column-major. The latter controls alignment, which is required
  *                 for vectorization. It defaults to aligning tensors. Note that tensors currently do not support any operations that profit from vectorization.
  *                 Support for such operations (i.e. adding two tensors etc.) is planned.
  *
  * You can access elements of tensors using normal subscripting:
  *
  * \code
  * Eigen::Tensor<double, 4> t(10, 10, 10, 10);
  * t(0, 1, 2, 3) = 42.0;
  * \endcode
  *
  * This class can be extended with the help of the plugin mechanism described on the page
  * \ref TopicCustomizingEigen by defining the preprocessor symbol \c EIGEN_TENSOR_PLUGIN.
  *
  * <i><b>Some notes:</b></i>
  *
  * <dl>
  * <dt><b>Relation to other parts of Eigen:</b></dt>
  * <dd>The midterm developement goal for this class is to have a similar hierarchy as Eigen uses for matrices, so that
  * taking blocks or using tensors in expressions is easily possible, including an interface with the vector/matrix code
  * by providing .asMatrix() and .asVector() (or similar) methods for rank 2 and 1 tensors. However, currently, the %Tensor
  * class does not provide any of these features and is only available as a stand-alone class that just allows for
  * coefficient access. Also, when fixed-size tensors are implemented, the number of template arguments is likely to
  * change dramatically.</dd>
  * </dl>
  *
  * \ref TopicStorageOrders 
  */
template<typename Scalar_, std::size_t NumIndices_, int Options_ = 0>
class Tensor;

namespace internal {
template<typename Scalar_, std::size_t NumIndices_, int Options_>
struct traits<Tensor<Scalar_, NumIndices_, Options_>>
{
  typedef Scalar_ Scalar;
  typedef Dense StorageKind;
  typedef DenseIndex Index;
  enum {
    Options = Options_
  };
};

template<typename Index, std::size_t NumIndices, std::size_t n, bool RowMajor>
struct tensor_index_linearization_helper
{
  constexpr static inline Index run(std::array<Index, NumIndices> const& indices, std::array<Index, NumIndices> const& dimensions)
  {
    return std_array_get<RowMajor ? n : (NumIndices - n - 1)>(indices) + 
      std_array_get<RowMajor ? n : (NumIndices - n - 1)>(dimensions) *
        tensor_index_linearization_helper<Index, NumIndices, n - 1, RowMajor>::run(indices, dimensions);
  }
};

template<typename Index, std::size_t NumIndices, bool RowMajor>
struct tensor_index_linearization_helper<Index, NumIndices, 0, RowMajor>
{
  constexpr static inline Index run(std::array<Index, NumIndices> const& indices, std::array<Index, NumIndices> const&)
  {
    return std_array_get<RowMajor ? 0 : NumIndices - 1>(indices);
  }
};

/* Forward-declaration required for the symmetry support. */
template<typename Tensor_, typename Symmetry_, int Flags = 0> class tensor_symmetry_value_setter;
} // end namespace internal

template<typename Scalar_, std::size_t NumIndices_, int Options_>
class Tensor
{
    static_assert(NumIndices_ >= 1, "A tensor must have at least one index.");
  
  public:
    typedef Tensor<Scalar_, NumIndices_, Options_> Self;
    typedef typename internal::traits<Self>::StorageKind StorageKind;
    typedef typename internal::traits<Self>::Index Index;
    typedef typename internal::traits<Self>::Scalar Scalar;
    typedef typename internal::packet_traits<Scalar>::type PacketScalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef Self DenseType;

    constexpr static int Options = Options_;
    constexpr static std::size_t NumIndices = NumIndices_;

  protected:
    TensorStorage<Scalar, NumIndices, Dynamic, Options> m_storage;

  public:
    EIGEN_STRONG_INLINE Index                         dimension(std::size_t n) const { return m_storage.dimensions()[n]; }
    EIGEN_STRONG_INLINE std::array<Index, NumIndices> dimensions()             const { return m_storage.dimensions(); }
    EIGEN_STRONG_INLINE Index                         size()                   const { return internal::array_prod(m_storage.dimensions()); }
    EIGEN_STRONG_INLINE Scalar                        *data()                        { return m_storage.data(); }
    EIGEN_STRONG_INLINE const Scalar                  *data()                  const { return m_storage.data(); }

    // This makes EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
    // work, because that uses base().coeffRef() - and we don't yet
    // implement a similar class hierarchy
    inline Self& base()             { return *this; }
    inline const Self& base() const { return *this; }

    void setZero()
    {
      // FIXME: until we have implemented packet access and the
      //        expression engine w.r.t. nullary ops, use this
      //        as a kludge. Only works with POD types, but for
      //        any standard usage, this shouldn't be a problem
      memset((void *)data(), 0, size() * sizeof(Scalar));
    }

    inline Self& operator=(Self const& other)
    {
      m_storage = other.m_storage;
      return *this;
    }

    template<typename... IndexTypes>
    inline const Scalar& coeff(Index firstIndex, Index secondIndex, IndexTypes... otherIndices) const
    {
      static_assert(sizeof...(otherIndices) + 2 == NumIndices, "Number of indices used to access a tensor coefficient must be equal to the rank of the tensor.");
      return coeff(std::array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
    }

    inline const Scalar& coeff(const std::array<Index, NumIndices>& indices) const
    {
      eigen_internal_assert(checkIndexRange(indices));
      return m_storage.data()[linearizedIndex(indices)];
    }

    inline const Scalar& coeff(Index index) const
    {
      eigen_internal_assert(index >= 0 && index < size());
      return m_storage.data()[index];
    }

    template<typename... IndexTypes>
    inline Scalar& coeffRef(Index firstIndex, Index secondIndex, IndexTypes... otherIndices)
    {
      static_assert(sizeof...(otherIndices) + 2 == NumIndices, "Number of indices used to access a tensor coefficient must be equal to the rank of the tensor.");
      return coeffRef(std::array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
    }

    inline Scalar& coeffRef(const std::array<Index, NumIndices>& indices)
    {
      eigen_internal_assert(checkIndexRange(indices));
      return m_storage.data()[linearizedIndex(indices)];
    }

    inline Scalar& coeffRef(Index index)
    {
      eigen_internal_assert(index >= 0 && index < size());
      return m_storage.data()[index];
    }

    template<typename... IndexTypes>
    inline const Scalar& operator()(Index firstIndex, Index secondIndex, IndexTypes... otherIndices) const
    {
      static_assert(sizeof...(otherIndices) + 2 == NumIndices, "Number of indices used to access a tensor coefficient must be equal to the rank of the tensor.");
      return this->operator()(std::array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
    }

    inline const Scalar& operator()(const std::array<Index, NumIndices>& indices) const
    {
      eigen_assert(checkIndexRange(indices));
      return coeff(indices);
    }

    inline const Scalar& operator()(Index index) const
    {
      eigen_internal_assert(index >= 0 && index < size());
      return coeff(index);
    }

    inline const Scalar& operator[](Index index) const
    {
      static_assert(NumIndices == 1, "The bracket operator is only for vectors, use the parenthesis operator instead.");
      return coeff(index);
    }

    template<typename... IndexTypes>
    inline Scalar& operator()(Index firstIndex, Index secondIndex, IndexTypes... otherIndices)
    {
      static_assert(sizeof...(otherIndices) + 2 == NumIndices, "Number of indices used to access a tensor coefficient must be equal to the rank of the tensor.");
      return operator()(std::array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
    }

    inline Scalar& operator()(const std::array<Index, NumIndices>& indices)
    {
      eigen_assert(checkIndexRange(indices));
      return coeffRef(indices);
    }

    inline Scalar& operator()(Index index)
    {
      eigen_assert(index >= 0 && index < size());
      return coeffRef(index);
    }

    inline Scalar& operator[](Index index)
    {
      static_assert(NumIndices == 1, "The bracket operator is only for vectors, use the parenthesis operator instead.");
      return coeffRef(index);
    }

    inline Tensor()
      : m_storage()
    {
    }

    inline Tensor(const Self& other)
      : m_storage(other.m_storage)
    {
    }

    inline Tensor(Self&& other)
      : m_storage(other.m_storage)
    {
    }

    template<typename... IndexTypes>
    inline Tensor(Index firstDimension, IndexTypes... otherDimensions)
      : m_storage()
    {
      static_assert(sizeof...(otherDimensions) + 1 == NumIndices, "Number of dimensions used to construct a tensor must be equal to the rank of the tensor.");
      resize(std::array<Index, NumIndices>{{firstDimension, otherDimensions...}});
    }

    inline Tensor(std::array<Index, NumIndices> dimensions)
      : m_storage(internal::array_prod(dimensions), dimensions)
    {
      EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
    }

    template<typename... IndexTypes>
    void resize(Index firstDimension, IndexTypes... otherDimensions)
    {
      static_assert(sizeof...(otherDimensions) + 1 == NumIndices, "Number of dimensions used to resize a tensor must be equal to the rank of the tensor.");
      resize(std::array<Index, NumIndices>{{firstDimension, otherDimensions...}});
    }

    void resize(const std::array<Index, NumIndices>& dimensions)
    {
      std::size_t i;
      Index size = Index(1);
      for (i = 0; i < NumIndices; i++) {
        internal::check_rows_cols_for_overflow<Dynamic>::run(size, dimensions[i]);
        size *= dimensions[i];
      }
      #ifdef EIGEN_INITIALIZE_COEFFS
        bool size_changed = size != this->size();
        m_storage.resize(size, dimensions);
        if(size_changed) EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
      #else
        m_storage.resize(size, dimensions);
      #endif
    }

    template<typename Symmetry_, typename... IndexTypes>
    internal::tensor_symmetry_value_setter<Self, Symmetry_> symCoeff(const Symmetry_& symmetry, Index firstIndex, IndexTypes... otherIndices)
    {
      return symCoeff(symmetry, std::array<Index, NumIndices>{{firstIndex, otherIndices...}});
    }

    template<typename Symmetry_, typename... IndexTypes>
    internal::tensor_symmetry_value_setter<Self, Symmetry_> symCoeff(const Symmetry_& symmetry, std::array<Index, NumIndices> const& indices)
    {
      return internal::tensor_symmetry_value_setter<Self, Symmetry_>(*this, symmetry, indices);
    }

  protected:
    bool checkIndexRange(const std::array<Index, NumIndices>& indices) const
    {
      using internal::array_apply_and_reduce;
      using internal::array_zip_and_reduce;
      using internal::greater_equal_zero_op;
      using internal::logical_and_op;
      using internal::lesser_op;

      return
        // check whether the indices are all >= 0
        array_apply_and_reduce<logical_and_op, greater_equal_zero_op>(indices) &&
        // check whether the indices fit in the dimensions
        array_zip_and_reduce<logical_and_op, lesser_op>(indices, m_storage.dimensions());
    }

    inline Index linearizedIndex(const std::array<Index, NumIndices>& indices) const
    {
      return internal::tensor_index_linearization_helper<Index, NumIndices, NumIndices - 1, Options&RowMajor>::run(indices, m_storage.dimensions());
    }
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_H

/*
 * kate: space-indent on; indent-width 2; mixedindent off; indent-mode cstyle;
 */
