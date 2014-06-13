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

template<typename Scalar_, std::size_t NumIndices_, int Options_>
class Tensor : public TensorBase<Tensor<Scalar_, NumIndices_, Options_> >
{
  public:
    typedef Tensor<Scalar_, NumIndices_, Options_> Self;
    typedef TensorBase<Tensor<Scalar_, NumIndices_, Options_> > Base;
    typedef typename Eigen::internal::nested<Self>::type Nested;
    typedef typename internal::traits<Self>::StorageKind StorageKind;
    typedef typename internal::traits<Self>::Index Index;
    typedef Scalar_ Scalar;
    typedef typename internal::packet_traits<Scalar>::type Packet;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef typename Base::CoeffReturnType CoeffReturnType;
    typedef typename Base::PacketReturnType PacketReturnType;

    enum {
      IsAligned = bool(EIGEN_ALIGN) & !(Options_&DontAlign),
      PacketAccess = true,
    };

    static const int Options = Options_;
    static const std::size_t NumIndices = NumIndices_;

  typedef DSizes<DenseIndex, NumIndices_> Dimensions;

  protected:
    TensorStorage<Scalar, NumIndices, Dynamic, Options> m_storage;

  public:
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index                         dimension(std::size_t n) const { return m_storage.dimensions()[n]; }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const DSizes<DenseIndex, NumIndices_>& dimensions()    const { return m_storage.dimensions(); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index                         size()                   const { return m_storage.size(); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar                        *data()                        { return m_storage.data(); }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar                  *data()                  const { return m_storage.data(); }

    // This makes EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
    // work, because that uses base().coeffRef() - and we don't yet
    // implement a similar class hierarchy
    inline Self& base()             { return *this; }
    inline const Self& base() const { return *this; }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes>
    inline const Scalar& coeff(Index firstIndex, Index secondIndex, IndexTypes... otherIndices) const
    {
      // The number of indices used to access a tensor coefficient must be equal to the rank of the tensor.
      EIGEN_STATIC_ASSERT(sizeof...(otherIndices) + 2 == NumIndices, YOU_MADE_A_PROGRAMMING_MISTAKE)
      return coeff(array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
    }
#endif

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar& coeff(const array<Index, NumIndices>& indices) const
    {
      eigen_internal_assert(checkIndexRange(indices));
      return m_storage.data()[linearizedIndex(indices)];
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar& coeff(Index index) const
    {
      eigen_internal_assert(index >= 0 && index < size());
      return m_storage.data()[index];
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes>
    inline Scalar& coeffRef(Index firstIndex, Index secondIndex, IndexTypes... otherIndices)
    {
      // The number of indices used to access a tensor coefficient must be equal to the rank of the tensor.
      EIGEN_STATIC_ASSERT(sizeof...(otherIndices) + 2 == NumIndices, YOU_MADE_A_PROGRAMMING_MISTAKE)
      return coeffRef(array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
    }
#endif

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(const array<Index, NumIndices>& indices)
    {
      eigen_internal_assert(checkIndexRange(indices));
      return m_storage.data()[linearizedIndex(indices)];
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(Index index)
    {
      eigen_internal_assert(index >= 0 && index < size());
      return m_storage.data()[index];
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes>
    inline const Scalar& operator()(Index firstIndex, Index secondIndex, IndexTypes... otherIndices) const
    {
      // The number of indices used to access a tensor coefficient must be equal to the rank of the tensor.
      EIGEN_STATIC_ASSERT(sizeof...(otherIndices) + 2 == NumIndices, YOU_MADE_A_PROGRAMMING_MISTAKE)
      return this->operator()(array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
    }
#endif

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar& operator()(const array<Index, NumIndices>& indices) const
    {
      eigen_assert(checkIndexRange(indices));
      return coeff(indices);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar& operator()(Index index) const
    {
      eigen_internal_assert(index >= 0 && index < size());
      return coeff(index);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar& operator[](Index index) const
    {
      // The bracket operator is only for vectors, use the parenthesis operator instead.
      EIGEN_STATIC_ASSERT(NumIndices == 1, YOU_MADE_A_PROGRAMMING_MISTAKE);
      return coeff(index);
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes>
    inline Scalar& operator()(Index firstIndex, Index secondIndex, IndexTypes... otherIndices)
    {
      // The number of indices used to access a tensor coefficient must be equal to the rank of the tensor.
      EIGEN_STATIC_ASSERT(sizeof...(otherIndices) + 2 == NumIndices, YOU_MADE_A_PROGRAMMING_MISTAKE)
      return operator()(array<Index, NumIndices>{{firstIndex, secondIndex, otherIndices...}});
    }
#endif

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& operator()(const array<Index, NumIndices>& indices)
    {
      eigen_assert(checkIndexRange(indices));
      return coeffRef(indices);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& operator()(Index index)
    {
      eigen_assert(index >= 0 && index < size());
      return coeffRef(index);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& operator[](Index index)
    {
      // The bracket operator is only for vectors, use the parenthesis operator instead
      EIGEN_STATIC_ASSERT(NumIndices == 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
      return coeffRef(index);
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Tensor()
      : m_storage()
    {
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Tensor(const Self& other)
      : m_storage(other.m_storage)
    {
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes>
    inline Tensor(Index firstDimension, IndexTypes... otherDimensions)
        : m_storage(internal::array_prod(array<Index, NumIndices>{{firstDimension, otherDimensions...}}), array<Index, NumIndices>{{firstDimension, otherDimensions...}})
    {
      // The number of dimensions used to construct a tensor must be equal to the rank of the tensor.
      EIGEN_STATIC_ASSERT(sizeof...(otherDimensions) + 1 == NumIndices, YOU_MADE_A_PROGRAMMING_MISTAKE)
    }
#endif

    inline Tensor(const array<Index, NumIndices>& dimensions)
      : m_storage(internal::array_prod(dimensions), dimensions)
    {
      EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE Tensor& operator=(const OtherDerived& other)
    {
      // FIXME: we need to resize the tensor to fix the dimensions of the other.
      // Unfortunately this isn't possible yet when the rhs is an expression.
      // resize(other.dimensions());
      typedef TensorAssignOp<Tensor, const OtherDerived> Assign;
      Assign assign(*this, other);
      internal::TensorExecutor<const Assign, DefaultDevice>::run(assign, DefaultDevice());
      return *this;
    }

#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
    template<typename... IndexTypes>
    void resize(Index firstDimension, IndexTypes... otherDimensions)
    {
      // The number of dimensions used to resize a tensor must be equal to the rank of the tensor.
      EIGEN_STATIC_ASSERT(sizeof...(otherDimensions) + 1 == NumIndices, YOU_MADE_A_PROGRAMMING_MISTAKE)
      resize(array<Index, NumIndices>{{firstDimension, otherDimensions...}});
    }
#endif

    void resize(const array<Index, NumIndices>& dimensions)
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

  protected:
    bool checkIndexRange(const array<Index, NumIndices>& indices) const
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

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index linearizedIndex(const array<Index, NumIndices>& indices) const
    {
      if (Options&RowMajor) {
        return m_storage.dimensions().IndexOfRowMajor(indices);
      } else {
        return m_storage.dimensions().IndexOfColMajor(indices);
      }
    }
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_H
