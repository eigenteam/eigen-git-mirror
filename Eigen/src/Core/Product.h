// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_PRODUCT_H
#define EIGEN_PRODUCT_H

/***************************
*** Forward declarations ***
***************************/

template<int VectorizationMode, int Index, typename Lhs, typename Rhs>
struct ei_product_coeff_impl;

template<int StorageOrder, int Index, typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl;

template<typename T> class ei_product_eval_to_column_major;

/** \class ProductReturnType
  *
  * \brief Helper class to get the correct and optimized returned type of operator*
  *
  * \param Lhs the type of the left-hand side
  * \param Rhs the type of the right-hand side
  * \param ProductMode the type of the product (determined automatically by ei_product_mode)
  *
  * This class defines the typename Type representing the optimized product expression
  * between two matrix expressions. In practice, using ProductReturnType<Lhs,Rhs>::Type
  * is the recommended way to define the result type of a function returning an expression
  * which involve a matrix product. The class Product or DiagonalProduct should never be
  * used directly.
  *
  * \sa class Product, class DiagonalProduct, MatrixBase::operator*(const MatrixBase<OtherDerived>&)
  */
template<typename Lhs, typename Rhs, int ProductMode>
struct ProductReturnType
{
  typedef typename ei_nested<Lhs,Rhs::ColsAtCompileTime>::type LhsNested;
  typedef typename ei_nested<Rhs,Lhs::RowsAtCompileTime>::type RhsNested;

  typedef Product<typename ei_unconst<LhsNested>::type,
                  typename ei_unconst<RhsNested>::type, ProductMode> Type;
};

// cache friendly specialization
template<typename Lhs, typename Rhs>
struct ProductReturnType<Lhs,Rhs,CacheFriendlyProduct>
{
  typedef typename ei_nested<Lhs,Rhs::ColsAtCompileTime>::type LhsNested;

  typedef typename ei_nested<Rhs,Lhs::RowsAtCompileTime,
              typename ei_product_eval_to_column_major<Rhs>::type
          >::type RhsNested;

  typedef Product<typename ei_unconst<LhsNested>::type,
                  typename ei_unconst<RhsNested>::type, CacheFriendlyProduct> Type;
};

/*  Helper class to determine the type of the product, can be either:
 *    - NormalProduct
 *    - CacheFriendlyProduct
 *    - NormalProduct
 */
template<typename Lhs, typename Rhs> struct ei_product_mode
{
  enum{ value = ((Rhs::Flags&Diagonal)==Diagonal) || ((Lhs::Flags&Diagonal)==Diagonal)
              ? DiagonalProduct
              : (Rhs::Flags & Lhs::Flags & SparseBit)
              ? SparseProduct
              :    Lhs::MaxRowsAtCompileTime >= EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
                && Rhs::MaxColsAtCompileTime >= EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
                && Lhs::MaxColsAtCompileTime >= EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
                ? CacheFriendlyProduct : NormalProduct };
};

/** \class Product
  *
  * \brief Expression of the product of two matrices
  *
  * \param LhsNested the type used to store the left-hand side
  * \param RhsNested the type used to store the right-hand side
  * \param ProductMode the type of the product
  *
  * This class represents an expression of the product of two matrices.
  * It is the return type of the operator* between matrices. Its template
  * arguments are determined automatically by ProductReturnType. Therefore,
  * Product should be used direclty. To determine the result type of a function
  * which involve a matrix product, use ProductReturnType::Type.
  *
  * \sa ProductReturnType, MatrixBase::operator*(const MatrixBase<OtherDerived>&)
  */
template<typename LhsNested, typename RhsNested, int ProductMode>
struct ei_traits<Product<LhsNested, RhsNested, ProductMode> >
{
  // clean the nested types:
  typedef typename ei_unconst<typename ei_unref<LhsNested>::type>::type _LhsNested;
  typedef typename ei_unconst<typename ei_unref<RhsNested>::type>::type _RhsNested;
  typedef typename _LhsNested::Scalar Scalar;

  enum {
    LhsCoeffReadCost = _LhsNested::CoeffReadCost,
    RhsCoeffReadCost = _RhsNested::CoeffReadCost,
    LhsFlags = _LhsNested::Flags,
    RhsFlags = _RhsNested::Flags,

    RowsAtCompileTime = _LhsNested::RowsAtCompileTime,
    ColsAtCompileTime = _RhsNested::ColsAtCompileTime,
    InnerSize = EIGEN_ENUM_MIN(_LhsNested::ColsAtCompileTime, _RhsNested::RowsAtCompileTime),

    MaxRowsAtCompileTime = _LhsNested::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = _RhsNested::MaxColsAtCompileTime,

    LhsRowMajor = LhsFlags & RowMajorBit,
    RhsRowMajor = RhsFlags & RowMajorBit,

    CanVectorizeRhs = RhsRowMajor && (RhsFlags & PacketAccessBit)
                    && (ColsAtCompileTime % ei_packet_traits<Scalar>::size == 0),

    CanVectorizeLhs = (!LhsRowMajor) && (LhsFlags & PacketAccessBit)
                    && (RowsAtCompileTime % ei_packet_traits<Scalar>::size == 0),

    CanVectorizeInner = LhsRowMajor && (!RhsRowMajor) && (LhsFlags & PacketAccessBit) && (RhsFlags & PacketAccessBit)
                      && (InnerSize!=Dynamic) && (InnerSize % ei_packet_traits<Scalar>::size == 0),

    EvalToRowMajor = RhsRowMajor && (ProductMode==(int)CacheFriendlyProduct ? LhsRowMajor : (!CanVectorizeLhs)),

    RemovedBits = ~((EvalToRowMajor ? 0 : RowMajorBit)
                | ((RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic) ? 0 : LargeBit)),

    Flags = ((unsigned int)(LhsFlags | RhsFlags) & HereditaryBits & RemovedBits)
          | EvalBeforeAssigningBit
          | EvalBeforeNestingBit
          | (CanVectorizeLhs || CanVectorizeRhs ? PacketAccessBit : 0),

    CoeffReadCost = InnerSize == Dynamic ? Dynamic
                  : InnerSize * (NumTraits<Scalar>::MulCost + LhsCoeffReadCost + RhsCoeffReadCost)
                    + (InnerSize - 1) * NumTraits<Scalar>::AddCost
  };
};

template<typename LhsNested, typename RhsNested, int ProductMode> class Product : ei_no_assignment_operator,
  public MatrixBase<Product<LhsNested, RhsNested, ProductMode> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Product)

  private:

    typedef typename ei_traits<Product>::_LhsNested _LhsNested;
    typedef typename ei_traits<Product>::_RhsNested _RhsNested;

    enum {
      PacketSize = ei_packet_traits<Scalar>::size,
      InnerSize  = ei_traits<Product>::InnerSize,
      Unroll = CoeffReadCost <= EIGEN_UNROLLING_LIMIT,
      CanVectorizeInner = ei_traits<Product>::CanVectorizeInner && Unroll
    };

    typedef ei_product_coeff_impl<CanVectorizeInner ? InnerVectorization : NoVectorization,
                                  Unroll ? InnerSize-1 : Dynamic,
                                  _LhsNested, _RhsNested> ScalarCoeffImpl;

  public:

    template<typename Lhs, typename Rhs>
    inline Product(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      ei_assert(lhs.cols() == rhs.rows());
    }

    /** \internal
      * compute \a res += \c *this using the cache friendly product.
      */
    template<typename DestDerived>
    void _cacheFriendlyEvalAndAdd(DestDerived& res) const;

    /** \internal
      * \returns whether it is worth it to use the cache friendly product.
      */
    inline bool _useCacheFriendlyProduct() const {
      return   rows()>=EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
            && cols()>=EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
            && m_lhs.cols()>=EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD;
    }

    inline int rows() const { return m_lhs.rows(); }
    inline int cols() const { return m_rhs.cols(); }

    const Scalar coeff(int row, int col) const
    {
      Scalar res;
      ScalarCoeffImpl::run(row, col, m_lhs, m_rhs, res);
      return res;
    }

    /* Allow index-based non-packet access. It is impossible though to allow index-based packed access,
     * which is why we don't set the LinearAccessBit.
     */
    const Scalar coeff(int index) const
    {
      Scalar res;
      const int row = RowsAtCompileTime == 1 ? 0 : index;
      const int col = RowsAtCompileTime == 1 ? index : 0;
      ScalarCoeffImpl::run(row, col, m_lhs, m_rhs, res);
      return res;
    }

    template<int LoadMode>
    const PacketScalar packet(int row, int col) const
    {
      PacketScalar res;
      ei_product_packet_impl<Flags&RowMajorBit ? RowMajor : ColMajor,
                                   Unroll ? InnerSize-1 : Dynamic,
                                   _LhsNested, _RhsNested, PacketScalar, LoadMode>
        ::run(row, col, m_lhs, m_rhs, res);
      return res;
    }

  protected:
    const LhsNested m_lhs;
    const RhsNested m_rhs;
};

/** \returns the matrix product of \c *this and \a other.
  *
  * \sa lazy(), operator*=(const MatrixBase&)
  */
template<typename Derived>
template<typename OtherDerived>
inline const typename ProductReturnType<Derived,OtherDerived>::Type
MatrixBase<Derived>::operator*(const MatrixBase<OtherDerived> &other) const
{
  return typename ProductReturnType<Derived,OtherDerived>::Type(derived(), other.derived());
}

/** replaces \c *this by \c *this * \a other.
  *
  * \returns a reference to \c *this
  */
template<typename Derived>
template<typename OtherDerived>
inline Derived &
MatrixBase<Derived>::operator*=(const MatrixBase<OtherDerived> &other)
{
  return *this = *this * other;
}

/***************************************************************************
* Normal product .coeff() implementation (with meta-unrolling)
***************************************************************************/

/**************************************
*** Scalar path  - no vectorization ***
**************************************/

template<int Index, typename Lhs, typename Rhs>
struct ei_product_coeff_impl<NoVectorization, Index, Lhs, Rhs>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    ei_product_coeff_impl<NoVectorization, Index-1, Lhs, Rhs>::run(row, col, lhs, rhs, res);
    res += lhs.coeff(row, Index) * rhs.coeff(Index, col);
  }
};

template<typename Lhs, typename Rhs>
struct ei_product_coeff_impl<NoVectorization, 0, Lhs, Rhs>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
  }
};

template<typename Lhs, typename Rhs>
struct ei_product_coeff_impl<NoVectorization, Dynamic, Lhs, Rhs>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar& res)
  {
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
      for(int i = 1; i < lhs.cols(); i++)
        res += lhs.coeff(row, i) * rhs.coeff(i, col);
  }
};

// prevent buggy user code from causing an infinite recursion
template<typename Lhs, typename Rhs>
struct ei_product_coeff_impl<NoVectorization, -1, Lhs, Rhs>
{
  inline static void run(int, int, const Lhs&, const Rhs&, typename Lhs::Scalar&) {}
};

/*******************************************
*** Scalar path with inner vectorization ***
*******************************************/

template<int Index, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_product_coeff_vectorized_impl
{
  enum { PacketSize = ei_packet_traits<typename Lhs::Scalar>::size };
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, typename Lhs::PacketScalar &pres)
  {
    ei_product_coeff_vectorized_impl<Index-PacketSize, Lhs, Rhs, PacketScalar>::run(row, col, lhs, rhs, pres);
    pres = ei_padd(pres, ei_pmul( lhs.template packet<Aligned>(row, Index) , rhs.template packet<Aligned>(Index, col) ));
  }
};

template<typename Lhs, typename Rhs, typename PacketScalar>
struct ei_product_coeff_vectorized_impl<0, Lhs, Rhs, PacketScalar>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, typename Lhs::PacketScalar &pres)
  {
    pres = ei_pmul(lhs.template packet<Aligned>(row, 0) , rhs.template packet<Aligned>(0, col));
  }
};

template<int Index, typename Lhs, typename Rhs>
struct ei_product_coeff_impl<InnerVectorization, Index, Lhs, Rhs>
{
  typedef typename Lhs::PacketScalar PacketScalar;
  enum { PacketSize = ei_packet_traits<typename Lhs::Scalar>::size };
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    PacketScalar pres;
    ei_product_coeff_vectorized_impl<Index+1-PacketSize, Lhs, Rhs, PacketScalar>::run(row, col, lhs, rhs, pres);
    ei_product_coeff_impl<NoVectorization,Index,Lhs,Rhs>::run(row, col, lhs, rhs, res);
    res = ei_predux(pres);
  }
};

/*******************
*** Packet path  ***
*******************/

template<int Index, typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl<RowMajor, Index, Lhs, Rhs, PacketScalar, LoadMode>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    ei_product_packet_impl<RowMajor, Index-1, Lhs, Rhs, PacketScalar, LoadMode>::run(row, col, lhs, rhs, res);
    res =  ei_pmadd(ei_pset1(lhs.coeff(row, Index)), rhs.template packet<LoadMode>(Index, col), res);
  }
};

template<int Index, typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl<ColMajor, Index, Lhs, Rhs, PacketScalar, LoadMode>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    ei_product_packet_impl<ColMajor, Index-1, Lhs, Rhs, PacketScalar, LoadMode>::run(row, col, lhs, rhs, res);
    res =  ei_pmadd(lhs.template packet<LoadMode>(row, Index), ei_pset1(rhs.coeff(Index, col)), res);
  }
};

template<typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl<RowMajor, 0, Lhs, Rhs, PacketScalar, LoadMode>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    res = ei_pmul(ei_pset1(lhs.coeff(row, 0)),rhs.template packet<LoadMode>(0, col));
  }
};

template<typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl<ColMajor, 0, Lhs, Rhs, PacketScalar, LoadMode>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    res = ei_pmul(lhs.template packet<LoadMode>(row, 0), ei_pset1(rhs.coeff(0, col)));
  }
};

template<typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl<RowMajor, Dynamic, Lhs, Rhs, PacketScalar, LoadMode>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar& res)
  {
    res = ei_pmul(ei_pset1(lhs.coeff(row, 0)),rhs.template packet<LoadMode>(0, col));
      for(int i = 1; i < lhs.cols(); i++)
        res =  ei_pmadd(ei_pset1(lhs.coeff(row, i)), rhs.template packet<LoadMode>(i, col), res);
  }
};

template<typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl<ColMajor, Dynamic, Lhs, Rhs, PacketScalar, LoadMode>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar& res)
  {
    res = ei_pmul(lhs.template packet<LoadMode>(row, 0), ei_pset1(rhs.coeff(0, col)));
      for(int i = 1; i < lhs.cols(); i++)
        res =  ei_pmadd(lhs.template packet<LoadMode>(row, i), ei_pset1(rhs.coeff(i, col)), res);
  }
};

/***************************************************************************
* Cache friendly product callers and specific nested evaluation strategies
***************************************************************************/

/** \internal */
template<typename Derived>
template<typename Lhs,typename Rhs>
inline Derived&
MatrixBase<Derived>::operator+=(const Flagged<Product<Lhs,Rhs,CacheFriendlyProduct>, 0, EvalBeforeNestingBit | EvalBeforeAssigningBit>& other)
{
  if (other._expression()._useCacheFriendlyProduct())
    other._expression()._cacheFriendlyEvalAndAdd(const_cast_derived());
  else
    lazyAssign(derived() + other._expression());
  return derived();
}

template<typename Derived>
template<typename Lhs, typename Rhs>
inline Derived& MatrixBase<Derived>::lazyAssign(const Product<Lhs,Rhs,CacheFriendlyProduct>& product)
{
  if (product._useCacheFriendlyProduct())
  {
    setZero();
    product._cacheFriendlyEvalAndAdd(derived());
  }
  else
  {
    lazyAssign<Product<Lhs,Rhs,CacheFriendlyProduct> >(product);
  }
  return derived();
}

template<typename T> class ei_product_eval_to_column_major
{
    typedef typename ei_traits<T>::Scalar _Scalar;
    enum {
          _Rows = ei_traits<T>::RowsAtCompileTime,
          _Cols = ei_traits<T>::ColsAtCompileTime,
          _MaxRows = ei_traits<T>::MaxRowsAtCompileTime,
          _MaxCols = ei_traits<T>::MaxColsAtCompileTime,
          _Flags = ei_traits<T>::Flags
    };

  public:
    typedef Matrix<_Scalar,
                  _Rows, _Cols, _MaxRows, _MaxCols,
                  ei_corrected_matrix_flags<
                      _Scalar,
                      _Rows, _Cols, _MaxRows, _MaxCols,
                      _Flags
                  >::ret & ~RowMajorBit
            > type;
};

template<typename T> struct ei_product_copy_rhs
{
  typedef typename ei_meta_if<
         (ei_traits<T>::Flags & RowMajorBit)
      || (!(ei_traits<T>::Flags & DirectAccessBit)),
      typename ei_product_eval_to_column_major<T>::type,
      const T&
    >::ret type;
};

template<typename T> struct ei_product_copy_lhs
{
  typedef typename ei_meta_if<
      (!(int(ei_traits<T>::Flags) & DirectAccessBit)),
      typename ei_eval<T>::type,
      const T&
    >::ret type;
};

template<typename Lhs, typename Rhs, int ProductMode>
template<typename DestDerived>
inline void Product<Lhs,Rhs,ProductMode>::_cacheFriendlyEvalAndAdd(DestDerived& res) const
{
  typedef typename ei_product_copy_lhs<_LhsNested>::type LhsCopy;
  typedef typename ei_unref<LhsCopy>::type _LhsCopy;
  typedef typename ei_product_copy_rhs<_RhsNested>::type RhsCopy;
  typedef typename ei_unref<RhsCopy>::type _RhsCopy;
  LhsCopy lhs(m_lhs);
  RhsCopy rhs(m_rhs);
  ei_cache_friendly_product<Scalar>(
    rows(), cols(), lhs.cols(),
    _LhsCopy::Flags&RowMajorBit, &(lhs.const_cast_derived().coeffRef(0,0)), lhs.stride(),
    _RhsCopy::Flags&RowMajorBit, &(rhs.const_cast_derived().coeffRef(0,0)), rhs.stride(),
    Flags&RowMajorBit, &(res.coeffRef(0,0)), res.stride()
  );
}

#endif // EIGEN_PRODUCT_H
