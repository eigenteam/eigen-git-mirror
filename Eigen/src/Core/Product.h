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

template<int Index, int Size, typename Lhs, typename Rhs>
struct ei_product_impl
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs,
                               typename Lhs::Scalar &res)
  {
    ei_product_impl<Index-1, Size, Lhs, Rhs>::run(row, col, lhs, rhs, res);
    res += lhs.coeff(row, Index) * rhs.coeff(Index, col);
  }
};

template<int Size, typename Lhs, typename Rhs>
struct ei_product_impl<0, Size, Lhs, Rhs>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs,
                  typename Lhs::Scalar &res)
  {
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
  }
};

template<int Index, typename Lhs, typename Rhs>
struct ei_product_impl<Index, Dynamic, Lhs, Rhs>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar& res)
  {
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
      for(int i = 1; i < lhs.cols(); i++)
        res += lhs.coeff(row, i) * rhs.coeff(i, col);
  }
};

// prevent buggy user code from causing an infinite recursion
template<int Index, typename Lhs, typename Rhs>
struct ei_product_impl<Index, 0, Lhs, Rhs>
{
  inline static void run(int, int, const Lhs&, const Rhs&, typename Lhs::Scalar&) {}
};

//----------

template<bool RowMajor, int Index, int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_impl;

template<int Index, int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_impl<true, Index, Size, Lhs, Rhs, PacketScalar>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    ei_packet_product_impl<true, Index-1, Size, Lhs, Rhs, PacketScalar>::run(row, col, lhs, rhs, res);
    res =  ei_pmadd(ei_pset1(lhs.coeff(row, Index)), rhs.template packetCoeff<Aligned>(Index, col), res);
  }
};

template<int Index, int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_impl<false, Index, Size, Lhs, Rhs, PacketScalar>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    ei_packet_product_impl<false, Index-1, Size, Lhs, Rhs, PacketScalar>::run(row, col, lhs, rhs, res);
    res =  ei_pmadd(lhs.template packetCoeff<Aligned>(row, Index), ei_pset1(rhs.coeff(Index, col)), res);
  }
};

template<int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_impl<true, 0, Size, Lhs, Rhs, PacketScalar>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    res = ei_pmul(ei_pset1(lhs.coeff(row, 0)),rhs.template packetCoeff<Aligned>(0, col));
  }
};

template<int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_impl<false, 0, Size, Lhs, Rhs, PacketScalar>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    res = ei_pmul(lhs.template packetCoeff<Aligned>(row, 0), ei_pset1(rhs.coeff(0, col)));
  }
};

template<bool RowMajor, int Index, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_impl<RowMajor, Index, Dynamic, Lhs, Rhs, PacketScalar>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar& res)
  {
    res = ei_pmul(ei_pset1(lhs.coeff(row, 0)),rhs.template packetCoeff<Aligned>(0, col));
      for(int i = 1; i < lhs.cols(); i++)
        res =  ei_pmadd(ei_pset1(lhs.coeff(row, i)), rhs.template packetCoeff<Aligned>(i, col), res);
  }
};

template<int Index, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_impl<false, Index, Dynamic, Lhs, Rhs, PacketScalar>
{
  inline static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar& res)
  {
    res = ei_pmul(lhs.template packetCoeff<Aligned>(row, 0), ei_pset1(rhs.coeff(0, col)));
      for(int i = 1; i < lhs.cols(); i++)
        res =  ei_pmadd(lhs.template packetCoeff<Aligned>(row, i), ei_pset1(rhs.coeff(i, col)), res);
  }
};

/** \class Product
  *
  * \brief Expression of the product of two matrices
  *
  * \param Lhs the type of the left-hand side
  * \param Rhs the type of the right-hand side
  * \param EvalMode internal use only
  *
  * This class represents an expression of the product of two matrices.
  * It is the return type of the operator* between matrices, and most of the time
  * this is the only way it is used.
  */
template<typename Lhs, typename Rhs> struct ei_product_eval_mode
{
  enum{ value =  Lhs::MaxRowsAtCompileTime >= EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
              && Rhs::MaxColsAtCompileTime >= EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
              && Lhs::MaxColsAtCompileTime >= EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
              ? CacheFriendlyProduct : NormalProduct };
};

template<typename T> class ei_product_eval_to_column_major
{
    typedef typename ei_traits<T>::Scalar _Scalar;
    enum {_MaxRows = ei_traits<T>::MaxRowsAtCompileTime,
          _MaxCols = ei_traits<T>::MaxColsAtCompileTime,
          _Flags = ei_traits<T>::Flags
    };

  public:
    typedef Matrix<_Scalar,
                  ei_traits<T>::RowsAtCompileTime,
                  ei_traits<T>::ColsAtCompileTime,
                  ei_corrected_matrix_flags<_Scalar, ei_size_at_compile_time<_MaxRows,_MaxCols>::ret, _Flags>::ret & ~RowMajorBit,
                  ei_traits<T>::MaxRowsAtCompileTime,
                  ei_traits<T>::MaxColsAtCompileTime> type;
};

// as ei_nested, but evaluate to a column-major matrix if an evaluation is required
template<typename T, int n=1> struct ei_product_nested_rhs
{
  typedef typename ei_meta_if<
    ei_must_nest_by_value<T>::ret,
    T,
    typename ei_meta_if<
        ((ei_traits<T>::Flags & EvalBeforeNestingBit)
      || (n+1) * (NumTraits<typename ei_traits<T>::Scalar>::ReadCost) < (n-1) * T::CoeffReadCost),
      typename ei_product_eval_to_column_major<T>::type,
      const T&
    >::ret
  >::ret type;
};

template<typename Lhs, typename Rhs, int EvalMode>
struct ei_traits<Product<Lhs, Rhs, EvalMode> >
{
  typedef typename Lhs::Scalar Scalar;
  typedef typename ei_nested<Lhs,Rhs::ColsAtCompileTime>::type LhsNested;
  typedef typename ei_meta_if<EvalMode==CacheFriendlyProduct,
      typename ei_product_nested_rhs<Rhs,Lhs::RowsAtCompileTime>::type,
      typename ei_nested<Rhs,Lhs::RowsAtCompileTime>::type>::ret RhsNested;
  typedef typename ei_unconst<typename ei_unref<LhsNested>::type>::type _LhsNested;
  typedef typename ei_unconst<typename ei_unref<RhsNested>::type>::type _RhsNested;
  enum {
    LhsCoeffReadCost = _LhsNested::CoeffReadCost,
    RhsCoeffReadCost = _RhsNested::CoeffReadCost,
    LhsFlags = _LhsNested::Flags,
    RhsFlags = _RhsNested::Flags,
    RowsAtCompileTime = Lhs::RowsAtCompileTime,
    ColsAtCompileTime = Rhs::ColsAtCompileTime,
    MaxRowsAtCompileTime = Lhs::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = Rhs::MaxColsAtCompileTime,
    // the vectorization flags are only used by the normal product,
    // the other one is always vectorized !
    _RhsVectorizable = (RhsFlags & RowMajorBit) && (RhsFlags & VectorizableBit) && (ColsAtCompileTime % ei_packet_traits<Scalar>::size == 0),
    _LhsVectorizable = (!(LhsFlags & RowMajorBit)) && (LhsFlags & VectorizableBit) && (RowsAtCompileTime % ei_packet_traits<Scalar>::size == 0),
    _Vectorizable = (_LhsVectorizable || _RhsVectorizable) ? 1 : 0,
    _RowMajor = (RhsFlags & RowMajorBit)
              && (EvalMode==(int)CacheFriendlyProduct ? (int)LhsFlags & RowMajorBit : (!_LhsVectorizable)),
    _LostBits = ~((_RowMajor ? 0 : RowMajorBit)
                | ((RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic) ? 0 : LargeBit)),
    Flags = ((unsigned int)(LhsFlags | RhsFlags) & HereditaryBits & _LostBits)
          | EvalBeforeAssigningBit
          | EvalBeforeNestingBit
          | (_Vectorizable ? VectorizableBit : 0),
    CoeffReadCost
      = Lhs::ColsAtCompileTime == Dynamic
      ? Dynamic
      : Lhs::ColsAtCompileTime
        * (NumTraits<Scalar>::MulCost + LhsCoeffReadCost + RhsCoeffReadCost)
        + (Lhs::ColsAtCompileTime - 1) * NumTraits<Scalar>::AddCost
  };
};

template<typename Lhs, typename Rhs, int EvalMode> class Product : ei_no_assignment_operator,
  public MatrixBase<Product<Lhs, Rhs, EvalMode> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Product)
    typedef typename ei_traits<Product>::LhsNested LhsNested;
    typedef typename ei_traits<Product>::RhsNested RhsNested;
    typedef typename ei_traits<Product>::_LhsNested _LhsNested;
    typedef typename ei_traits<Product>::_RhsNested _RhsNested;

    enum {
      PacketSize = ei_packet_traits<Scalar>::size
    };

    inline Product(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      ei_assert(lhs.cols() == rhs.rows());
    }

    /** \internal */
    template<typename DestDerived>
    void _cacheFriendlyEval(DestDerived& res) const;

    /** \internal */
    template<typename DestDerived>
    void _cacheFriendlyEvalAndAdd(DestDerived& res) const;

  private:

    inline int _rows() const { return m_lhs.rows(); }
    inline int _cols() const { return m_rhs.cols(); }

    const Scalar _coeff(int row, int col) const
    {
      Scalar res;
      const bool unroll = CoeffReadCost <= EIGEN_UNROLLING_LIMIT;
      ei_product_impl<Lhs::ColsAtCompileTime-1,
                          unroll ? Lhs::ColsAtCompileTime : Dynamic,
                          _LhsNested, _RhsNested>
        ::run(row, col, m_lhs, m_rhs, res);
      return res;
    }

    template<int LoadMode>
    const PacketScalar _packetCoeff(int row, int col) const
    {
      const bool unroll = CoeffReadCost <= EIGEN_UNROLLING_LIMIT;
      PacketScalar res;
      ei_packet_product_impl<Flags&RowMajorBit ? true : false, Lhs::ColsAtCompileTime-1,
                          unroll ? Lhs::ColsAtCompileTime : Dynamic,
                          _LhsNested, _RhsNested, PacketScalar>
        ::run(row, col, m_lhs, m_rhs, res);
      return res;
    }

    template<typename Lhs_, typename Rhs_, int EvalMode_, typename DestDerived_, bool DirectAccess_>
    friend struct ei_cache_friendly_selector;

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
inline const typename MatrixBase<Derived>::template ProductReturnType<OtherDerived>::Type
MatrixBase<Derived>::operator*(const MatrixBase<OtherDerived> &other) const
{
  assert( (Derived::Flags&ArrayBit) == (OtherDerived::Flags&ArrayBit) );
  return typename ProductReturnType<OtherDerived>::Type(derived(), other.derived());
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

/** \internal */
template<typename Derived>
template<typename Lhs,typename Rhs>
inline Derived&
MatrixBase<Derived>::operator+=(const Flagged<Product<Lhs,Rhs,CacheFriendlyProduct>, 0, EvalBeforeNestingBit | EvalBeforeAssigningBit>& other)
{
  other._expression()._cacheFriendlyEvalAndAdd(const_cast_derived());
  return derived();
}

template<typename Derived>
template<typename Lhs, typename Rhs>
inline Derived& MatrixBase<Derived>::lazyAssign(const Product<Lhs,Rhs,CacheFriendlyProduct>& product)
{
  product._cacheFriendlyEval(derived());
  return derived();
}

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

template<typename Lhs, typename Rhs, int EvalMode>
template<typename DestDerived>
inline void Product<Lhs,Rhs,EvalMode>::_cacheFriendlyEval(DestDerived& res) const
{
    if ( _rows()>=EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
      && _cols()>=EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
      && m_lhs.cols()>=EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
    )
    {
      res.setZero();
      typedef typename ei_product_copy_lhs<_LhsNested>::type LhsCopy;
      typedef typename ei_unref<LhsCopy>::type _LhsCopy;
      typedef typename ei_product_copy_rhs<_RhsNested>::type RhsCopy;
      typedef typename ei_unref<RhsCopy>::type _RhsCopy;
      LhsCopy lhs(m_lhs);
      RhsCopy rhs(m_rhs);
      ei_cache_friendly_product<Scalar>(
        _rows(), _cols(), lhs.cols(),
        _LhsCopy::Flags&RowMajorBit, &(lhs.const_cast_derived().coeffRef(0,0)), lhs.stride(),
        _RhsCopy::Flags&RowMajorBit, &(rhs.const_cast_derived().coeffRef(0,0)), rhs.stride(),
        Flags&RowMajorBit, &(res.coeffRef(0,0)), res.stride()
      );
    }
    else
    {
      res = Product<_LhsNested,_RhsNested,NormalProduct>(m_lhs, m_rhs).lazy();
    }
}

template<typename Lhs, typename Rhs, int EvalMode>
template<typename DestDerived>
inline void Product<Lhs,Rhs,EvalMode>::_cacheFriendlyEvalAndAdd(DestDerived& res) const
{
    if ( _rows()>=EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
      && _cols()>=EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
      && m_lhs.cols()>=EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
    )
    {
      typedef typename ei_product_copy_lhs<_LhsNested>::type LhsCopy;
      typedef typename ei_unref<LhsCopy>::type _LhsCopy;
      typedef typename ei_product_copy_rhs<_RhsNested>::type RhsCopy;
      typedef typename ei_unref<RhsCopy>::type _RhsCopy;
      LhsCopy lhs(m_lhs);
      RhsCopy rhs(m_rhs);
      ei_cache_friendly_product<Scalar>(
        _rows(), _cols(), lhs.cols(),
        _LhsCopy::Flags&RowMajorBit, &(lhs.const_cast_derived().coeffRef(0,0)), lhs.stride(),
        _RhsCopy::Flags&RowMajorBit, &(rhs.const_cast_derived().coeffRef(0,0)), rhs.stride(),
        Flags&RowMajorBit, &(res.coeffRef(0,0)), res.stride()
      );
    }
    else
    {
      res += Product<_LhsNested,_RhsNested,NormalProduct>(m_lhs, m_rhs).lazy();
    }
}

#endif // EIGEN_PRODUCT_H
