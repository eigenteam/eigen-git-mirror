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
struct ei_product_unroller
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs,
                  typename Lhs::Scalar &res)
  {
    ei_product_unroller<Index-1, Size, Lhs, Rhs>::run(row, col, lhs, rhs, res);
    res += lhs.coeff(row, Index) * rhs.coeff(Index, col);
  }
};

template<int Size, typename Lhs, typename Rhs>
struct ei_product_unroller<0, Size, Lhs, Rhs>
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs,
                  typename Lhs::Scalar &res)
  {
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
  }
};

template<int Index, typename Lhs, typename Rhs>
struct ei_product_unroller<Index, Dynamic, Lhs, Rhs>
{
  static void run(int, int, const Lhs&, const Rhs&, typename Lhs::Scalar&) {}
};

// prevent buggy user code from causing an infinite recursion
template<int Index, typename Lhs, typename Rhs>
struct ei_product_unroller<Index, 0, Lhs, Rhs>
{
  static void run(int, int, const Lhs&, const Rhs&, typename Lhs::Scalar&) {}
};

template<bool RowMajor, int Index, int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller;

template<int Index, int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<true, Index, Size, Lhs, Rhs, PacketScalar>
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    ei_packet_product_unroller<true, Index-1, Size, Lhs, Rhs, PacketScalar>::run(row, col, lhs, rhs, res);
    res =  ei_pmadd(ei_pset1(lhs.coeff(row, Index)), rhs.packetCoeff(Index, col), res);
  }
};

template<int Index, int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<false, Index, Size, Lhs, Rhs, PacketScalar>
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    ei_packet_product_unroller<false, Index-1, Size, Lhs, Rhs, PacketScalar>::run(row, col, lhs, rhs, res);
    res =  ei_pmadd(lhs.packetCoeff(row, Index), ei_pset1(rhs.coeff(Index, col)), res);
  }
};

template<int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<true, 0, Size, Lhs, Rhs, PacketScalar>
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    res = ei_pmul(ei_pset1(lhs.coeff(row, 0)),rhs.packetCoeff(0, col));
  }
};

template<int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<false, 0, Size, Lhs, Rhs, PacketScalar>
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    res = ei_pmul(lhs.packetCoeff(row, 0), ei_pset1(rhs.coeff(0, col)));
  }
};

template<bool RowMajor, int Index, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<RowMajor, Index, Dynamic, Lhs, Rhs, PacketScalar>
{
  static void run(int, int, const Lhs&, const Rhs&, PacketScalar&) {}
};

template<typename Product, bool RowMajor = true> struct ProductPacketCoeffImpl {
  inline static typename Product::PacketScalar execute(const Product& product, int row, int col)
  { return product._packetCoeffRowMajor(row,col); }
};

template<typename Product> struct ProductPacketCoeffImpl<Product, false> {
  inline static typename Product::PacketScalar execute(const Product& product, int row, int col)
  { return product._packetCoeffColumnMajor(row,col); }
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
  *
  * \sa class Sum, class Difference
  */
template<typename Lhs, typename Rhs> struct ei_product_eval_mode
{
  enum{ value =  Lhs::MaxRowsAtCompileTime >= 16
              && Rhs::MaxColsAtCompileTime >= 16
              && (!( (Lhs::Flags&RowMajorBit) && (Rhs::Flags&RowMajorBit xor RowMajorBit)))
              ? CacheOptimalProduct : NormalProduct };
};

template<typename Lhs, typename Rhs, int EvalMode>
struct ei_traits<Product<Lhs, Rhs, EvalMode> >
{
  typedef typename Lhs::Scalar Scalar;
  typedef typename ei_nested<Lhs,Rhs::ColsAtCompileTime>::type LhsNested;
  typedef typename ei_nested<Rhs,Lhs::RowsAtCompileTime>::type RhsNested;
  typedef typename ei_unref<LhsNested>::type _LhsNested;
  typedef typename ei_unref<RhsNested>::type _RhsNested;
  enum {
    LhsCoeffReadCost = _LhsNested::CoeffReadCost,
    RhsCoeffReadCost = _RhsNested::CoeffReadCost,
    LhsFlags = _LhsNested::Flags,
    RhsFlags = _RhsNested::Flags,
    RowsAtCompileTime = Lhs::RowsAtCompileTime,
    ColsAtCompileTime = Rhs::ColsAtCompileTime,
    MaxRowsAtCompileTime = Lhs::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = Rhs::MaxColsAtCompileTime,
    _RhsVectorizable = (RhsFlags & RowMajorBit) && (RhsFlags & VectorizableBit) && (ColsAtCompileTime % ei_packet_traits<Scalar>::size == 0),
    _LhsVectorizable = (!(LhsFlags & RowMajorBit)) && (LhsFlags & VectorizableBit) && (RowsAtCompileTime % ei_packet_traits<Scalar>::size == 0),
    _Vectorizable = (_LhsVectorizable || _RhsVectorizable) ? 1 : 0,
    _RowMajor = (RhsFlags & RowMajorBit)
              && (EvalMode==(int)CacheOptimalProduct ? (int)LhsFlags & RowMajorBit : (!_LhsVectorizable)),
    _LostBits = DefaultLostFlagMask & ~(
                (_RowMajor ? 0 : RowMajorBit)
              | ((RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic) ? 0 : LargeBit)),
    Flags = ((unsigned int)(LhsFlags | RhsFlags) & _LostBits)
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
    friend class ProductPacketCoeffImpl<Product,Flags&RowMajorBit>;
    typedef typename ei_traits<Product>::LhsNested LhsNested;
    typedef typename ei_traits<Product>::RhsNested RhsNested;
    typedef typename ei_traits<Product>::_LhsNested _LhsNested;
    typedef typename ei_traits<Product>::_RhsNested _RhsNested;

    Product(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      ei_assert(lhs.cols() == rhs.rows());
    }

    /** \internal */
    template<typename DestDerived>
    void _cacheOptimalEval(DestDerived& res, ei_meta_false) const;
    #ifdef EIGEN_VECTORIZE
    template<typename DestDerived>
    void _cacheOptimalEval(DestDerived& res, ei_meta_true) const;
    #endif

  private:

    int _rows() const { return m_lhs.rows(); }
    int _cols() const { return m_rhs.cols(); }

    const Scalar _coeff(int row, int col) const
    {
      Scalar res;
      const bool unroll = CoeffReadCost <= EIGEN_UNROLLING_LIMIT;
      if(unroll)
      {
        ei_product_unroller<Lhs::ColsAtCompileTime-1,
                            unroll ? Lhs::ColsAtCompileTime : Dynamic,
                            _LhsNested, _RhsNested>
          ::run(row, col, m_lhs, m_rhs, res);
      }
      else
      {
        res = m_lhs.coeff(row, 0) * m_rhs.coeff(0, col);
        for(int i = 1; i < m_lhs.cols(); i++)
          res += m_lhs.coeff(row, i) * m_rhs.coeff(i, col);
      }
      return res;
    }

    PacketScalar _packetCoeff(int row, int col) const
    {
      if(Lhs::ColsAtCompileTime <= EIGEN_UNROLLING_LIMIT)
      {
        PacketScalar res;
        ei_packet_product_unroller<Flags&RowMajorBit, Lhs::ColsAtCompileTime-1,
                            Lhs::ColsAtCompileTime <= EIGEN_UNROLLING_LIMIT
                              ? Lhs::ColsAtCompileTime : Dynamic,
                            _LhsNested, _RhsNested, PacketScalar>
          ::run(row, col, m_lhs, m_rhs, res);
        return res;
      }
      else
        return ProductPacketCoeffImpl<Product,Flags&RowMajorBit>::execute(*this, row, col);
    }

    PacketScalar _packetCoeffRowMajor(int row, int col) const
    {
      PacketScalar res;
      res = ei_pmul(ei_pset1(m_lhs.coeff(row, 0)),m_rhs.packetCoeff(0, col));
      for(int i = 1; i < m_lhs.cols(); i++)
        res =  ei_pmadd(ei_pset1(m_lhs.coeff(row, i)), m_rhs.packetCoeff(i, col), res);
      return res;
    }

    PacketScalar _packetCoeffColumnMajor(int row, int col) const
    {
      PacketScalar res;
      res = ei_pmul(m_lhs.packetCoeff(row, 0), ei_pset1(m_rhs.coeff(0, col)));
      for(int i = 1; i < m_lhs.cols(); i++)
        res =  ei_pmadd(m_lhs.packetCoeff(row, i), ei_pset1(m_rhs.coeff(i, col)), res);
      return res;
    }

  protected:
    const LhsNested m_lhs;
    const RhsNested m_rhs;
};

/** \returns the matrix product of \c *this and \a other.
  *
  * \note This function causes an immediate evaluation. If you want to perform a matrix product
  * without immediate evaluation, call .lazy() on one of the matrices before taking the product.
  *
  * \sa lazy(), operator*=(const MatrixBase&)
  */
template<typename Derived>
template<typename OtherDerived>
const Product<Derived,OtherDerived>
MatrixBase<Derived>::operator*(const MatrixBase<OtherDerived> &other) const
{
  return Product<Derived,OtherDerived>(derived(), other.derived());
}

/** replaces \c *this by \c *this * \a other.
  *
  * \returns a reference to \c *this
  */
template<typename Derived>
template<typename OtherDerived>
Derived &
MatrixBase<Derived>::operator*=(const MatrixBase<OtherDerived> &other)
{
  return *this = *this * other;
}

template<typename Derived>
template<typename Lhs, typename Rhs>
Derived& MatrixBase<Derived>::lazyAssign(const Product<Lhs,Rhs,CacheOptimalProduct>& product)
{
  product._cacheOptimalEval(*this,
    #ifdef EIGEN_VECTORIZE
    typename ei_meta_if<Flags & VectorizableBit, ei_meta_true, ei_meta_false>::ret()
    #else
    ei_meta_false()
    #endif
    );
  return derived();
}

template<typename Lhs, typename Rhs, int EvalMode>
template<typename DestDerived>
void Product<Lhs,Rhs,EvalMode>::_cacheOptimalEval(DestDerived& res, ei_meta_false) const
{
  res.setZero();
  const int cols4 = m_lhs.cols() & 0xfffffffC;
  if (Lhs::Flags&RowMajorBit)
  {
//     std::cout << "opt rhs\n";
    int j=0;
    for(; j<cols4; j+=4)
    {
      for(int k=0; k<this->rows(); ++k)
      {
        const Scalar tmp0 = m_lhs.coeff(k,j  );
        const Scalar tmp1 = m_lhs.coeff(k,j+1);
        const Scalar tmp2 = m_lhs.coeff(k,j+2);
        const Scalar tmp3 = m_lhs.coeff(k,j+3);
        for (int i=0; i<this->cols(); ++i)
          res.coeffRef(k,i) += tmp0 * m_rhs.coeff(j+0,i) + tmp1 * m_rhs.coeff(j+1,i)
                             + tmp2 * m_rhs.coeff(j+2,i) + tmp3 * m_rhs.coeff(j+3,i);
      }
    }
    for(; j<m_lhs.cols(); ++j)
    {
      for(int k=0; k<this->rows(); ++k)
      {
        const Scalar tmp = m_rhs.coeff(k,j);
        for (int i=0; i<this->cols(); ++i)
          res.coeffRef(k,i) += tmp * m_lhs.coeff(j,i);
      }
    }
  }
  else
  {
//     std::cout << "opt lhs\n";
    int j = 0;
    for(; j<cols4; j+=4)
    {
      for(int k=0; k<this->cols(); ++k)
      {
        const Scalar tmp0 = m_rhs.coeff(j  ,k);
        const Scalar tmp1 = m_rhs.coeff(j+1,k);
        const Scalar tmp2 = m_rhs.coeff(j+2,k);
        const Scalar tmp3 = m_rhs.coeff(j+3,k);
        for (int i=0; i<this->rows(); ++i)
          res.coeffRef(i,k) += tmp0 * m_lhs.coeff(i,j+0) + tmp1 * m_lhs.coeff(i,j+1)
                             + tmp2 * m_lhs.coeff(i,j+2) + tmp3 * m_lhs.coeff(i,j+3);
      }
    }
    for(; j<m_lhs.cols(); ++j)
    {
      for(int k=0; k<this->cols(); ++k)
      {
        const Scalar tmp = m_rhs.coeff(j,k);
        for (int i=0; i<this->rows(); ++i)
          res.coeffRef(i,k) += tmp * m_lhs.coeff(i,j);
      }
    }
  }
}

#ifdef EIGEN_VECTORIZE
template<typename Lhs, typename Rhs, int EvalMode>
template<typename DestDerived>
void Product<Lhs,Rhs,EvalMode>::_cacheOptimalEval(DestDerived& res, ei_meta_true) const
{

  if (((Lhs::Flags&RowMajorBit) && (_cols() % ei_packet_traits<Scalar>::size != 0))
    || (_rows() % ei_packet_traits<Scalar>::size != 0))
  {
    return _cacheOptimalEval(res, ei_meta_false());
  }

  res.setZero();
  const int cols4 = m_lhs.cols() & 0xfffffffC;
  if (Lhs::Flags&RowMajorBit)
  {
//     std::cout << "packet rhs\n";
    int j=0;
    for(; j<cols4; j+=4)
    {
      for(int k=0; k<this->rows(); k++)
      {
        const typename ei_packet_traits<Scalar>::type tmp0 = ei_pset1(m_lhs.coeff(k,j+0));
        const typename ei_packet_traits<Scalar>::type tmp1 = ei_pset1(m_lhs.coeff(k,j+1));
        const typename ei_packet_traits<Scalar>::type tmp2 = ei_pset1(m_lhs.coeff(k,j+2));
        const typename ei_packet_traits<Scalar>::type tmp3 = ei_pset1(m_lhs.coeff(k,j+3));
        for (int i=0; i<this->cols(); i+=ei_packet_traits<Scalar>::size)
        {
          // FIXME the following could be implemented using only mul-add, check if this is still OK for SSE
          res.writePacketCoeff(k,i,
            ei_padd(
              res.packetCoeff(k,i),
              ei_padd(
                ei_pmadd(tmp0, m_rhs.packetCoeff(j+0,i), ei_pmul(tmp1, m_rhs.packetCoeff(j+1,i))),
                ei_pmadd(tmp2, m_rhs.packetCoeff(j+2,i), ei_pmul(tmp3, m_rhs.packetCoeff(j+3,i)))
              )
            )
          );
        }
      }
    }
    for(; j<m_lhs.cols(); ++j)
    {
      for(int k=0; k<this->rows(); k++)
      {
        const typename ei_packet_traits<Scalar>::type tmp = ei_pset1(m_lhs.coeff(k,j));
        for (int i=0; i<this->cols(); i+=ei_packet_traits<Scalar>::size)
          res.writePacketCoeff(k,i, ei_pmadd(tmp, m_rhs.packetCoeff(j,i), res.packetCoeff(k,i)));
      }
    }
  }
  else
  {
//     std::cout << "packet lhs\n";
    int j=0;
    for(; j<cols4; j+=4)
    {
      for(int k=0; k<this->cols(); k++)
      {
        const typename ei_packet_traits<Scalar>::type tmp0 = ei_pset1(m_rhs.coeff(j+0,k));
        const typename ei_packet_traits<Scalar>::type tmp1 = ei_pset1(m_rhs.coeff(j+1,k));
        const typename ei_packet_traits<Scalar>::type tmp2 = ei_pset1(m_rhs.coeff(j+2,k));
        const typename ei_packet_traits<Scalar>::type tmp3 = ei_pset1(m_rhs.coeff(j+3,k));
        for (int i=0; i<this->rows(); i+=ei_packet_traits<Scalar>::size)
        {
          res.writePacketCoeff(i,k,
            ei_padd(
              res.packetCoeff(i,k),
              ei_padd(
                ei_pmadd(tmp0, m_lhs.packetCoeff(i,j),  ei_pmul(tmp1, m_lhs.packetCoeff(i,j+1))),
                ei_pmadd(tmp2, m_lhs.packetCoeff(i,j+2),ei_pmul(tmp3, m_lhs.packetCoeff(i,j+3)))

              )
            )
          );
        }
      }
    }
    for(; j<m_lhs.cols(); ++j)
    {
      for(int k=0; k<this->cols(); k++)
      {
        const typename ei_packet_traits<Scalar>::type tmp = ei_pset1(m_rhs.coeff(j,k));
        for (int i=0; i<this->rows(); i+=ei_packet_traits<Scalar>::size)
          res.writePacketCoeff(i,k, ei_pmadd(tmp, m_lhs.packetCoeff(i,j), res.packetCoeff(i,k)));
      }
    }
  }
}
#endif // EIGEN_VECTORIZE

#endif // EIGEN_PRODUCT_H
