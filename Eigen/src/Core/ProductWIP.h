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

#ifndef EIGEN_VECTORIZE
#error you must enable vectorization to try this experimental product implementation
#endif

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
  enum{ value =  Lhs::MaxRowsAtCompileTime >= EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
              && Rhs::MaxColsAtCompileTime >= EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
              && (!( (Lhs::Flags&RowMajorBit) && ((Rhs::Flags&RowMajorBit) ^ RowMajorBit)))
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
//          | EvalBeforeAssigningBit //FIXME
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
    template<typename DestDerived> void _cacheFriendlyEval(DestDerived& res) const;

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

    template<int LoadMode>
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
  product._cacheFriendlyEval(*this);
  return derived();
}

template<typename Lhs, typename Rhs, int EvalMode>
template<typename DestDerived>
void Product<Lhs,Rhs,EvalMode>::_cacheFriendlyEval(DestDerived& res) const
{
  // allow direct access to data for benchmark purpose
  const Scalar* __restrict__ a = m_lhs.derived().data();
  const Scalar* __restrict__ b = m_rhs.derived().data();
  Scalar* __restrict__ c = res.derived().data();

  // FIXME find a way to optimize: (an_xpr) + (a * b)
  // then we don't need to clear res and avoid and additional mat-mat sum
//   res.setZero();

  const int ps = ei_packet_traits<Scalar>::size;  // size of a packet
  #if (defined __i386__)
  // i386 architectures provides only 8 xmmm register,
  // so let's reduce the max number of rows processed at once
  const int bw = 4;                               // number of rows treated at once
  #else
  const int bw = 8;                               // number of rows treated at once
  #endif
  const int bs = ps * bw;                         // total number of elements treated at once
  const int rows = _rows();
  const int cols = _cols();
  const int size = m_lhs.cols();                  // third dimension of the product
  const int l2blocksize = 256 > _cols() ? _cols() : 256;
  const bool rhsIsAligned = ((size%ps) == 0);
  const bool resIsAligned = ((cols%ps) == 0);
  Scalar* __restrict__ block = new Scalar[l2blocksize*size];

  // loops on each L2 cache friendly blocks of the result
  for(int l2i=0; l2i<_rows(); l2i+=l2blocksize)
  {
    const int l2blockRowEnd = std::min(l2i+l2blocksize, rows);
    const int l2blockRowEndBW = l2blockRowEnd & 0xFFFFF8;             // end of the rows aligned to bw
    const int l2blockRowRemaining = l2blockRowEnd - l2blockRowEndBW;  // number of remaining rows

    // build a cache friendly block
    int count = 0;

    // copy l2blocksize rows of m_lhs to blocks of ps x bw
    for(int l2k=0; l2k<size; l2k+=l2blocksize)
    {
      const int l2blockSizeEnd = std::min(l2k+l2blocksize, size);

      for (int i = l2i; i<l2blockRowEndBW; i+=bw)
      {
        for (int k=l2k; k<l2blockSizeEnd; k+=ps)
        {
          // TODO write these two loops using meta unrolling
          // negligible for large matrices but useful for small ones
          for (int w=0; w<bw; ++w)
            for (int s=0; s<ps; ++s)
              block[count++] = m_lhs.coeff(i+w,k+s);
        }
      }
      if (l2blockRowRemaining>0)
      {
        for (int k=l2k; k<l2blockSizeEnd; k+=ps)
        {
          for (int w=0; w<l2blockRowRemaining; ++w)
            for (int s=0; s<ps; ++s)
              block[count++] = m_lhs.coeff(l2blockRowEndBW+w,k+s);
        }
      }
    }

    for(int l2j=0; l2j<cols; l2j+=l2blocksize)
    {
      int l2blockColEnd = std::min(l2j+l2blocksize, cols);

      for(int l2k=0; l2k<size; l2k+=l2blocksize)
      {
        // acumulate a full row of current a block time 4 cols of current a block
        // to a 1x4 c block
        int l2blockSizeEnd = std::min(l2k+l2blocksize, size);

        // for each 4x1 result's block sub blocks...
        for(int l1i=l2i; l1i<l2blockRowEndBW; l1i+=bw)
        {
          for(int l1j=l2j; l1j<l2blockColEnd; l1j+=1)
          {
            int offsetblock = l2k * (l2blockRowEnd-l2i) + (l1i-l2i)*(l2blockSizeEnd-l2k) - l2k*bw/*bs*/;
            const Scalar* localB = &block[offsetblock];

            int l1jsize = l1j * size; //TODO find a better way to optimize address computation ?

            PacketScalar dst[bw];
            dst[0] = ei_pset1(Scalar(0.));
            dst[1] = dst[0];
            dst[2] = dst[0];
            dst[3] = dst[0];
            if (bw==8)
            {
              dst[4] = dst[0];
              dst[5] = dst[0];
              dst[6] = dst[0];
              dst[7] = dst[0];
            }
            PacketScalar b0, b1, tmp;
            // TODO in unaligned mode, preload the next element
//             PacketScalar tmp1 = _mm_load_ps(&m_rhs.derived().data()[l1jsize+l2k]);
            asm("#eigen begincore");
            for(int k=l2k; k<l2blockSizeEnd; k+=ps)
            {
              //PacketScalar tmp = m_rhs.packetCoeff(k, l1j);
              if (rhsIsAligned)
                tmp = ei_pload(&m_rhs.derived().data()[l1jsize + k]);
              else
                tmp = ei_ploadu(&m_rhs.derived().data()[l1jsize + k]);

              b0 = ei_pload(&(localB[k*bw]));
              b1 = ei_pload(&(localB[k*bw+ps]));
              dst[0] = ei_pmadd(tmp, b0, dst[0]);
              b0 = ei_pload(&(localB[k*bw+2*ps]));
              dst[1] = ei_pmadd(tmp, b1, dst[1]);
              b1 = ei_pload(&(localB[k*bw+3*ps]));
              dst[2] = ei_pmadd(tmp, b0, dst[2]);
              if (bw==8)
                b0 = ei_pload(&(localB[k*bw+4*ps]));
              dst[3] = ei_pmadd(tmp, b1, dst[3]);
              if (bw==8)
              {
                b1 = ei_pload(&(localB[k*bw+5*ps]));
                dst[4] = ei_pmadd(tmp, b0, dst[4]);
                b0 = ei_pload(&(localB[k*bw+6*ps]));
                dst[5] = ei_pmadd(tmp, b1, dst[5]);
                b1 = ei_pload(&(localB[k*bw+7*ps]));
                dst[6] = ei_pmadd(tmp, b0, dst[6]);
                dst[7] = ei_pmadd(tmp, b1, dst[7]);
              }
            }

            res.template writePacketCoeff<Aligned>(l1i, l1j, ei_padd(res.template packetCoeff<Aligned>(l1i, l1j), ei_predux(dst)));
            if (ps==2)
              res.template writePacketCoeff<Aligned>(l1i+2,l1j, ei_padd(res.template packetCoeff<Aligned>(l1i+2,l1j), ei_predux(&(dst[2]))));
            if (bw==8)
            {
              res.template writePacketCoeff<Aligned>(l1i+4,l1j, ei_padd(res.template packetCoeff<Aligned>(l1i+4,l1j), ei_predux(&(dst[4]))));
              if (ps==2)
                res.template writePacketCoeff<Aligned>(l1i+6,l1j, ei_padd(res.template packetCoeff<Aligned>(l1i+6,l1j), ei_predux(&(dst[6]))));
            }

            asm("#eigen endcore");
          }
        }
        if (l2blockRowRemaining>0)
        {
          // TODO optimize this part using a generic templated function that processes N rows
          // here we process the remaining l2blockRowRemaining rows
          for(int l1j=l2j; l1j<l2blockColEnd; l1j+=1)
          {
            int offsetblock = l2k * (l2blockRowEnd-l2i) + (l2blockRowEndBW-l2i)*(l2blockSizeEnd-l2k) - l2k*l2blockRowRemaining;
            const Scalar* localB = &block[offsetblock];

            int l1jsize = l1j * size;

            PacketScalar dst[bw];
            dst[0] = ei_pset1(Scalar(0.));
            for (int w = 1; w<l2blockRowRemaining; ++w)
              dst[w] = dst[0];
            PacketScalar b0, b1, tmp;
            asm("#eigen begincore dynamic");
            for(int k=l2k; k<l2blockSizeEnd; k+=ps)
            {
              //PacketScalar tmp = m_rhs.packetCoeff(k, l1j);
              if (rhsIsAligned)
                tmp = ei_pload(&m_rhs.derived().data()[l1jsize + k]);
              else
                tmp = ei_ploadu(&m_rhs.derived().data()[l1jsize + k]);

              // TODO optimize this loop
              for (int w = 0; w<l2blockRowRemaining; ++w)
                dst[w] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRowRemaining+w*ps])), dst[w]);
            }

            // TODO optimize this loop
            for (int w = 0; w<l2blockRowRemaining; ++w)
              res.coeffRef(l2blockRowEndBW+w, l1j) += ei_predux(dst[w]);

            asm("#eigen endcore dynamic");
          }
        }
      }
    }
  }

  delete[] block;
}

#endif // EIGEN_PRODUCT_H
