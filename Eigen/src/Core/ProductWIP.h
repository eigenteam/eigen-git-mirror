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
    res =  ei_pmadd(ei_pset1(lhs.coeff(row, Index)), rhs.template packetCoeff<Aligned>(Index, col), res);
  }
};

template<int Index, int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<false, Index, Size, Lhs, Rhs, PacketScalar>
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    ei_packet_product_unroller<false, Index-1, Size, Lhs, Rhs, PacketScalar>::run(row, col, lhs, rhs, res);
    res =  ei_pmadd(lhs.template packetCoeff<Aligned>(row, Index), ei_pset1(rhs.coeff(Index, col)), res);
  }
};

template<int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<true, 0, Size, Lhs, Rhs, PacketScalar>
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    res = ei_pmul(ei_pset1(lhs.coeff(row, 0)),rhs.template packetCoeff<Aligned>(0, col));
  }
};

template<int Size, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<false, 0, Size, Lhs, Rhs, PacketScalar>
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    res = ei_pmul(lhs.template packetCoeff<Aligned>(row, 0), ei_pset1(rhs.coeff(0, col)));
  }
};

template<bool RowMajor, int Index, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<RowMajor, Index, Dynamic, Lhs, Rhs, PacketScalar>
{
  static void run(int, int, const Lhs&, const Rhs&, PacketScalar&) {}
};

template<int Index, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_packet_product_unroller<false, Index, Dynamic, Lhs, Rhs, PacketScalar>
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
              ? CacheOptimalProduct : NormalProduct };
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

template<typename T, int n=1> struct ei_product_nested_rhs
{
  typedef typename ei_meta_if<
    ei_is_temporary<T>::ret && !(ei_traits<T>::Flags & RowMajorBit),
    T,
    typename ei_meta_if<
         (ei_traits<T>::Flags & EvalBeforeNestingBit)
      || (ei_traits<T>::Flags & RowMajorBit)
      || (!(ei_traits<T>::Flags & ReferencableBit))
      || (n+1) * NumTraits<typename ei_traits<T>::Scalar>::ReadCost < (n-1) * T::CoeffReadCost,
      typename ei_product_eval_to_column_major<T>::type,
      const T&
    >::ret
  >::ret type;
};

template<typename Lhs, typename Rhs, int EvalMode>
struct ei_traits<Product<Lhs, Rhs, EvalMode> >
{
  typedef typename Lhs::Scalar Scalar;
  // the cache friendly product evals lhs once only
  // FIXME what to do if we chose to dynamically call the normal product from the cache friendly one for small matrices ?
  typedef typename ei_nested<Lhs, EvalMode==CacheOptimalProduct ? 0 : Rhs::ColsAtCompileTime>::type LhsNested;
  // NOTE that rhs must be ColumnMajor, so we might need a special nested type calculation
  typedef typename ei_meta_if<EvalMode==CacheOptimalProduct,
      typename ei_product_nested_rhs<Rhs,Lhs::RowsAtCompileTime>::type,
      typename ei_nested<Rhs,Lhs::RowsAtCompileTime>::type>::ret RhsNested;
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
    // the vectorization flags are only used by the normal product,
    // the other one is always vectorized !
    _RhsVectorizable = (RhsFlags & RowMajorBit) && (RhsFlags & VectorizableBit) && (ColsAtCompileTime % ei_packet_traits<Scalar>::size == 0),
    _LhsVectorizable = (!(LhsFlags & RowMajorBit)) && (LhsFlags & VectorizableBit) && (RowsAtCompileTime % ei_packet_traits<Scalar>::size == 0),
    _Vectorizable = (_LhsVectorizable || _RhsVectorizable) ? 0 : 0,
    _RowMajor = (RhsFlags & RowMajorBit)
              && (EvalMode==(int)CacheOptimalProduct ? (int)LhsFlags & RowMajorBit : (!_LhsVectorizable)),
    _LostBits = DefaultLostFlagMask & ~(
                (_RowMajor ? 0 : RowMajorBit)
              | ((RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic) ? 0 : LargeBit)),
    Flags = ((unsigned int)(LhsFlags | RhsFlags) & _LostBits)
    #ifndef EIGEN_WIP_PRODUCT_DIRTY
          | EvalBeforeAssigningBit //FIXME
    #endif
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

    enum {
      PacketSize = ei_packet_traits<Scalar>::size,
      #if (defined __i386__)
      // i386 architectures provides only 8 xmmm register,
      // so let's reduce the max number of rows processed at once.
      // NOTE that so far the maximal supported value is 8.
      MaxBlockRows = 4,
      MaxBlockRows_ClampingMask = 0xFFFFFC,
      #else
      MaxBlockRows = 8,
      MaxBlockRows_ClampingMask = 0xFFFFF8,
      #endif
      // maximal size of the blocks fitted in L2 cache
      MaxL2BlockSize = EIGEN_TUNE_FOR_L2_CACHE_SIZE / sizeof(Scalar)
    };

    Product(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      ei_assert(lhs.cols() == rhs.rows());
    }

    /** \internal */
    template<typename DestDerived>
    void _cacheFriendlyEval(DestDerived& res) const;

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
        ei_packet_product_unroller<Flags&RowMajorBit ? true : false, Lhs::ColsAtCompileTime-1,
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
      res = ei_pmul(ei_pset1(m_lhs.coeff(row, 0)),m_rhs.template packetCoeff<Aligned>(0, col));
      for(int i = 1; i < m_lhs.cols(); i++)
        res =  ei_pmadd(ei_pset1(m_lhs.coeff(row, i)), m_rhs.template packetCoeff<Aligned>(i, col), res);
      return res;
    }

    PacketScalar _packetCoeffColumnMajor(int row, int col) const
    {
      PacketScalar res;
      res = ei_pmul(m_lhs.template packetCoeff<Aligned>(row, 0), ei_pset1(m_rhs.coeff(0, col)));
      for(int i = 1; i < m_lhs.cols(); i++)
        res =  ei_pmadd(m_lhs.template packetCoeff<Aligned>(row, i), ei_pset1(m_rhs.coeff(i, col)), res);
      return res;
    }

    /** \internal */
    template<typename DestDerived, int RhsAlignment, int ResAlignment>
    void _cacheFriendlyEvalImpl(DestDerived& res) const __attribute__ ((noinline));

    /** \internal */
    template<typename DestDerived, int RhsAlignment, int ResAlignment, int BlockRows>
    void _cacheFriendlyEvalKernel(DestDerived& res,
      int l2i, int l2j, int l2k, int l1i,
      int l2blockRowEnd, int l2blockColEnd, int l2blockSizeEnd, const Scalar* block) const EIGEN_DONT_INLINE;

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
  product._cacheFriendlyEval(derived());
  return derived();
}

template<typename Lhs, typename Rhs, int EvalMode>
template<typename DestDerived>
void Product<Lhs,Rhs,EvalMode>::_cacheFriendlyEval(DestDerived& res) const
{
  const bool rhsIsAligned = (m_lhs.cols()%PacketSize == 0);
  const bool resIsAligned = ((_rows()%PacketSize) == 0);

  if (rhsIsAligned && resIsAligned)
    _cacheFriendlyEvalImpl<DestDerived, Aligned, Aligned>(res);
  else if (rhsIsAligned && (!resIsAligned))
    _cacheFriendlyEvalImpl<DestDerived, Aligned, UnAligned>(res);
  else if ((!rhsIsAligned) && resIsAligned)
    _cacheFriendlyEvalImpl<DestDerived, UnAligned, Aligned>(res);
  else
    _cacheFriendlyEvalImpl<DestDerived, UnAligned, UnAligned>(res);

}

template<typename Lhs, typename Rhs, int EvalMode>
template<typename DestDerived, int RhsAlignment, int ResAlignment, int BlockRows>
void Product<Lhs,Rhs,EvalMode>::_cacheFriendlyEvalKernel(DestDerived& res,
  int l2i, int l2j, int l2k, int l1i,
  int l2blockRowEnd, int l2blockColEnd, int l2blockSizeEnd, const Scalar* block) const
{
  asm("#eigen begin kernel");

  ei_internal_assert(BlockRows<=8);

  // NOTE: sounds like we cannot rely on meta-unrolling to access dst[I] without enforcing GCC
  // to create the dst's elements in memory, hence killing the performance.

  for(int l1j=l2j; l1j<l2blockColEnd; l1j+=1)
  {
    int offsetblock = l2k * (l2blockRowEnd-l2i) + (l1i-l2i)*(l2blockSizeEnd-l2k) - l2k*BlockRows;
    const Scalar* localB = &block[offsetblock];

//     int l1jsize = l1j * m_lhs.cols(); //TODO find a better way to optimize address computation ?
    Scalar* rhsColumn = &(m_rhs.const_cast_derived().coeffRef(0, l1j));

    // don't worry, dst is a set of registers
    PacketScalar dst[BlockRows];
    dst[0] = ei_pset1(Scalar(0.));
    switch(BlockRows)
    {
    case 8: dst[7] = dst[0];
    case 7: dst[6] = dst[0];
    case 6: dst[5] = dst[0];
    case 5: dst[4] = dst[0];
    case 4: dst[3] = dst[0];
    case 3: dst[2] = dst[0];
    case 2: dst[1] = dst[0];
    default: break;
    }

    // let's declare a few other temporary registers
    PacketScalar tmp, tmp1;

    // unaligned loads are expensive, therefore let's preload the next element in advance
    if (RhsAlignment==UnAligned)
      //tmp1 = ei_ploadu(&m_rhs.data()[l1jsize+l2k]);
      tmp1 = ei_ploadu(&rhsColumn[l2k]);

    for(int k=l2k; k<l2blockSizeEnd; k+=PacketSize)
    {
      // FIXME if we don't cache l1j*m_lhs.cols() then the performance are poor,
      // let's directly access to the data
      //PacketScalar tmp = m_rhs.template packetCoeff<Aligned>(k, l1j);
      if (RhsAlignment==Aligned)
      {
        //tmp = ei_pload(&m_rhs.data()[l1jsize + k]);
        tmp = ei_pload(&rhsColumn[k]);
      }
      else
      {
        tmp = tmp1;
        if (k+PacketSize<l2blockSizeEnd)
          //tmp1 = ei_ploadu(&m_rhs.data()[l1jsize + k+PacketSize]);
          tmp1 = ei_ploadu(&rhsColumn[k+PacketSize]);
      }

                        dst[0] = ei_pmadd(tmp, ei_pload(&(localB[k*BlockRows             ])), dst[0]);
      if (BlockRows>=2) dst[1] = ei_pmadd(tmp, ei_pload(&(localB[k*BlockRows+  PacketSize])), dst[1]);
      if (BlockRows>=3) dst[2] = ei_pmadd(tmp, ei_pload(&(localB[k*BlockRows+2*PacketSize])), dst[2]);
      if (BlockRows>=4) dst[3] = ei_pmadd(tmp, ei_pload(&(localB[k*BlockRows+3*PacketSize])), dst[3]);
      if (BlockRows>=5) dst[4] = ei_pmadd(tmp, ei_pload(&(localB[k*BlockRows+4*PacketSize])), dst[4]);
      if (BlockRows>=6) dst[5] = ei_pmadd(tmp, ei_pload(&(localB[k*BlockRows+5*PacketSize])), dst[5]);
      if (BlockRows>=7) dst[6] = ei_pmadd(tmp, ei_pload(&(localB[k*BlockRows+6*PacketSize])), dst[6]);
      if (BlockRows>=8) dst[7] = ei_pmadd(tmp, ei_pload(&(localB[k*BlockRows+7*PacketSize])), dst[7]);
    }

    enum {
      // Number of rows we can reduce per packet
      PacketRows = (ResAlignment==Aligned && PacketSize>1) ? (BlockRows / PacketSize) : 0,
      // First row index from which we have to to do redux once at a time
      RemainingStart = PacketSize * PacketRows
    };

    // we have up to 4 packets (for doubles: 8 rows / 2)
    if (PacketRows>=1)
      res.template writePacketCoeff<Aligned>(l1i, l1j,
        ei_padd(res.template packetCoeff<Aligned>(l1i, l1j), ei_preduxp(&(dst[0]))));
    if (PacketRows>=2)
      res.template writePacketCoeff<Aligned>(l1i+PacketSize, l1j,
        ei_padd(res.template packetCoeff<Aligned>(l1i+PacketSize, l1j), ei_preduxp(&(dst[PacketSize]))));
    if (PacketRows>=3)
      res.template writePacketCoeff<Aligned>(l1i+2*PacketSize, l1j,
        ei_padd(res.template packetCoeff<Aligned>(l1i+2*PacketSize, l1j), ei_preduxp(&(dst[2*PacketSize]))));
    if (PacketRows>=4)
      res.template writePacketCoeff<Aligned>(l1i+3*PacketSize, l1j,
        ei_padd(res.template packetCoeff<Aligned>(l1i+3*PacketSize, l1j), ei_preduxp(&(dst[3*PacketSize]))));

    // process the remaining rows one at a time
    if (RemainingStart<=0 && BlockRows>=1) res.coeffRef(l1i+0, l1j) += ei_predux(dst[0]);
    if (RemainingStart<=1 && BlockRows>=2) res.coeffRef(l1i+1, l1j) += ei_predux(dst[1]);
    if (RemainingStart<=2 && BlockRows>=3) res.coeffRef(l1i+2, l1j) += ei_predux(dst[2]);
    if (RemainingStart<=3 && BlockRows>=4) res.coeffRef(l1i+3, l1j) += ei_predux(dst[3]);
    if (RemainingStart<=4 && BlockRows>=5) res.coeffRef(l1i+4, l1j) += ei_predux(dst[4]);
    if (RemainingStart<=5 && BlockRows>=6) res.coeffRef(l1i+5, l1j) += ei_predux(dst[5]);
    if (RemainingStart<=6 && BlockRows>=7) res.coeffRef(l1i+6, l1j) += ei_predux(dst[6]);
    if (RemainingStart<=7 && BlockRows>=8) res.coeffRef(l1i+7, l1j) += ei_predux(dst[7]);

    asm("#eigen end kernel");
  }
}

template<typename Lhs, typename Rhs, int EvalMode>
template<typename DestDerived, int RhsAlignment, int ResAlignment>
void Product<Lhs,Rhs,EvalMode>::_cacheFriendlyEvalImpl(DestDerived& res) const
{
  // FIXME find a way to optimize: (an_xpr) + (a * b)
  // then we don't need to clear res and avoid and additional mat-mat sum
  #ifndef EIGEN_WIP_PRODUCT_DIRTY
//   std::cout << "wip product\n";
  res.setZero();
  #endif

  const int rows = _rows();
  const int cols = _cols();
  const int remainingSize = m_lhs.cols()%PacketSize;
  const int size = m_lhs.cols() - remainingSize; // third dimension of the product clamped to packet boundaries
  const int l2BlockRows = MaxL2BlockSize > _rows() ? _rows() : MaxL2BlockSize;
  const int l2BlockCols = MaxL2BlockSize > _cols() ? _cols() : MaxL2BlockSize;
  const int l2BlockSize = MaxL2BlockSize > size    ? size    : MaxL2BlockSize;
  //Scalar* __restrict__ block = new Scalar[l2blocksize*size];;
  Scalar* __restrict__ block = (Scalar*)alloca(sizeof(Scalar)*l2BlockRows*size);

  // loops on each L2 cache friendly blocks of the result
  for(int l2i=0; l2i<_rows(); l2i+=l2BlockRows)
  {
    const int l2blockRowEnd = std::min(l2i+l2BlockRows, rows);
    const int l2blockRowEndBW = l2blockRowEnd & MaxBlockRows_ClampingMask;    // end of the rows aligned to bw
    const int l2blockRemainingRows = l2blockRowEnd - l2blockRowEndBW;         // number of remaining rows

    // build a cache friendly block
    int count = 0;

    // copy l2blocksize rows of m_lhs to blocks of ps x bw
    for(int l2k=0; l2k<size; l2k+=l2BlockSize)
    {
      const int l2blockSizeEnd = std::min(l2k+l2BlockSize, size);

      for (int i = l2i; i<l2blockRowEndBW; i+=MaxBlockRows)
      {
        for (int k=l2k; k<l2blockSizeEnd; k+=PacketSize)
        {
          // TODO write these two loops using meta unrolling
          // negligible for large matrices but useful for small ones
          for (int w=0; w<MaxBlockRows; ++w)
            for (int s=0; s<PacketSize; ++s)
              block[count++] = m_lhs.coeff(i+w,k+s);
        }
      }
      if (l2blockRemainingRows>0)
      {
        for (int k=l2k; k<l2blockSizeEnd; k+=PacketSize)
        {
          for (int w=0; w<l2blockRemainingRows; ++w)
            for (int s=0; s<PacketSize; ++s)
              block[count++] = m_lhs.coeff(l2blockRowEndBW+w,k+s);
        }
      }
    }

    for(int l2j=0; l2j<cols; l2j+=l2BlockCols)
    {
      int l2blockColEnd = std::min(l2j+l2BlockCols, cols);

      for(int l2k=0; l2k<size; l2k+=l2BlockSize)
      {
        // acumulate a bw rows of lhs time a single column of rhs to a bw x 1 block of res
        int l2blockSizeEnd = std::min(l2k+l2BlockSize, size);

        // for each bw x 1 result's block
        for(int l1i=l2i; l1i<l2blockRowEndBW; l1i+=MaxBlockRows)
        {
            _cacheFriendlyEvalKernel<DestDerived, RhsAlignment, ResAlignment, MaxBlockRows>(
              res, l2i, l2j, l2k, l1i, l2blockRowEnd, l2blockColEnd, l2blockSizeEnd, block);
#if 0
          for(int l1j=l2j; l1j<l2blockColEnd; l1j+=1)
          {
            int offsetblock = l2k * (l2blockRowEnd-l2i) + (l1i-l2i)*(l2blockSizeEnd-l2k) - l2k*MaxBlockRows;
            const Scalar* localB = &block[offsetblock];

            int l1jsize = l1j * m_lhs.cols(); //TODO find a better way to optimize address computation ?

            PacketScalar dst[bw];
            dst[0] = ei_pset1(Scalar(0.));
            dst[1] = dst[0];
            dst[2] = dst[0];
            dst[3] = dst[0];
            if (MaxBlockRows==8)
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
            for(int k=l2k; k<l2blockSizeEnd; k+=PacketSize)
            {
//               PacketScalar tmp = m_rhs.template packetCoeff<Aligned>(k, l1j);
              // TODO make this branching compile time (costly for doubles)
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
              if (MaxBlockRows==8)
                b0 = ei_pload(&(localB[k*bw+4*ps]));
              dst[3] = ei_pmadd(tmp, b1, dst[3]);
              if (MaxBlockRows==8)
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

//             if (resIsAligned)
            {
              res.template writePacketCoeff<Aligned>(l1i, l1j, ei_padd(res.template packetCoeff<Aligned>(l1i, l1j), ei_preduxp(dst)));
              if (PacketSize==2)
                res.template writePacketCoeff<Aligned>(l1i+2,l1j, ei_padd(res.template packetCoeff<Aligned>(l1i+2,l1j), ei_preduxp(&(dst[2]))));
              if (MaxBlockRows==8)
              {
                res.template writePacketCoeff<Aligned>(l1i+4,l1j, ei_padd(res.template packetCoeff<Aligned>(l1i+4,l1j), ei_preduxp(&(dst[4]))));
                if (PacketSize==2)
                  res.template writePacketCoeff<Aligned>(l1i+6,l1j, ei_padd(res.template packetCoeff<Aligned>(l1i+6,l1j), ei_preduxp(&(dst[6]))));
              }
            }
//             else
//             {
//               // TODO uncommenting this code kill the perf, even though it is never called !!
//               // this is because dst cannot be a set of registers only
//               // TODO optimize this loop
//               // TODO is it better to do one redux at once or packet reduxes + unaligned store ?
//               for (int w = 0; w<bw; ++w)
//                 res.coeffRef(l1i+w, l1j) += ei_predux(dst[w]);
//               std::cout << "!\n";
//             }

            asm("#eigen endcore");
          }
#endif
        }
        if (l2blockRemainingRows>0)
        {
          // this is an attempt to build an array of kernels, but I did not manage to get it compiles
//           typedef void (*Kernel)(DestDerived& , int, int, int, int, int, int, int, const Scalar*);
//           Kernel kernels[8];
//           kernels[0] = (Kernel)(&Product<Lhs,Rhs,EvalMode>::template _cacheFriendlyEvalKernel<DestDerived, RhsAlignment, ResAlignment, 1>);
//           kernels[l2blockRemainingRows](res, l2i, l2j, l2k, l2blockRowEndBW, l2blockRowEnd, l2blockColEnd, l2blockSizeEnd, block);

          switch(l2blockRemainingRows)
          {
          case 1:_cacheFriendlyEvalKernel<DestDerived, RhsAlignment, ResAlignment, 1>(
              res, l2i, l2j, l2k, l2blockRowEndBW, l2blockRowEnd, l2blockColEnd, l2blockSizeEnd, block); break;
          case 2:_cacheFriendlyEvalKernel<DestDerived, RhsAlignment, ResAlignment, 2>(
              res, l2i, l2j, l2k, l2blockRowEndBW, l2blockRowEnd, l2blockColEnd, l2blockSizeEnd, block); break;
          case 3:_cacheFriendlyEvalKernel<DestDerived, RhsAlignment, ResAlignment, 3>(
              res, l2i, l2j, l2k, l2blockRowEndBW, l2blockRowEnd, l2blockColEnd, l2blockSizeEnd, block); break;
          case 4:_cacheFriendlyEvalKernel<DestDerived, RhsAlignment, ResAlignment, 4>(
              res, l2i, l2j, l2k, l2blockRowEndBW, l2blockRowEnd, l2blockColEnd, l2blockSizeEnd, block); break;
          case 5:_cacheFriendlyEvalKernel<DestDerived, RhsAlignment, ResAlignment, 5>(
              res, l2i, l2j, l2k, l2blockRowEndBW, l2blockRowEnd, l2blockColEnd, l2blockSizeEnd, block); break;
          case 6:_cacheFriendlyEvalKernel<DestDerived, RhsAlignment, ResAlignment, 6>(
              res, l2i, l2j, l2k, l2blockRowEndBW, l2blockRowEnd, l2blockColEnd, l2blockSizeEnd, block); break;
          case 7:_cacheFriendlyEvalKernel<DestDerived, RhsAlignment, ResAlignment, 7>(
              res, l2i, l2j, l2k, l2blockRowEndBW, l2blockRowEnd, l2blockColEnd, l2blockSizeEnd, block); break;
          default:
            ei_internal_assert(false && "internal error"); break;
          }
        }
      }
    }
  }

  // handle the part which cannot be processed by the vectorized path
  if (remainingSize)
  {
    res += Product<
      Block<typename ei_unconst<_LhsNested>::type,Dynamic,Dynamic>,
      Block<typename ei_unconst<_RhsNested>::type,Dynamic,Dynamic>,
      NormalProduct>(
        m_lhs.block(0,size, _rows(), remainingSize),
        m_rhs.block(size,0, remainingSize, _cols())).lazy();
//     res += m_lhs.block(0,size, _rows(), remainingSize)._lazyProduct(m_rhs.block(size,0, remainingSize, _cols()));
  }

//   delete[] block;
}

#endif // EIGEN_PRODUCT_H
