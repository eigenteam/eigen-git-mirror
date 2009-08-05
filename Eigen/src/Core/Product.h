// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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

/** \class GeneralProduct
  *
  * \brief Expression of the product of two general matrices or vectors
  *
  * \param LhsNested the type used to store the left-hand side
  * \param RhsNested the type used to store the right-hand side
  * \param ProductMode the type of the product
  *
  * This class represents an expression of the product of two general matrices.
  * We call a general matrix, a dense matrix with full storage. For instance,
  * This excludes triangular, selfadjoint, and sparse matrices.
  * It is the return type of the operator* between general matrices. Its template
  * arguments are determined automatically by ProductReturnType. Therefore,
  * GeneralProduct should never be used direclty. To determine the result type of a
  * function which involves a matrix product, use ProductReturnType::Type.
  *
  * \sa ProductReturnType, MatrixBase::operator*(const MatrixBase<OtherDerived>&)
  */
template<typename Lhs, typename Rhs, int ProductType = ei_product_type<Lhs,Rhs>::value>
class GeneralProduct;

template<int Rows, int Cols, int Depth> struct ei_product_type_selector;

enum {
  Large = Dynamic,
  Small = Dynamic/2
};

enum { OuterProduct, InnerProduct, UnrolledProduct, GemvProduct, GemmProduct };

template<typename Lhs, typename Rhs> struct ei_product_type
{
  enum {
    Rows  = Lhs::RowsAtCompileTime,
    Cols  = Rhs::ColsAtCompileTime,
    Depth = EIGEN_ENUM_MIN(Lhs::ColsAtCompileTime,Rhs::RowsAtCompileTime),

    value = ei_product_type_selector<(Rows>8  ? Large : (Rows==1   ? 1 : Small)),
                                     (Cols>8  ? Large : (Cols==1   ? 1 : Small)),
                                     (Depth>8 ? Large : (Depth==1  ? 1 : Small))>::ret
  };
};

template<int Rows, int Cols>  struct ei_product_type_selector<Rows,Cols,1>        { enum { ret = OuterProduct }; };
template<int Depth>           struct ei_product_type_selector<1,1,Depth>          { enum { ret = InnerProduct }; };
template<>                    struct ei_product_type_selector<1,1,1>              { enum { ret = InnerProduct }; };
template<>                    struct ei_product_type_selector<Small,1,Small>      { enum { ret = UnrolledProduct }; };
template<>                    struct ei_product_type_selector<1,Small,Small>      { enum { ret = UnrolledProduct }; };
template<>                    struct ei_product_type_selector<Small,Small,Small>  { enum { ret = UnrolledProduct }; };

// template<>                    struct ei_product_type_selector<Small,1,Small>      { enum { ret = GemvProduct }; };
// template<>                    struct ei_product_type_selector<1,Small,Small>      { enum { ret = GemvProduct }; };
// template<>                    struct ei_product_type_selector<Small,Small,Small>  { enum { ret = GemmProduct }; };

template<>                    struct ei_product_type_selector<1,Large,Small>      { enum { ret = GemvProduct }; };
template<>                    struct ei_product_type_selector<1,Large,Large>      { enum { ret = GemvProduct }; };
template<>                    struct ei_product_type_selector<1,Small,Large>      { enum { ret = GemvProduct }; };
template<>                    struct ei_product_type_selector<Large,1,Small>      { enum { ret = GemvProduct }; };
template<>                    struct ei_product_type_selector<Large,1,Large>      { enum { ret = GemvProduct }; };
template<>                    struct ei_product_type_selector<Small,1,Large>      { enum { ret = GemvProduct }; };
template<>                    struct ei_product_type_selector<Small,Small,Large>  { enum { ret = GemmProduct }; };
template<>                    struct ei_product_type_selector<Large,Small,Large>  { enum { ret = GemmProduct }; };
template<>                    struct ei_product_type_selector<Small,Large,Large>  { enum { ret = GemmProduct }; };
template<>                    struct ei_product_type_selector<Large,Large,Large>  { enum { ret = GemmProduct }; };
template<>                    struct ei_product_type_selector<Large,Small,Small>  { enum { ret = GemmProduct }; };
template<>                    struct ei_product_type_selector<Small,Large,Small>  { enum { ret = GemmProduct }; };
template<>                    struct ei_product_type_selector<Large,Large,Small>  { enum { ret = GemmProduct }; };

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
  * which involve a matrix product. The class Product should never be
  * used directly.
  *
  * \sa class Product, MatrixBase::operator*(const MatrixBase<OtherDerived>&)
  */
template<typename Lhs, typename Rhs, int ProductType>
struct ProductReturnType
{
  // TODO use the nested type to reduce instanciations ????
//   typedef typename ei_nested<Lhs,Rhs::ColsAtCompileTime>::type LhsNested;
//   typedef typename ei_nested<Rhs,Lhs::RowsAtCompileTime>::type RhsNested;

  typedef GeneralProduct<Lhs/*Nested*/, Rhs/*Nested*/, ProductType> Type;
};

template<typename Lhs, typename Rhs>
struct ProductReturnType<Lhs,Rhs,UnrolledProduct>
{
  typedef typename ei_nested<Lhs,Rhs::ColsAtCompileTime>::type LhsNested;
  typedef typename ei_nested<Rhs,Lhs::RowsAtCompileTime>::type RhsNested;
  typedef GeneralProduct<Lhs, Rhs, UnrolledProduct> Type;
};


/***********************************************************************
*  Implementation of General Matrix Matrix Product
***********************************************************************/

template<typename Lhs, typename Rhs>
struct ei_traits<GeneralProduct<Lhs,Rhs,GemmProduct> >
 : ei_traits<ProductBase<GeneralProduct<Lhs,Rhs,GemmProduct>, Lhs, Rhs> >
{};

template<typename Lhs, typename Rhs>
class GeneralProduct<Lhs, Rhs, GemmProduct>
  : public ProductBase<GeneralProduct<Lhs,Rhs,GemmProduct>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(GeneralProduct)

    GeneralProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}

    template<typename Dest> void addTo(Dest& dst, Scalar alpha) const
    {
      ei_assert(dst.rows()==m_lhs.rows() && dst.cols()==m_rhs.cols());

      const ActualLhsType lhs = LhsBlasTraits::extract(m_lhs);
      const ActualRhsType rhs = RhsBlasTraits::extract(m_rhs);

      Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(m_lhs)
                                 * RhsBlasTraits::extractScalarFactor(m_rhs);

      ei_general_matrix_matrix_product<
        Scalar,
        (_ActualLhsType::Flags&RowMajorBit)?RowMajor:ColMajor, bool(LhsBlasTraits::NeedToConjugate),
        (_ActualRhsType::Flags&RowMajorBit)?RowMajor:ColMajor, bool(RhsBlasTraits::NeedToConjugate),
        (Dest::Flags&RowMajorBit)?RowMajor:ColMajor>
      ::run(
          this->rows(), this->cols(), lhs.cols(),
          (const Scalar*)&(lhs.const_cast_derived().coeffRef(0,0)), lhs.stride(),
          (const Scalar*)&(rhs.const_cast_derived().coeffRef(0,0)), rhs.stride(),
          (Scalar*)&(dst.coeffRef(0,0)), dst.stride(),
          actualAlpha);
    }
};

/***********************************************************************
*  Implementation of Inner Vector Vector Product
***********************************************************************/

template<typename Lhs, typename Rhs>
struct ei_traits<GeneralProduct<Lhs,Rhs,InnerProduct> >
 : ei_traits<ProductBase<GeneralProduct<Lhs,Rhs,InnerProduct>, Lhs, Rhs> >
{};

template<typename Lhs, typename Rhs>
class GeneralProduct<Lhs, Rhs, InnerProduct>
  : public ProductBase<GeneralProduct<Lhs,Rhs,InnerProduct>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(GeneralProduct)

    GeneralProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}

    template<typename Dest> void addTo(Dest& dst, Scalar alpha) const
    {
      ei_assert(dst.rows()==1 && dst.cols()==1);
      dst.coeffRef(0,0) += (m_lhs.cwise()*m_rhs).sum();
    }
};

/***********************************************************************
*  Implementation of Outer Vector Vector Product
***********************************************************************/
template<int StorageOrder> struct ei_outer_product_selector;

template<typename Lhs, typename Rhs>
struct ei_traits<GeneralProduct<Lhs,Rhs,OuterProduct> >
 : ei_traits<ProductBase<GeneralProduct<Lhs,Rhs,OuterProduct>, Lhs, Rhs> >
{};

template<typename Lhs, typename Rhs>
class GeneralProduct<Lhs, Rhs, OuterProduct>
  : public ProductBase<GeneralProduct<Lhs,Rhs,OuterProduct>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(GeneralProduct)

    GeneralProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}

    template<typename Dest> void addTo(Dest& dest, Scalar alpha) const
    {
      ei_outer_product_selector<Dest::Flags&RowMajorBit>::run(*this, dest, alpha);
    }
};

template<> struct ei_outer_product_selector<ColMajor> {
  template<typename ProductType, typename Dest>
  static void run(const ProductType& prod, Dest& dest, typename ProductType::Scalar alpha) {
    // FIXME make sure lhs is sequentially stored
    const int cols = dest.cols();
    for (int j=0; j<cols; ++j)
      dest.col(j) += (alpha * prod.rhs().coeff(j)) * prod.lhs();
  }
};

template<> struct ei_outer_product_selector<RowMajor> {
  template<typename ProductType, typename Dest>
  static void run(const ProductType& prod, Dest& dest, typename ProductType::Scalar alpha) {
    // FIXME make sure rhs is sequentially stored
    const int rows = dest.rows();
    for (int i=0; i<rows; ++i)
      dest.row(i) += (alpha * prod.lhs().coeff(i)) * prod.rhs();
  }
};

/***********************************************************************
*  Implementation of General Matrix Vector Product
***********************************************************************/

/*  According to the shape/flags of the matrix we have to distinghish 3 different cases:
 *   1 - the matrix is col-major, BLAS compatible and M is large => call fast BLAS-like colmajor routine
 *   2 - the matrix is row-major, BLAS compatible and N is large => call fast BLAS-like rowmajor routine
 *   3 - all other cases are handled using a simple loop along the outer-storage direction.
 *  Therefore we need a lower level meta selector.
 *  Furthermore, if the matrix is the rhs, then the product has to be transposed.
 */
template<typename Lhs, typename Rhs>
struct ei_traits<GeneralProduct<Lhs,Rhs,GemvProduct> >
 : ei_traits<ProductBase<GeneralProduct<Lhs,Rhs,GemvProduct>, Lhs, Rhs> >
{};

template<int Side, int StorageOrder, bool BlasCompatible>
struct ei_gemv_selector;

template<typename Lhs, typename Rhs>
class GeneralProduct<Lhs, Rhs, GemvProduct>
  : public ProductBase<GeneralProduct<Lhs,Rhs,GemvProduct>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(GeneralProduct)

    GeneralProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}

    enum { Side = Lhs::IsVectorAtCompileTime ? OnTheLeft : OnTheRight };
    typedef typename ei_meta_if<int(Side)==OnTheRight,_LhsNested,_RhsNested>::ret MatrixType;

    template<typename Dest> void addTo(Dest& dst, Scalar alpha) const
    {
      ei_assert(m_lhs.rows() == dst.rows() && m_rhs.cols() == dst.cols());
      ei_gemv_selector<Side,int(MatrixType::Flags)&RowMajorBit,
                       ei_blas_traits<MatrixType>::ActualAccess>::run(*this, dst, alpha);
    }
};

// The vector is on the left => transposition
template<int StorageOrder, bool BlasCompatible>
struct ei_gemv_selector<OnTheLeft,StorageOrder,BlasCompatible>
{
  template<typename ProductType, typename Dest>
  static void run(const ProductType& prod, Dest& dest, typename ProductType::Scalar alpha)
  {
    Transpose<Dest> destT(dest);
    ei_gemv_selector<OnTheRight,!StorageOrder,BlasCompatible>
      ::run(GeneralProduct<Transpose<typename ProductType::_RhsNested>,Transpose<typename ProductType::_LhsNested> >
        (prod.rhs().transpose(), prod.lhs().transpose()), destT, alpha);
  }
};

template<> struct ei_gemv_selector<OnTheRight,ColMajor,true>
{
  template<typename ProductType, typename Dest>
  static void run(const ProductType& prod, Dest& dest, typename ProductType::Scalar alpha)
  {
    typedef typename ProductType::Scalar Scalar;
    typedef typename ProductType::ActualLhsType ActualLhsType;
    typedef typename ProductType::ActualRhsType ActualRhsType;
    typedef typename ProductType::LhsBlasTraits LhsBlasTraits;
    typedef typename ProductType::RhsBlasTraits RhsBlasTraits;

    ActualLhsType actualLhs = LhsBlasTraits::extract(prod.lhs());
    ActualRhsType actualRhs = RhsBlasTraits::extract(prod.rhs());

    Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(prod.lhs())
                               * RhsBlasTraits::extractScalarFactor(prod.rhs());

    enum {
      EvalToDest = (ei_packet_traits<Scalar>::size==1)
                 ||((Dest::Flags&ActualPacketAccessBit) && (!(Dest::Flags & RowMajorBit)))
    };
    Scalar* EIGEN_RESTRICT actualDest;
    if (EvalToDest)
      actualDest = &dest.coeffRef(0);
    else
    {
      actualDest = ei_aligned_stack_new(Scalar,dest.size());
      Map<Matrix<Scalar,Dest::RowsAtCompileTime,1> >(actualDest, dest.size()) = dest;
    }

    ei_cache_friendly_product_colmajor_times_vector
      <LhsBlasTraits::NeedToConjugate,RhsBlasTraits::NeedToConjugate>(
      dest.size(),
      &actualLhs.const_cast_derived().coeffRef(0,0), actualLhs.stride(),
      actualRhs, actualDest, actualAlpha);

    if (!EvalToDest)
    {
      dest = Map<Matrix<Scalar,Dest::SizeAtCompileTime,1> >(actualDest, dest.size());
      ei_aligned_stack_delete(Scalar, actualDest, dest.size());
    }
  }
};

template<> struct ei_gemv_selector<OnTheRight,RowMajor,true>
{
  template<typename ProductType, typename Dest>
  static void run(const ProductType& prod, Dest& dest, typename ProductType::Scalar alpha)
  {
    typedef typename ProductType::Scalar Scalar;
    typedef typename ProductType::ActualLhsType ActualLhsType;
    typedef typename ProductType::ActualRhsType ActualRhsType;
    typedef typename ProductType::_ActualRhsType _ActualRhsType;
    typedef typename ProductType::LhsBlasTraits LhsBlasTraits;
    typedef typename ProductType::RhsBlasTraits RhsBlasTraits;

    ActualLhsType actualLhs = LhsBlasTraits::extract(prod.lhs());
    ActualRhsType actualRhs = RhsBlasTraits::extract(prod.rhs());

    Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(prod.lhs())
                               * RhsBlasTraits::extractScalarFactor(prod.rhs());
                               
    enum {
      DirectlyUseRhs = ((ei_packet_traits<Scalar>::size==1) || (_ActualRhsType::Flags&ActualPacketAccessBit))
                     && (!(_ActualRhsType::Flags & RowMajorBit))
    };

    Scalar* EIGEN_RESTRICT rhs_data;
    if (DirectlyUseRhs)
       rhs_data = &actualRhs.const_cast_derived().coeffRef(0);
    else
    {
      rhs_data = ei_aligned_stack_new(Scalar, actualRhs.size());
      Map<Matrix<Scalar,_ActualRhsType::SizeAtCompileTime,1> >(rhs_data, actualRhs.size()) = actualRhs;
    }

    ei_cache_friendly_product_rowmajor_times_vector
      <LhsBlasTraits::NeedToConjugate,RhsBlasTraits::NeedToConjugate>(
        &actualLhs.const_cast_derived().coeffRef(0,0), actualLhs.stride(),
        rhs_data, prod.rhs().size(), dest, actualAlpha);

    if (!DirectlyUseRhs) ei_aligned_stack_delete(Scalar, rhs_data, prod.rhs().size());
  }
};

template<> struct ei_gemv_selector<OnTheRight,ColMajor,false>
{
  template<typename ProductType, typename Dest>
  static void run(const ProductType& prod, Dest& dest, typename ProductType::Scalar alpha)
  {
    // TODO makes sure dest is sequentially stored in memory, otherwise use a temp
    const int size = prod.rhs().rows();
    for(int k=0; k<size; ++k)
      dest += (alpha*prod.rhs().coeff(k)) * prod.lhs().col(k);
  }
};

template<> struct ei_gemv_selector<OnTheRight,RowMajor,false>
{
  template<typename ProductType, typename Dest>
  static void run(const ProductType& prod, Dest& dest, typename ProductType::Scalar alpha)
  {
    // TODO makes sure rhs is sequentially stored in memory, otherwise use a temp
    const int rows = prod.rows();
    for(int i=0; i<rows; ++i)
      dest.coeffRef(i) += alpha * (prod.lhs().row(i).cwise() * prod.rhs().transpose()).sum();
  }
};

/***********************************************************************
*  Implementation of products with small fixed sizes
***********************************************************************/

/* Since the all the dimensions of the product are small, here we can rely
 * on the generic Assign mechanism to evaluate the product per coeff (or packet).
 * 
 * Note that the here inner-loops should always be unrolled.
 */

template<int VectorizationMode, int Index, typename Lhs, typename Rhs, typename RetScalar>
struct ei_product_coeff_impl;

template<int StorageOrder, int Index, typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl;

template<typename LhsNested, typename RhsNested>
struct ei_traits<GeneralProduct<LhsNested,RhsNested,UnrolledProduct> >
{
  typedef typename ei_cleantype<LhsNested>::type _LhsNested;
  typedef typename ei_cleantype<RhsNested>::type _RhsNested;
  typedef typename ei_scalar_product_traits<typename _LhsNested::Scalar, typename _RhsNested::Scalar>::ReturnType Scalar;
  
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
                      && (ColsAtCompileTime == Dynamic || (ColsAtCompileTime % ei_packet_traits<Scalar>::size) == 0),

      CanVectorizeLhs = (!LhsRowMajor) && (LhsFlags & PacketAccessBit)
                      && (RowsAtCompileTime == Dynamic || (RowsAtCompileTime % ei_packet_traits<Scalar>::size) == 0),

      EvalToRowMajor = RhsRowMajor && (!CanVectorizeLhs),

      RemovedBits = ~(EvalToRowMajor ? 0 : RowMajorBit),

      Flags = ((unsigned int)(LhsFlags | RhsFlags) & HereditaryBits & RemovedBits)
            | EvalBeforeAssigningBit
            | EvalBeforeNestingBit
            | (CanVectorizeLhs || CanVectorizeRhs ? PacketAccessBit : 0)
            | (LhsFlags & RhsFlags & AlignedBit),

      CoeffReadCost = InnerSize == Dynamic ? Dynamic
                    : InnerSize * (NumTraits<Scalar>::MulCost + LhsCoeffReadCost + RhsCoeffReadCost)
                      + (InnerSize - 1) * NumTraits<Scalar>::AddCost,

      /* CanVectorizeInner deserves special explanation. It does not affect the product flags. It is not used outside
      * of Product. If the Product itself is not a packet-access expression, there is still a chance that the inner
      * loop of the product might be vectorized. This is the meaning of CanVectorizeInner. Since it doesn't affect
      * the Flags, it is safe to make this value depend on ActualPacketAccessBit, that doesn't affect the ABI.
      */
      CanVectorizeInner = LhsRowMajor && (!RhsRowMajor) && (LhsFlags & RhsFlags & ActualPacketAccessBit)
                        && (InnerSize % ei_packet_traits<Scalar>::size == 0)
    };
};

template<typename LhsNested, typename RhsNested> class GeneralProduct<LhsNested,RhsNested,UnrolledProduct>
  : ei_no_assignment_operator,
    public MatrixBase<GeneralProduct<LhsNested, RhsNested, UnrolledProduct> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(GeneralProduct)

  private:

    typedef typename ei_traits<GeneralProduct>::_LhsNested _LhsNested;
    typedef typename ei_traits<GeneralProduct>::_RhsNested _RhsNested;

    enum {
      PacketSize = ei_packet_traits<Scalar>::size,
      InnerSize  = ei_traits<GeneralProduct>::InnerSize,
      Unroll = CoeffReadCost <= EIGEN_UNROLLING_LIMIT,
      CanVectorizeInner = ei_traits<GeneralProduct>::CanVectorizeInner
    };

    typedef ei_product_coeff_impl<CanVectorizeInner ? InnerVectorization : NoVectorization,
                                  Unroll ? InnerSize-1 : Dynamic,
                                  _LhsNested, _RhsNested, Scalar> ScalarCoeffImpl;

  public:

    template<typename Lhs, typename Rhs>
    inline GeneralProduct(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      // we don't allow taking products of matrices of different real types, as that wouldn't be vectorizable.
      // We still allow to mix T and complex<T>.
      EIGEN_STATIC_ASSERT((ei_is_same_type<typename Lhs::RealScalar, typename Rhs::RealScalar>::ret),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
      ei_assert(lhs.cols() == rhs.rows()
        && "invalid matrix product"
        && "if you wanted a coeff-wise or a dot product use the respective explicit functions");
    }

    EIGEN_STRONG_INLINE int rows() const { return m_lhs.rows(); }
    EIGEN_STRONG_INLINE int cols() const { return m_rhs.cols(); }

    EIGEN_STRONG_INLINE const Scalar coeff(int row, int col) const
    {
      Scalar res;
      ScalarCoeffImpl::run(row, col, m_lhs, m_rhs, res);
      return res;
    }

    /* Allow index-based non-packet access. It is impossible though to allow index-based packed access,
     * which is why we don't set the LinearAccessBit.
     */
    EIGEN_STRONG_INLINE const Scalar coeff(int index) const
    {
      Scalar res;
      const int row = RowsAtCompileTime == 1 ? 0 : index;
      const int col = RowsAtCompileTime == 1 ? index : 0;
      ScalarCoeffImpl::run(row, col, m_lhs, m_rhs, res);
      return res;
    }

    template<int LoadMode>
    EIGEN_STRONG_INLINE const PacketScalar packet(int row, int col) const
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


/***************************************************************************
* Normal product .coeff() implementation (with meta-unrolling)
***************************************************************************/

/**************************************
*** Scalar path  - no vectorization ***
**************************************/

template<int Index, typename Lhs, typename Rhs, typename RetScalar>
struct ei_product_coeff_impl<NoVectorization, Index, Lhs, Rhs, RetScalar>
{
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, RetScalar &res)
  {
    ei_product_coeff_impl<NoVectorization, Index-1, Lhs, Rhs, RetScalar>::run(row, col, lhs, rhs, res);
    res += lhs.coeff(row, Index) * rhs.coeff(Index, col);
  }
};

template<typename Lhs, typename Rhs, typename RetScalar>
struct ei_product_coeff_impl<NoVectorization, 0, Lhs, Rhs, RetScalar>
{
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, RetScalar &res)
  {
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
  }
};

template<typename Lhs, typename Rhs, typename RetScalar>
struct ei_product_coeff_impl<NoVectorization, Dynamic, Lhs, Rhs, RetScalar>
{
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, RetScalar& res)
  {
    ei_assert(lhs.cols()>0 && "you are using a non initialized matrix");
    res = lhs.coeff(row, 0) * rhs.coeff(0, col);
      for(int i = 1; i < lhs.cols(); ++i)
        res += lhs.coeff(row, i) * rhs.coeff(i, col);
  }
};

// prevent buggy user code from causing an infinite recursion
template<typename Lhs, typename Rhs, typename RetScalar>
struct ei_product_coeff_impl<NoVectorization, -1, Lhs, Rhs, RetScalar>
{
  EIGEN_STRONG_INLINE static void run(int, int, const Lhs&, const Rhs&, RetScalar&) {}
};

/*******************************************
*** Scalar path with inner vectorization ***
*******************************************/

template<int Index, typename Lhs, typename Rhs, typename PacketScalar>
struct ei_product_coeff_vectorized_unroller
{
  enum { PacketSize = ei_packet_traits<typename Lhs::Scalar>::size };
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, typename Lhs::PacketScalar &pres)
  {
    ei_product_coeff_vectorized_unroller<Index-PacketSize, Lhs, Rhs, PacketScalar>::run(row, col, lhs, rhs, pres);
    pres = ei_padd(pres, ei_pmul( lhs.template packet<Aligned>(row, Index) , rhs.template packet<Aligned>(Index, col) ));
  }
};

template<typename Lhs, typename Rhs, typename PacketScalar>
struct ei_product_coeff_vectorized_unroller<0, Lhs, Rhs, PacketScalar>
{
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, typename Lhs::PacketScalar &pres)
  {
    pres = ei_pmul(lhs.template packet<Aligned>(row, 0) , rhs.template packet<Aligned>(0, col));
  }
};

template<int Index, typename Lhs, typename Rhs, typename RetScalar>
struct ei_product_coeff_impl<InnerVectorization, Index, Lhs, Rhs, RetScalar>
{
  typedef typename Lhs::PacketScalar PacketScalar;
  enum { PacketSize = ei_packet_traits<typename Lhs::Scalar>::size };
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, RetScalar &res)
  {
    PacketScalar pres;
    ei_product_coeff_vectorized_unroller<Index+1-PacketSize, Lhs, Rhs, PacketScalar>::run(row, col, lhs, rhs, pres);
    ei_product_coeff_impl<NoVectorization,Index,Lhs,Rhs,RetScalar>::run(row, col, lhs, rhs, res);
    res = ei_predux(pres);
  }
};

template<typename Lhs, typename Rhs, int LhsRows = Lhs::RowsAtCompileTime, int RhsCols = Rhs::ColsAtCompileTime>
struct ei_product_coeff_vectorized_dyn_selector
{
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    res = ei_dot_impl<
      Block<Lhs, 1, ei_traits<Lhs>::ColsAtCompileTime>,
      Block<Rhs, ei_traits<Rhs>::RowsAtCompileTime, 1>,
      LinearVectorization, NoUnrolling>::run(lhs.row(row), rhs.col(col));
  }
};

// NOTE the 3 following specializations are because taking .col(0) on a vector is a bit slower
// NOTE maybe they are now useless since we have a specialization for Block<Matrix>
template<typename Lhs, typename Rhs, int RhsCols>
struct ei_product_coeff_vectorized_dyn_selector<Lhs,Rhs,1,RhsCols>
{
  EIGEN_STRONG_INLINE static void run(int /*row*/, int col, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    res = ei_dot_impl<
      Lhs,
      Block<Rhs, ei_traits<Rhs>::RowsAtCompileTime, 1>,
      LinearVectorization, NoUnrolling>::run(lhs, rhs.col(col));
  }
};

template<typename Lhs, typename Rhs, int LhsRows>
struct ei_product_coeff_vectorized_dyn_selector<Lhs,Rhs,LhsRows,1>
{
  EIGEN_STRONG_INLINE static void run(int row, int /*col*/, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    res = ei_dot_impl<
      Block<Lhs, 1, ei_traits<Lhs>::ColsAtCompileTime>,
      Rhs,
      LinearVectorization, NoUnrolling>::run(lhs.row(row), rhs);
  }
};

template<typename Lhs, typename Rhs>
struct ei_product_coeff_vectorized_dyn_selector<Lhs,Rhs,1,1>
{
  EIGEN_STRONG_INLINE static void run(int /*row*/, int /*col*/, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    res = ei_dot_impl<
      Lhs,
      Rhs,
      LinearVectorization, NoUnrolling>::run(lhs, rhs);
  }
};

template<typename Lhs, typename Rhs, typename RetScalar>
struct ei_product_coeff_impl<InnerVectorization, Dynamic, Lhs, Rhs, RetScalar>
{
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, typename Lhs::Scalar &res)
  {
    ei_product_coeff_vectorized_dyn_selector<Lhs,Rhs>::run(row, col, lhs, rhs, res);
  }
};

/*******************
*** Packet path  ***
*******************/

template<int Index, typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl<RowMajor, Index, Lhs, Rhs, PacketScalar, LoadMode>
{
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    ei_product_packet_impl<RowMajor, Index-1, Lhs, Rhs, PacketScalar, LoadMode>::run(row, col, lhs, rhs, res);
    res =  ei_pmadd(ei_pset1(lhs.coeff(row, Index)), rhs.template packet<LoadMode>(Index, col), res);
  }
};

template<int Index, typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl<ColMajor, Index, Lhs, Rhs, PacketScalar, LoadMode>
{
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    ei_product_packet_impl<ColMajor, Index-1, Lhs, Rhs, PacketScalar, LoadMode>::run(row, col, lhs, rhs, res);
    res =  ei_pmadd(lhs.template packet<LoadMode>(row, Index), ei_pset1(rhs.coeff(Index, col)), res);
  }
};

template<typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl<RowMajor, 0, Lhs, Rhs, PacketScalar, LoadMode>
{
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    res = ei_pmul(ei_pset1(lhs.coeff(row, 0)),rhs.template packet<LoadMode>(0, col));
  }
};

template<typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl<ColMajor, 0, Lhs, Rhs, PacketScalar, LoadMode>
{
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar &res)
  {
    res = ei_pmul(lhs.template packet<LoadMode>(row, 0), ei_pset1(rhs.coeff(0, col)));
  }
};

template<typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl<RowMajor, Dynamic, Lhs, Rhs, PacketScalar, LoadMode>
{
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar& res)
  {
    ei_assert(lhs.cols()>0 && "you are using a non initialized matrix");
    res = ei_pmul(ei_pset1(lhs.coeff(row, 0)),rhs.template packet<LoadMode>(0, col));
      for(int i = 1; i < lhs.cols(); ++i)
        res =  ei_pmadd(ei_pset1(lhs.coeff(row, i)), rhs.template packet<LoadMode>(i, col), res);
  }
};

template<typename Lhs, typename Rhs, typename PacketScalar, int LoadMode>
struct ei_product_packet_impl<ColMajor, Dynamic, Lhs, Rhs, PacketScalar, LoadMode>
{
  EIGEN_STRONG_INLINE static void run(int row, int col, const Lhs& lhs, const Rhs& rhs, PacketScalar& res)
  {
    ei_assert(lhs.cols()>0 && "you are using a non initialized matrix");
    res = ei_pmul(lhs.template packet<LoadMode>(row, 0), ei_pset1(rhs.coeff(0, col)));
      for(int i = 1; i < lhs.cols(); ++i)
        res =  ei_pmadd(lhs.template packet<LoadMode>(row, i), ei_pset1(rhs.coeff(i, col)), res);
  }
};

/***************************************************************************
* Implementation of matrix base methods
***************************************************************************/

/** \returns the matrix product of \c *this and \a other.
  *
  * \note If instead of the matrix product you want the coefficient-wise product, see Cwise::operator*().
  *
  * \sa lazy(), operator*=(const MatrixBase&), Cwise::operator*()
  */
template<typename Derived>
template<typename OtherDerived>
inline const typename ProductReturnType<Derived,OtherDerived>::Type
MatrixBase<Derived>::operator*(const MatrixBase<OtherDerived> &other) const
{
  enum {
    ProductIsValid =  Derived::ColsAtCompileTime==Dynamic
                   || OtherDerived::RowsAtCompileTime==Dynamic
                   || int(Derived::ColsAtCompileTime)==int(OtherDerived::RowsAtCompileTime),
    AreVectors = Derived::IsVectorAtCompileTime && OtherDerived::IsVectorAtCompileTime,
    SameSizes = EIGEN_PREDICATE_SAME_MATRIX_SIZE(Derived,OtherDerived)
  };
  // note to the lost user:
  //    * for a dot product use: v1.dot(v2)
  //    * for a coeff-wise product use: v1.cwise()*v2
  EIGEN_STATIC_ASSERT(ProductIsValid || !(AreVectors && SameSizes),
    INVALID_VECTOR_VECTOR_PRODUCT__IF_YOU_WANTED_A_DOT_OR_COEFF_WISE_PRODUCT_YOU_MUST_USE_THE_EXPLICIT_FUNCTIONS)
  EIGEN_STATIC_ASSERT(ProductIsValid || !(SameSizes && !AreVectors),
    INVALID_MATRIX_PRODUCT__IF_YOU_WANTED_A_COEFF_WISE_PRODUCT_YOU_MUST_USE_THE_EXPLICIT_FUNCTION)
  EIGEN_STATIC_ASSERT(ProductIsValid || SameSizes, INVALID_MATRIX_PRODUCT)
  return typename ProductReturnType<Derived,OtherDerived>::Type(derived(), other.derived());
}



/** replaces \c *this by \c *this * \a other.
  *
  * \returns a reference to \c *this
  */
template<typename Derived>
template<typename OtherDerived>
inline Derived &
MatrixBase<Derived>::operator*=(const AnyMatrixBase<OtherDerived> &other)
{
  return derived() = derived() * other.derived();
}

#endif // EIGEN_PRODUCT_H
