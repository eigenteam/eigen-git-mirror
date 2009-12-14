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
    Depth = EIGEN_ENUM_MIN(Lhs::ColsAtCompileTime,Rhs::RowsAtCompileTime)
  };

  // the splitting into different lines of code here, introducing the _select enums and the typedef below,
  // is to work around an internal compiler error with gcc 4.1 and 4.2.
private:
  enum {
    rows_select = Rows >=EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD ? Large : (Rows==1   ? 1 : Small),
    cols_select = Cols >=EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD ? Large : (Cols==1   ? 1 : Small),
    depth_select = Depth>=EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD ? Large : (Depth==1  ? 1 : Small)
  };
  typedef ei_product_type_selector<rows_select, cols_select, depth_select> product_type_selector;

public:
  enum {
    value = product_type_selector::ret
  };
};

/* The following allows to select the kind of product at compile time
 * based on the three dimensions of the product.
 * This is a compile time mapping from {1,Small,Large}^3 -> {product types} */
// FIXME I'm not sure the current mapping is the ideal one.
template<int Rows, int Cols>  struct ei_product_type_selector<Rows, Cols, 1>      { enum { ret = OuterProduct }; };
template<int Depth>           struct ei_product_type_selector<1,    1,    Depth>  { enum { ret = InnerProduct }; };
template<>                    struct ei_product_type_selector<1,    1,    1>      { enum { ret = InnerProduct }; };
template<>                    struct ei_product_type_selector<Small,1,    Small>  { enum { ret = UnrolledProduct }; };
template<>                    struct ei_product_type_selector<1,    Small,Small>  { enum { ret = UnrolledProduct }; };
template<>                    struct ei_product_type_selector<Small,Small,Small>  { enum { ret = UnrolledProduct }; };
template<>                    struct ei_product_type_selector<1,    Large,Small>  { enum { ret = GemvProduct }; };
template<>                    struct ei_product_type_selector<1,    Large,Large>  { enum { ret = GemvProduct }; };
template<>                    struct ei_product_type_selector<1,    Small,Large>  { enum { ret = GemvProduct }; };
template<>                    struct ei_product_type_selector<Large,1,    Small>  { enum { ret = GemvProduct }; };
template<>                    struct ei_product_type_selector<Large,1,    Large>  { enum { ret = GemvProduct }; };
template<>                    struct ei_product_type_selector<Small,1,    Large>  { enum { ret = GemvProduct }; };
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
  typedef GeneralProduct<LhsNested, RhsNested, UnrolledProduct> Type;
};


/***********************************************************************
*  Implementation of Inner Vector Vector Product
***********************************************************************/

// FIXME : maybe the "inner product" could return a Scalar
// instead of a 1x1 matrix ??
// Pro: more natural for the user
// Cons: this could be a problem if in a meta unrolled algorithm a matrix-matrix
// product ends up to a row-vector times col-vector product... To tackle this use
// case, we could have a specialization for Block<MatrixType,1,1> with: operator=(Scalar x);

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

    GeneralProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {
      EIGEN_STATIC_ASSERT((ei_is_same_type<typename Lhs::RealScalar, typename Rhs::RealScalar>::ret),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
    }

    EIGEN_STRONG_INLINE Scalar value() const
    {
      return (m_lhs.transpose().cwise()*m_rhs).sum();
    }

    template<typename Dest> void scaleAndAddTo(Dest& dst, Scalar alpha) const
    {
      ei_assert(dst.rows()==1 && dst.cols()==1);
      dst.coeffRef(0,0) += alpha * value();
    }

    EIGEN_STRONG_INLINE Scalar coeff(int, int) const { return value(); }

    EIGEN_STRONG_INLINE Scalar coeff(int) const { return value(); }
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

    GeneralProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {
      EIGEN_STATIC_ASSERT((ei_is_same_type<typename Lhs::RealScalar, typename Rhs::RealScalar>::ret),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
    }

    template<typename Dest> void scaleAndAddTo(Dest& dest, Scalar alpha) const
    {
      ei_outer_product_selector<(int(Dest::Flags)&RowMajorBit) ? RowMajor : ColMajor>::run(*this, dest, alpha);
    }
};

template<> struct ei_outer_product_selector<ColMajor> {
  template<typename ProductType, typename Dest>
  EIGEN_DONT_INLINE static void run(const ProductType& prod, Dest& dest, typename ProductType::Scalar alpha) {
    // FIXME make sure lhs is sequentially stored
    // FIXME not very good if rhs is real and lhs complex while alpha is real too
    const int cols = dest.cols();
    for (int j=0; j<cols; ++j)
      dest.col(j) += (alpha * prod.rhs().coeff(j)) * prod.lhs();
  }
};

template<> struct ei_outer_product_selector<RowMajor> {
  template<typename ProductType, typename Dest>
  EIGEN_DONT_INLINE static void run(const ProductType& prod, Dest& dest, typename ProductType::Scalar alpha) {
    // FIXME make sure rhs is sequentially stored
    // FIXME not very good if lhs is real and rhs complex while alpha is real too
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

    GeneralProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {
      EIGEN_STATIC_ASSERT((ei_is_same_type<typename Lhs::Scalar, typename Rhs::Scalar>::ret),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
    }

    enum { Side = Lhs::IsVectorAtCompileTime ? OnTheLeft : OnTheRight };
    typedef typename ei_meta_if<int(Side)==OnTheRight,_LhsNested,_RhsNested>::ret MatrixType;

    template<typename Dest> void scaleAndAddTo(Dest& dst, Scalar alpha) const
    {
      ei_assert(m_lhs.rows() == dst.rows() && m_rhs.cols() == dst.cols());
      ei_gemv_selector<Side,(int(MatrixType::Flags)&RowMajorBit) ? RowMajor : ColMajor,
                       bool(ei_blas_traits<MatrixType>::ActualAccess)>::run(*this, dst, alpha);
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

#endif // EIGEN_PRODUCT_H
