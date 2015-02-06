// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_PERMUTATION_H
#define EIGEN_SPARSE_PERMUTATION_H

// This file implements sparse * permutation products

namespace Eigen { 

namespace internal {

template<typename PermutationType, typename MatrixType, int Side, bool Transposed>
struct traits<permut_sparsematrix_product_retval<PermutationType, MatrixType, Side, Transposed> >
{
  typedef typename remove_all<typename MatrixType::Nested>::type MatrixTypeNestedCleaned;
  typedef typename MatrixTypeNestedCleaned::Scalar Scalar;
  typedef typename MatrixTypeNestedCleaned::Index Index;
  enum {
    SrcStorageOrder = MatrixTypeNestedCleaned::Flags&RowMajorBit ? RowMajor : ColMajor,
    MoveOuter = SrcStorageOrder==RowMajor ? Side==OnTheLeft : Side==OnTheRight
  };

  typedef typename internal::conditional<MoveOuter,
        SparseMatrix<Scalar,SrcStorageOrder,Index>,
        SparseMatrix<Scalar,int(SrcStorageOrder)==RowMajor?ColMajor:RowMajor,Index> >::type ReturnType;
};

template<typename PermutationType, typename MatrixType, int Side, bool Transposed>
struct permut_sparsematrix_product_retval
 : public ReturnByValue<permut_sparsematrix_product_retval<PermutationType, MatrixType, Side, Transposed> >
{
    typedef typename remove_all<typename MatrixType::Nested>::type MatrixTypeNestedCleaned;
    typedef typename MatrixTypeNestedCleaned::Scalar Scalar;
    typedef typename MatrixTypeNestedCleaned::Index Index;

    enum {
      SrcStorageOrder = MatrixTypeNestedCleaned::Flags&RowMajorBit ? RowMajor : ColMajor,
      MoveOuter = SrcStorageOrder==RowMajor ? Side==OnTheLeft : Side==OnTheRight
    };

    permut_sparsematrix_product_retval(const PermutationType& perm, const MatrixType& matrix)
      : m_permutation(perm), m_matrix(matrix)
    {}

    inline int rows() const { return m_matrix.rows(); }
    inline int cols() const { return m_matrix.cols(); }

    template<typename Dest> inline void evalTo(Dest& dst) const
    {
      if(MoveOuter)
      {
        SparseMatrix<Scalar,SrcStorageOrder,Index> tmp(m_matrix.rows(), m_matrix.cols());
        Matrix<Index,Dynamic,1> sizes(m_matrix.outerSize());
        for(Index j=0; j<m_matrix.outerSize(); ++j)
        {
          Index jp = m_permutation.indices().coeff(j);
          sizes[((Side==OnTheLeft) ^ Transposed) ? jp : j] = m_matrix.innerVector(((Side==OnTheRight) ^ Transposed) ? jp : j).nonZeros();
        }
        tmp.reserve(sizes);
        for(Index j=0; j<m_matrix.outerSize(); ++j)
        {
          Index jp = m_permutation.indices().coeff(j);
          Index jsrc = ((Side==OnTheRight) ^ Transposed) ? jp : j;
          Index jdst = ((Side==OnTheLeft) ^ Transposed) ? jp : j;
          for(typename MatrixTypeNestedCleaned::InnerIterator it(m_matrix,jsrc); it; ++it)
            tmp.insertByOuterInner(jdst,it.index()) = it.value();
        }
        dst = tmp;
      }
      else
      {
        SparseMatrix<Scalar,int(SrcStorageOrder)==RowMajor?ColMajor:RowMajor,Index> tmp(m_matrix.rows(), m_matrix.cols());
        Matrix<Index,Dynamic,1> sizes(tmp.outerSize());
        sizes.setZero();
        PermutationMatrix<Dynamic,Dynamic,Index> perm;
        if((Side==OnTheLeft) ^ Transposed)
          perm = m_permutation;
        else
          perm = m_permutation.transpose();

        for(Index j=0; j<m_matrix.outerSize(); ++j)
          for(typename MatrixTypeNestedCleaned::InnerIterator it(m_matrix,j); it; ++it)
            sizes[perm.indices().coeff(it.index())]++;
        tmp.reserve(sizes);
        for(Index j=0; j<m_matrix.outerSize(); ++j)
          for(typename MatrixTypeNestedCleaned::InnerIterator it(m_matrix,j); it; ++it)
            tmp.insertByOuterInner(perm.indices().coeff(it.index()),j) = it.value();
        dst = tmp;
      }
    }

  protected:
    const PermutationType& m_permutation;
    typename MatrixType::Nested m_matrix;
};

}

namespace internal {

template <int ProductTag> struct product_promote_storage_type<Sparse,             PermutationStorage, ProductTag> { typedef Sparse ret; };
template <int ProductTag> struct product_promote_storage_type<PermutationStorage, Sparse,             ProductTag> { typedef Sparse ret; };
  
// TODO, the following need cleaning, this is just a copy-past of the dense case  

template<typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs, Rhs, PermutationShape, SparseShape, ProductTag>
{
  template<typename Dest>
  static void evalTo(Dest& dst, const Lhs& lhs, const Rhs& rhs)
  {
    permut_sparsematrix_product_retval<Lhs, Rhs, OnTheLeft, false> pmpr(lhs, rhs);
    pmpr.evalTo(dst);
  }
};

template<typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs, Rhs, SparseShape, PermutationShape, ProductTag>
{
  template<typename Dest>
  static void evalTo(Dest& dst, const Lhs& lhs, const Rhs& rhs)
  {
    permut_sparsematrix_product_retval<Rhs, Lhs, OnTheRight, false> pmpr(rhs, lhs);
    pmpr.evalTo(dst);
  }
};

template<typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Transpose<Lhs>, Rhs, PermutationShape, SparseShape, ProductTag>
{
  template<typename Dest>
  static void evalTo(Dest& dst, const Transpose<Lhs>& lhs, const Rhs& rhs)
  {
    permut_sparsematrix_product_retval<Lhs, Rhs, OnTheLeft, true> pmpr(lhs.nestedPermutation(), rhs);
    pmpr.evalTo(dst);
  }
};

template<typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs, Transpose<Rhs>, SparseShape, PermutationShape, ProductTag>
{
  template<typename Dest>
  static void evalTo(Dest& dst, const Lhs& lhs, const Transpose<Rhs>& rhs)
  {
    permut_sparsematrix_product_retval<Rhs, Lhs, OnTheRight, true> pmpr(rhs.nestedPermutation(), lhs);
    pmpr.evalTo(dst);
  }
};

// TODO, the following two overloads are only needed to define the right temporary type through 
// typename traits<permut_sparsematrix_product_retval<Rhs,Lhs,OnTheRight,false> >::ReturnType
// while it should be correctly handled by traits<Product<> >::PlainObject

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, ProductTag, PermutationShape, SparseShape, typename traits<Lhs>::Scalar, typename traits<Rhs>::Scalar> 
  : public evaluator<typename traits<permut_sparsematrix_product_retval<Lhs,Rhs,OnTheRight,false> >::ReturnType>::type
{
  typedef Product<Lhs, Rhs, DefaultProduct> XprType;
  typedef typename traits<permut_sparsematrix_product_retval<Lhs,Rhs,OnTheRight,false> >::ReturnType PlainObject;
  typedef typename evaluator<PlainObject>::type Base;

  explicit product_evaluator(const XprType& xpr)
    : m_result(xpr.rows(), xpr.cols())
  {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<Lhs, Rhs, PermutationShape, SparseShape, ProductTag>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }
  
protected:  
  PlainObject m_result;
};

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, ProductTag, SparseShape, PermutationShape, typename traits<Lhs>::Scalar, typename traits<Rhs>::Scalar> 
  : public evaluator<typename traits<permut_sparsematrix_product_retval<Rhs,Lhs,OnTheRight,false> >::ReturnType>::type
{
  typedef Product<Lhs, Rhs, DefaultProduct> XprType;
  typedef typename traits<permut_sparsematrix_product_retval<Rhs,Lhs,OnTheRight,false> >::ReturnType PlainObject;
  typedef typename evaluator<PlainObject>::type Base;

  explicit product_evaluator(const XprType& xpr)
    : m_result(xpr.rows(), xpr.cols())
  {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<Lhs, Rhs, SparseShape, PermutationShape, ProductTag>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }
  
protected:  
  PlainObject m_result;
};

} // end namespace internal

/** \returns the matrix with the permutation applied to the columns
  */
template<typename SparseDerived, typename PermDerived>
inline const Product<SparseDerived, PermDerived>
operator*(const SparseMatrixBase<SparseDerived>& matrix, const PermutationBase<PermDerived>& perm)
{ return Product<SparseDerived, PermDerived>(matrix.derived(), perm.derived()); }

/** \returns the matrix with the permutation applied to the rows
  */
template<typename SparseDerived, typename PermDerived>
inline const Product<PermDerived, SparseDerived>
operator*( const PermutationBase<PermDerived>& perm, const SparseMatrixBase<SparseDerived>& matrix)
{ return  Product<PermDerived, SparseDerived>(perm.derived(), matrix.derived()); }


// TODO, the following specializations should not be needed as Transpose<Permutation*> should be a PermutationBase.
/** \returns the matrix with the inverse permutation applied to the columns.
  */
template<typename SparseDerived, typename PermDerived>
inline const Product<SparseDerived, Transpose<PermutationBase<PermDerived> > >
operator*(const SparseMatrixBase<SparseDerived>& matrix, const Transpose<PermutationBase<PermDerived> >& tperm)
{
  return Product<SparseDerived, Transpose<PermutationBase<PermDerived> > >(matrix.derived(), tperm);
}

/** \returns the matrix with the inverse permutation applied to the rows.
  */
template<typename SparseDerived, typename PermDerived>
inline const Product<Transpose<PermutationBase<PermDerived> >, SparseDerived>
operator*(const Transpose<PermutationBase<PermDerived> >& tperm, const SparseMatrixBase<SparseDerived>& matrix)
{
  return Product<Transpose<PermutationBase<PermDerived> >, SparseDerived>(tperm, matrix.derived());
}

} // end namespace Eigen

#endif // EIGEN_SPARSE_SELFADJOINTVIEW_H
