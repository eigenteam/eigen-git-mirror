// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Kolja Brix <brix@igpm.rwth-aachen.de>
// Copyright (C) 2011 Andreas Platen <andiplaten@gmx.de>
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef KRONECKER_TENSOR_PRODUCT_H
#define KRONECKER_TENSOR_PRODUCT_H

#define EIGEN_SIZE_PRODUCT(a,b) (!((int)a && (int)b) ? 0 \
                : ((int)a == Dynamic || (int)b == Dynamic) ? Dynamic \
                : (int)a * (int)b)

namespace Eigen { 

namespace internal {

template<typename _Lhs, typename _Rhs>
struct traits<KroneckerProduct<_Lhs,_Rhs> >
{
  typedef MatrixXpr XprKind;
  typedef typename remove_all<_Lhs>::type Lhs;
  typedef typename remove_all<_Rhs>::type Rhs;
  typedef typename scalar_product_traits<typename Lhs::Scalar, typename Rhs::Scalar>::ReturnType Scalar;
  typedef Dense StorageKind;
  typedef typename promote_index_type<typename Lhs::Index, typename Rhs::Index>::type Index;

  enum {
    RowsAtCompileTime = EIGEN_SIZE_PRODUCT(traits<Lhs>::RowsAtCompileTime, traits<Rhs>::RowsAtCompileTime),
    ColsAtCompileTime = EIGEN_SIZE_PRODUCT(traits<Lhs>::ColsAtCompileTime, traits<Rhs>::ColsAtCompileTime),
    MaxRowsAtCompileTime = EIGEN_SIZE_PRODUCT(traits<Lhs>::MaxRowsAtCompileTime, traits<Rhs>::MaxRowsAtCompileTime),
    MaxColsAtCompileTime = EIGEN_SIZE_PRODUCT(traits<Lhs>::MaxColsAtCompileTime, traits<Rhs>::MaxColsAtCompileTime),
    Flags = (MaxRowsAtCompileTime==1 ? RowMajorBit : 0)
          | EvalBeforeNestingBit | EvalBeforeAssigningBit | NestByRefBit,
    CoeffReadCost = Lhs::CoeffReadCost + Rhs::CoeffReadCost + NumTraits<Scalar>::MulCost
  };
};

template<typename _Lhs, typename _Rhs>
struct traits<KroneckerProductSparse<_Lhs,_Rhs> >
{
  typedef MatrixXpr XprKind;
  typedef typename remove_all<_Lhs>::type Lhs;
  typedef typename remove_all<_Rhs>::type Rhs;
  typedef typename scalar_product_traits<typename Lhs::Scalar, typename Rhs::Scalar>::ReturnType Scalar;
  typedef Sparse StorageKind;
  typedef typename promote_index_type<typename Lhs::Index, typename Rhs::Index>::type Index;

  enum {
    LhsFlags = Lhs::Flags,
    RhsFlags = Rhs::Flags,

    RowsAtCompileTime = EIGEN_SIZE_PRODUCT(traits<Lhs>::RowsAtCompileTime, traits<Rhs>::RowsAtCompileTime),
    ColsAtCompileTime = EIGEN_SIZE_PRODUCT(traits<Lhs>::ColsAtCompileTime, traits<Rhs>::ColsAtCompileTime),
    MaxRowsAtCompileTime = EIGEN_SIZE_PRODUCT(traits<Lhs>::MaxRowsAtCompileTime, traits<Rhs>::MaxRowsAtCompileTime),
    MaxColsAtCompileTime = EIGEN_SIZE_PRODUCT(traits<Lhs>::MaxColsAtCompileTime, traits<Rhs>::MaxColsAtCompileTime),

    EvalToRowMajor = (LhsFlags & RhsFlags & RowMajorBit),
    RemovedBits = ~(EvalToRowMajor ? 0 : RowMajorBit),

    Flags = ((LhsFlags | RhsFlags) & HereditaryBits & RemovedBits)
          | EvalBeforeNestingBit | EvalBeforeAssigningBit,
    CoeffReadCost = Dynamic
  };
};

} // end namespace internal

/*!
 * \brief Kronecker tensor product helper class for dense matrices
 *
 * This class is the return value of kroneckerProduct(MatrixBase,
 * MatrixBase). Use the function rather than construct this class
 * directly to avoid specifying template prarameters.
 *
 * \tparam Lhs  Type of the left-hand side, a matrix expression.
 * \tparam Rhs  Type of the rignt-hand side, a matrix expression.
 */
template<typename Lhs, typename Rhs>
class KroneckerProduct : public MatrixBase<KroneckerProduct<Lhs,Rhs> >
{
  public:
    typedef MatrixBase<KroneckerProduct> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(KroneckerProduct)

    /*! \brief Constructor. */
    KroneckerProduct(const Lhs& A, const Rhs& B)
      : m_A(A), m_B(B)
    {}

    /*! \brief Evaluate the Kronecker tensor product. */
    template<typename Dest> void evalTo(Dest& dst) const;
    
    inline Index rows() const { return m_A.rows() * m_B.rows(); }
    inline Index cols() const { return m_A.cols() * m_B.cols(); }

    typename Base::CoeffReturnType coeff(Index row, Index col) const
    {
      return m_A.coeff(row / m_A.cols(), col / m_A.rows()) *
             m_B.coeff(row % m_A.cols(), col % m_A.rows());
    }

    typename Base::CoeffReturnType coeff(Index i) const
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(KroneckerProduct);
      return m_A.coeff(i / m_A.size()) * m_B.coeff(i % m_A.size());
    }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    struct Unusable {};
    Unusable& coeffRef(Index) { return *reinterpret_cast<Unusable*>(this); }
    Unusable& coeffRef(Index,Index) { return *reinterpret_cast<Unusable*>(this); }
#endif

  private:
    typename Lhs::Nested m_A;
    typename Rhs::Nested m_B;
};

/*!
 * \brief Kronecker tensor product helper class for sparse matrices
 *
 * If at least one of the operands is a sparse matrix expression,
 * then this class is returned and evaluates into a sparse matrix.
 *
 * This class is the return value of kroneckerProduct(EigenBase,
 * EigenBase). Use the function rather than construct this class
 * directly to avoid specifying template prarameters.
 *
 * \tparam Lhs  Type of the left-hand side, a matrix expression.
 * \tparam Rhs  Type of the rignt-hand side, a matrix expression.
 */
template<typename Lhs, typename Rhs>
class KroneckerProductSparse : public SparseMatrixBase<KroneckerProductSparse<Lhs,Rhs> >
{
  public:
    typedef SparseMatrixBase<KroneckerProductSparse> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(KroneckerProductSparse)

    /*! \brief Constructor. */
    KroneckerProductSparse(const Lhs& A, const Rhs& B)
      : m_A(A), m_B(B)
    {}

    /*! \brief Evaluate the Kronecker tensor product. */
    template<typename Dest> void evalTo(Dest& dst) const;
    
    inline Index rows() const { return m_A.rows() * m_B.rows(); }
    inline Index cols() const { return m_A.cols() * m_B.cols(); }

  private:
    typename Lhs::Nested m_A;
    typename Rhs::Nested m_B;
};

template<typename Lhs, typename Rhs>
template<typename Dest>
void KroneckerProduct<Lhs,Rhs>::evalTo(Dest& dst) const
{
  const int BlockRows = Rhs::RowsAtCompileTime,
            BlockCols = Rhs::ColsAtCompileTime;
  const Index Br = m_B.rows(),
              Bc = m_B.cols();
  for (Index i=0; i < m_A.rows(); ++i)
    for (Index j=0; j < m_A.cols(); ++j)
      Block<Dest,BlockRows,BlockCols>(dst,i*Br,j*Bc,Br,Bc) = m_A.coeff(i,j) * m_B;
}

template<typename Lhs, typename Rhs>
template<typename Dest>
void KroneckerProductSparse<Lhs,Rhs>::evalTo(Dest& dst) const
{
  const Index Br = m_B.rows(),
              Bc = m_B.cols();
  dst.resize(rows(),cols());
  dst.resizeNonZeros(0);
  dst.reserve(m_A.nonZeros() * m_B.nonZeros());

  for (Index kA=0; kA < m_A.outerSize(); ++kA)
  {
    for (Index kB=0; kB < m_B.outerSize(); ++kB)
    {
      for (typename Lhs::InnerIterator itA(m_A,kA); itA; ++itA)
      {
        for (typename Rhs::InnerIterator itB(m_B,kB); itB; ++itB)
        {
          const Index i = itA.row() * Br + itB.row(),
                      j = itA.col() * Bc + itB.col();
          dst.insert(i,j) = itA.value() * itB.value();
        }
      }
    }
  }
}

/*!
 * Computes Kronecker tensor product of two dense matrices
 *
 * \param a  Dense matrix a
 * \param b  Dense matrix b
 * \return   Kronecker tensor product of a and b
 */
template<typename A, typename B>
KroneckerProduct<A,B> kroneckerProduct(const MatrixBase<A>& a, const MatrixBase<B>& b)
{
  return KroneckerProduct<A, B>(a.derived(), b.derived());
}

/*!
 * Computes Kronecker tensor product of two matrices, at least one of
 * which is sparse.
 *
 * \param a  Dense/sparse matrix a
 * \param b  Dense/sparse matrix b
 * \return   Kronecker tensor product of a and b, stored in a sparse
 *           matrix
 */
template<typename A, typename B>
KroneckerProductSparse<A,B> kroneckerProduct(const EigenBase<A>& a, const EigenBase<B>& b)
{
  return KroneckerProductSparse<A,B>(a.derived(), b.derived());
}

template<typename Derived>
template<typename Lhs, typename Rhs>
Derived& MatrixBase<Derived>::lazyAssign(const KroneckerProduct<Lhs,Rhs>& other)
{
  other.evalTo(derived());
  return derived();
}

template<typename Derived>
template<typename Lhs, typename Rhs>
Derived& SparseMatrixBase<Derived>::operator=(const KroneckerProductSparse<Lhs,Rhs>& product)
{
  product.evalTo(derived());
  return derived();
}

} // end namespace Eigen

#endif // KRONECKER_TENSOR_PRODUCT_H
