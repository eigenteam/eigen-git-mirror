// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_PARTIALLU_H
#define EIGEN_PARTIALLU_H

/** \ingroup LU_Module
  *
  * \class PartialLU
  *
  * \brief LU decomposition of a matrix with partial pivoting, and related features
  *
  * \param MatrixType the type of the matrix of which we are computing the LU decomposition
  *
  * This class represents a LU decomposition of a \b square \b invertible matrix, with partial pivoting: the matrix A
  * is decomposed as A = PLU where L is unit-lower-triangular, U is upper-triangular, and P
  * is a permutation matrix.
  *
  * Typically, partial pivoting LU decomposition is only considered numerically stable for square invertible matrices.
  * So in this class, we plainly require that and take advantage of that to do some simplifications and optimizations.
  * This class will assert that the matrix is square, but it won't (actually it can't) check that the matrix is invertible:
  * it is your task to check that you only use this decomposition on invertible matrices.
  *
  * The guaranteed safe alternative, working for all matrices, is the full pivoting LU decomposition, provided by class LU.
  *
  * This is \b not a rank-revealing LU decomposition. Many features are intentionally absent from this class,
  * such as rank computation. If you need these features, use class LU.
  *
  * This LU decomposition is suitable to invert invertible matrices. It is what MatrixBase::inverse() uses. On the other hand,
  * it is \b not suitable to determine whether a given matrix is invertible.
  *
  * The data of the LU decomposition can be directly accessed through the methods matrixLU(), permutationP().
  *
  * \sa MatrixBase::partialLu(), MatrixBase::determinant(), MatrixBase::inverse(), MatrixBase::computeInverse(), class LU
  */
template<typename MatrixType> class PartialLU
{
  public:

    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef Matrix<int, 1, MatrixType::ColsAtCompileTime> IntRowVectorType;
    typedef Matrix<int, MatrixType::RowsAtCompileTime, 1> IntColVectorType;
    typedef Matrix<Scalar, 1, MatrixType::ColsAtCompileTime> RowVectorType;
    typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> ColVectorType;

    enum { MaxSmallDimAtCompileTime = EIGEN_ENUM_MIN(
             MatrixType::MaxColsAtCompileTime,
             MatrixType::MaxRowsAtCompileTime)
    };

    /** Constructor.
      *
      * \param matrix the matrix of which to compute the LU decomposition.
      *
      * \warning The matrix should have full rank (e.g. if it's square, it should be invertible).
      * If you need to deal with non-full rank, use class LU instead.
      */
    PartialLU(const MatrixType& matrix);

    /** \returns the LU decomposition matrix: the upper-triangular part is U, the
      * unit-lower-triangular part is L (at least for square matrices; in the non-square
      * case, special care is needed, see the documentation of class LU).
      *
      * \sa matrixL(), matrixU()
      */
    inline const MatrixType& matrixLU() const
    {
      return m_lu;
    }

    /** \returns a vector of integers, whose size is the number of rows of the matrix being decomposed,
      * representing the P permutation i.e. the permutation of the rows. For its precise meaning,
      * see the examples given in the documentation of class LU.
      */
    inline const IntColVectorType& permutationP() const
    {
      return m_p;
    }

    /** This method finds the solution x to the equation Ax=b, where A is the matrix of which
      * *this is the LU decomposition. Since if this partial pivoting decomposition the matrix is assumed
      * to have full rank, such a solution is assumed to exist and to be unique.
      *
      * \warning Again, if your matrix may not have full rank, use class LU instead. See LU::solve().
      *
      * \param b the right-hand-side of the equation to solve. Can be a vector or a matrix,
      *          the only requirement in order for the equation to make sense is that
      *          b.rows()==A.rows(), where A is the matrix of which *this is the LU decomposition.
      * \param result a pointer to the vector or matrix in which to store the solution, if any exists.
      *          Resized if necessary, so that result->rows()==A.cols() and result->cols()==b.cols().
      *          If no solution exists, *result is left with undefined coefficients.
      *
      * Example: \include PartialLU_solve.cpp
      * Output: \verbinclude PartialLU_solve.out
      *
      * \sa MatrixBase::solveTriangular(), inverse(), computeInverse()
      */
    template<typename OtherDerived, typename ResultType>
    void solve(const MatrixBase<OtherDerived>& b, ResultType *result) const;

    /** \returns the determinant of the matrix of which
      * *this is the LU decomposition. It has only linear complexity
      * (that is, O(n) where n is the dimension of the square matrix)
      * as the LU decomposition has already been computed.
      *
      * \note For fixed-size matrices of size up to 4, MatrixBase::determinant() offers
      *       optimized paths.
      *
      * \warning a determinant can be very big or small, so for matrices
      * of large enough dimension, there is a risk of overflow/underflow.
      *
      * \sa MatrixBase::determinant()
      */
    typename ei_traits<MatrixType>::Scalar determinant() const;

    /** Computes the inverse of the matrix of which *this is the LU decomposition.
      *
      * \param result a pointer to the matrix into which to store the inverse. Resized if needed.
      *
      * \warning The matrix being decomposed here is assumed to be invertible. If you need to check for
      *          invertibility, use class LU instead.
      *
      * \sa MatrixBase::computeInverse(), inverse()
      */
    inline void computeInverse(MatrixType *result) const
    {
      solve(MatrixType::Identity(m_lu.rows(), m_lu.cols()), result);
    }

    /** \returns the inverse of the matrix of which *this is the LU decomposition.
      *
      * \warning The matrix being decomposed here is assumed to be invertible. If you need to check for
      *          invertibility, use class LU instead.
      *
      * \sa computeInverse(), MatrixBase::inverse()
      */
    inline MatrixType inverse() const
    {
      MatrixType result;
      computeInverse(&result);
      return result;
    }

  protected:
    const MatrixType& m_originalMatrix;
    MatrixType m_lu;
    IntColVectorType m_p;
    int m_det_p;
};

template<typename MatrixType>
PartialLU<MatrixType>::PartialLU(const MatrixType& matrix)
  : m_originalMatrix(matrix),
    m_lu(matrix),
    m_p(matrix.rows())
{
  ei_assert(matrix.rows() == matrix.cols() && "PartialLU is only for square (and moreover invertible) matrices");
  const int size = matrix.rows();

  IntColVectorType rows_transpositions(size);
  int number_of_transpositions = 0;

  for(int k = 0; k < size; ++k)
  {
    int row_of_biggest_in_col;
    m_lu.block(k,k,size-k,1).cwise().abs().maxCoeff(&row_of_biggest_in_col);
    row_of_biggest_in_col += k;

    rows_transpositions.coeffRef(k) = row_of_biggest_in_col;

    if(k != row_of_biggest_in_col) {
      m_lu.row(k).swap(m_lu.row(row_of_biggest_in_col));
      ++number_of_transpositions;
    }

    if(k<size-1) {
      m_lu.col(k).end(size-k-1) /= m_lu.coeff(k,k);
      for(int col = k + 1; col < size; ++col)
        m_lu.col(col).end(size-k-1) -= m_lu.col(k).end(size-k-1) * m_lu.coeff(k,col); 
    }
  }

  for(int k = 0; k < size; ++k) m_p.coeffRef(k) = k;
  for(int k = size-1; k >= 0; --k)
    std::swap(m_p.coeffRef(k), m_p.coeffRef(rows_transpositions.coeff(k)));

  m_det_p = (number_of_transpositions%2) ? -1 : 1;
}

template<typename MatrixType>
typename ei_traits<MatrixType>::Scalar PartialLU<MatrixType>::determinant() const
{
  return Scalar(m_det_p) * m_lu.diagonal().prod();
}

template<typename MatrixType>
template<typename OtherDerived, typename ResultType>
void PartialLU<MatrixType>::solve(
  const MatrixBase<OtherDerived>& b,
  ResultType *result
) const
{
  /* The decomposition PA = LU can be rewritten as A = P^{-1} L U.
   * So we proceed as follows:
   * Step 1: compute c = Pb.
   * Step 2: replace c by the solution x to Lx = c.
   * Step 3: replace c by the solution x to Ux = c.
   */

  const int size = m_lu.rows();
  ei_assert(b.rows() == size);

  result->resize(size, b.cols());

  // Step 1
  for(int i = 0; i < size; ++i) result->row(m_p.coeff(i)) = b.row(i);

  // Step 2
  m_lu.template marked<UnitLowerTriangular>()
      .solveTriangularInPlace(*result);

  // Step 3
  m_lu.template marked<UpperTriangular>()
      .solveTriangularInPlace(*result);
}

/** \lu_module
  *
  * \return the LU decomposition of \c *this.
  *
  * \sa class LU
  */
template<typename Derived>
inline const PartialLU<typename MatrixBase<Derived>::PlainMatrixType>
MatrixBase<Derived>::partialLu() const
{
  return PartialLU<PlainMatrixType>(eval());
}

#endif // EIGEN_PARTIALLU_H
