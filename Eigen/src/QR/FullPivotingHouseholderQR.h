// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_FULLPIVOTINGHOUSEHOLDERQR_H
#define EIGEN_FULLPIVOTINGHOUSEHOLDERQR_H

/** \ingroup QR_Module
  * \nonstableyet
  *
  * \class FullPivotingHouseholderQR
  *
  * \brief Householder rank-revealing QR decomposition of a matrix with full pivoting
  *
  * \param MatrixType the type of the matrix of which we are computing the QR decomposition
  *
  * This class performs a rank-revealing QR decomposition using Householder transformations.
  *
  * This decomposition performs a very prudent full pivoting in order to be rank-revealing and achieve optimal
  * numerical stability. The trade-off is that it is slower than HouseholderQR and ColPivotingHouseholderQR.
  *
  * \sa MatrixBase::fullPivotingHouseholderQr()
  */
template<typename MatrixType> class FullPivotingHouseholderQR
{
  public:
    
    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      Options = MatrixType::Options,
      DiagSizeAtCompileTime = EIGEN_ENUM_MIN(ColsAtCompileTime,RowsAtCompileTime)
    };
    
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> MatrixQType;
    typedef Matrix<Scalar, DiagSizeAtCompileTime, 1> HCoeffsType;
    typedef Matrix<int, 1, ColsAtCompileTime> IntRowVectorType;
    typedef Matrix<int, RowsAtCompileTime, 1> IntColVectorType;
    typedef Matrix<Scalar, 1, ColsAtCompileTime> RowVectorType;
    typedef Matrix<Scalar, RowsAtCompileTime, 1> ColVectorType;

    /** \brief Default Constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via FullPivotingHouseholderQR::compute(const MatrixType&).
      */
    FullPivotingHouseholderQR() : m_qr(), m_hCoeffs(), m_isInitialized(false) {}

    FullPivotingHouseholderQR(const MatrixType& matrix)
      : m_qr(matrix.rows(), matrix.cols()),
        m_hCoeffs(std::min(matrix.rows(),matrix.cols())),
        m_isInitialized(false)
    {
      compute(matrix);
    }

    /** This method finds a solution x to the equation Ax=b, where A is the matrix of which
      * *this is the QR decomposition, if any exists.
      *
      * \returns \c true if a solution exists, \c false if no solution exists.
      *
      * \param b the right-hand-side of the equation to solve.
      *
      * \param result a pointer to the vector/matrix in which to store the solution, if any exists.
      *          Resized if necessary, so that result->rows()==A.cols() and result->cols()==b.cols().
      *          If no solution exists, *result is left with undefined coefficients.
      *
      * \note The case where b is a matrix is not yet implemented. Also, this
      *       code is space inefficient.
      *
      * Example: \include FullPivotingHouseholderQR_solve.cpp
      * Output: \verbinclude FullPivotingHouseholderQR_solve.out
      */
    template<typename OtherDerived, typename ResultType>
    bool solve(const MatrixBase<OtherDerived>& b, ResultType *result) const;

    MatrixType matrixQ(void) const;

    /** \returns a reference to the matrix where the Householder QR decomposition is stored
      */
    const MatrixType& matrixQR() const { return m_qr; }

    FullPivotingHouseholderQR& compute(const MatrixType& matrix);
    
    const IntRowVectorType& colsPermutation() const
    {
      ei_assert(m_isInitialized && "FullPivotingHouseholderQR is not initialized.");
      return m_cols_permutation;
    }
    
    const IntColVectorType& rowsTranspositions() const
    {
      ei_assert(m_isInitialized && "FullPivotingHouseholderQR is not initialized.");
      return m_rows_transpositions;
    }

    /** \returns the absolute value of the determinant of the matrix of which
      * *this is the QR decomposition. It has only linear complexity
      * (that is, O(n) where n is the dimension of the square matrix)
      * as the QR decomposition has already been computed.
      *
      * \note This is only for square matrices.
      *
      * \warning a determinant can be very big or small, so for matrices
      * of large enough dimension, there is a risk of overflow/underflow.
      * One way to work around that is to use logAbsDeterminant() instead.
      *
      * \sa logAbsDeterminant(), MatrixBase::determinant()
      */
    typename MatrixType::RealScalar absDeterminant() const;

    /** \returns the natural log of the absolute value of the determinant of the matrix of which
      * *this is the QR decomposition. It has only linear complexity
      * (that is, O(n) where n is the dimension of the square matrix)
      * as the QR decomposition has already been computed.
      *
      * \note This is only for square matrices.
      *
      * \note This method is useful to work around the risk of overflow/underflow that's inherent
      * to determinant computation.
      *
      * \sa absDeterminant(), MatrixBase::determinant()
      */
    typename MatrixType::RealScalar logAbsDeterminant() const;
    
    /** \returns the rank of the matrix of which *this is the QR decomposition.
      *
      * \note This is computed at the time of the construction of the QR decomposition. This
      *       method does not perform any further computation.
      */
    inline int rank() const
    {
      ei_assert(m_isInitialized && "FullPivotingHouseholderQR is not initialized.");
      return m_rank;
    }

    /** \returns the dimension of the kernel of the matrix of which *this is the QR decomposition.
      *
      * \note Since the rank is computed at the time of the construction of the QR decomposition, this
      *       method almost does not perform any further computation.
      */
    inline int dimensionOfKernel() const
    {
      ei_assert(m_isInitialized && "FullPivotingHouseholderQR is not initialized.");
      return m_qr.cols() - m_rank;
    }

    /** \returns true if the matrix of which *this is the QR decomposition represents an injective
      *          linear map, i.e. has trivial kernel; false otherwise.
      *
      * \note Since the rank is computed at the time of the construction of the QR decomposition, this
      *       method almost does not perform any further computation.
      */
    inline bool isInjective() const
    {
      ei_assert(m_isInitialized && "FullPivotingHouseholderQR is not initialized.");
      return m_rank == m_qr.cols();
    }

    /** \returns true if the matrix of which *this is the QR decomposition represents a surjective
      *          linear map; false otherwise.
      *
      * \note Since the rank is computed at the time of the construction of the QR decomposition, this
      *       method almost does not perform any further computation.
      */
    inline bool isSurjective() const
    {
      ei_assert(m_isInitialized && "FullPivotingHouseholderQR is not initialized.");
      return m_rank == m_qr.rows();
    }

    /** \returns true if the matrix of which *this is the QR decomposition is invertible.
      *
      * \note Since the rank is computed at the time of the construction of the QR decomposition, this
      *       method almost does not perform any further computation.
      */
    inline bool isInvertible() const
    {
      ei_assert(m_isInitialized && "FullPivotingHouseholderQR is not initialized.");
      return isInjective() && isSurjective();
    }

    /** Computes the inverse of the matrix of which *this is the QR decomposition.
      *
      * \param result a pointer to the matrix into which to store the inverse. Resized if needed.
      *
      * \note If this matrix is not invertible, *result is left with undefined coefficients.
      *       Use isInvertible() to first determine whether this matrix is invertible.
      *
      * \sa inverse()
      */
    inline void computeInverse(MatrixType *result) const
    {
      ei_assert(m_isInitialized && "FullPivotingHouseholderQR is not initialized.");
      ei_assert(m_qr.rows() == m_qr.cols() && "You can't take the inverse of a non-square matrix!");
      solve(MatrixType::Identity(m_qr.rows(), m_qr.cols()), result);
    }

    /** \returns the inverse of the matrix of which *this is the QR decomposition.
      *
      * \note If this matrix is not invertible, the returned matrix has undefined coefficients.
      *       Use isInvertible() to first determine whether this matrix is invertible.
      *
      * \sa computeInverse()
      */
    inline MatrixType inverse() const
    {
      MatrixType result;
      computeInverse(&result);
      return result;
    }

  protected:
    MatrixType m_qr;
    HCoeffsType m_hCoeffs;
    IntColVectorType m_rows_transpositions;
    IntRowVectorType m_cols_permutation;
    bool m_isInitialized;
    RealScalar m_precision;
    int m_rank;
    int m_det_pq;
};

#ifndef EIGEN_HIDE_HEAVY_CODE

template<typename MatrixType>
typename MatrixType::RealScalar FullPivotingHouseholderQR<MatrixType>::absDeterminant() const
{
  ei_assert(m_isInitialized && "FullPivotingHouseholderQR is not initialized.");
  ei_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  return ei_abs(m_qr.diagonal().prod());
}

template<typename MatrixType>
typename MatrixType::RealScalar FullPivotingHouseholderQR<MatrixType>::logAbsDeterminant() const
{
  ei_assert(m_isInitialized && "FullPivotingHouseholderQR is not initialized.");
  ei_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  return m_qr.diagonal().cwise().abs().cwise().log().sum();
}

template<typename MatrixType>
FullPivotingHouseholderQR<MatrixType>& FullPivotingHouseholderQR<MatrixType>::compute(const MatrixType& matrix)
{
  int rows = matrix.rows();
  int cols = matrix.cols();
  int size = std::min(rows,cols);
  m_rank = size;
  
  m_qr = matrix;
  m_hCoeffs.resize(size);

  RowVectorType temp(cols);

  m_precision = epsilon<Scalar>() * size;

  m_rows_transpositions.resize(matrix.rows());
  IntRowVectorType cols_transpositions(matrix.cols());
  m_cols_permutation.resize(matrix.cols());
  int number_of_transpositions = 0;
  
  RealScalar biggest;
  
  for (int k = 0; k < size; ++k)
  {
    int row_of_biggest_in_corner, col_of_biggest_in_corner;
    RealScalar biggest_in_corner;

    biggest_in_corner = m_qr.corner(Eigen::BottomRight, rows-k, cols-k)
                            .cwise().abs()
                            .maxCoeff(&row_of_biggest_in_corner, &col_of_biggest_in_corner);
    row_of_biggest_in_corner += k;
    col_of_biggest_in_corner += k;
    if(k==0) biggest = biggest_in_corner;
    
    // if the corner is negligible, then we have less than full rank, and we can finish early
    if(ei_isMuchSmallerThan(biggest_in_corner, biggest, m_precision))
    {
      m_rank = k;
      for(int i = k; i < size; i++)
      {
        m_rows_transpositions.coeffRef(i) = i;
        cols_transpositions.coeffRef(i) = i;
        m_hCoeffs.coeffRef(i) = Scalar(0);
      }
      break;
    }

    m_rows_transpositions.coeffRef(k) = row_of_biggest_in_corner;
    cols_transpositions.coeffRef(k) = col_of_biggest_in_corner;
    if(k != row_of_biggest_in_corner) {
      m_qr.row(k).end(cols-k).swap(m_qr.row(row_of_biggest_in_corner).end(cols-k));
      ++number_of_transpositions;
    }
    if(k != col_of_biggest_in_corner) {
      m_qr.col(k).swap(m_qr.col(col_of_biggest_in_corner));
      ++number_of_transpositions;
    }

    RealScalar beta;
    m_qr.col(k).end(rows-k).makeHouseholderInPlace(&m_hCoeffs.coeffRef(k), &beta);
    m_qr.coeffRef(k,k) = beta;

    m_qr.corner(BottomRight, rows-k, cols-k-1)
        .applyHouseholderOnTheLeft(m_qr.col(k).end(rows-k-1), m_hCoeffs.coeffRef(k), &temp.coeffRef(k+1));
  }

  for(int k = 0; k < matrix.cols(); ++k) m_cols_permutation.coeffRef(k) = k;
  for(int k = 0; k < size; ++k)
    std::swap(m_cols_permutation.coeffRef(k), m_cols_permutation.coeffRef(cols_transpositions.coeff(k)));

  m_det_pq = (number_of_transpositions%2) ? -1 : 1;
  m_isInitialized = true;
  
  return *this;
}

template<typename MatrixType>
template<typename OtherDerived, typename ResultType>
bool FullPivotingHouseholderQR<MatrixType>::solve(
  const MatrixBase<OtherDerived>& b,
  ResultType *result
) const
{
  ei_assert(m_isInitialized && "FullPivotingHouseholderQR is not initialized.");
  result->resize(m_qr.cols(), b.cols());
  if(m_rank==0)
  {
    if(b.squaredNorm() == RealScalar(0))
    {
      result->setZero();
      return true;
    }
    else return false;
  }
  
  const int rows = m_qr.rows();
  const int cols = b.cols();
  ei_assert(b.rows() == rows);
  
  typename OtherDerived::PlainMatrixType c(b);
  
  Matrix<Scalar,1,MatrixType::ColsAtCompileTime> temp(cols);
  for (int k = 0; k < m_rank; ++k)
  {
    int remainingSize = rows-k;
    c.row(k).swap(c.row(m_rows_transpositions.coeff(k)));
    c.corner(BottomRight, remainingSize, cols)
     .applyHouseholderOnTheLeft(m_qr.col(k).end(remainingSize-1), m_hCoeffs.coeff(k), &temp.coeffRef(0));
  }

  if(!isSurjective())
  {
    // is c is in the image of R ?
    RealScalar biggest_in_upper_part_of_c = c.corner(TopLeft, m_rank, c.cols()).cwise().abs().maxCoeff();
    RealScalar biggest_in_lower_part_of_c = c.corner(BottomLeft, rows-m_rank, c.cols()).cwise().abs().maxCoeff();
    if(!ei_isMuchSmallerThan(biggest_in_lower_part_of_c, biggest_in_upper_part_of_c, m_precision))
      return false;
  }
  m_qr.corner(TopLeft, m_rank, m_rank)
      .template triangularView<UpperTriangular>()
      .solveInPlace(c.corner(TopLeft, m_rank, c.cols()));

  for(int i = 0; i < m_rank; ++i) result->row(m_cols_permutation.coeff(i)) = c.row(i);
  for(int i = m_rank; i < m_qr.cols(); ++i) result->row(m_cols_permutation.coeff(i)).setZero();
  return true;
}

/** \returns the matrix Q */
template<typename MatrixType>
MatrixType FullPivotingHouseholderQR<MatrixType>::matrixQ() const
{
  ei_assert(m_isInitialized && "FullPivotingHouseholderQR is not initialized.");
  // compute the product H'_0 H'_1 ... H'_n-1,
  // where H_k is the k-th Householder transformation I - h_k v_k v_k'
  // and v_k is the k-th Householder vector [1,m_qr(k+1,k), m_qr(k+2,k), ...]
  int rows = m_qr.rows();
  int cols = m_qr.cols();
  int size = std::min(rows,cols);
  MatrixType res = MatrixType::Identity(rows, rows);
  Matrix<Scalar,1,MatrixType::RowsAtCompileTime> temp(rows);
  for (int k = size-1; k >= 0; k--)
  {
    res.block(k, k, rows-k, rows-k)
       .applyHouseholderOnTheLeft(m_qr.col(k).end(rows-k-1), ei_conj(m_hCoeffs.coeff(k)), &temp.coeffRef(k));
    res.row(k).swap(res.row(m_rows_transpositions.coeff(k)));
  }
  return res;
}

#endif // EIGEN_HIDE_HEAVY_CODE

/** \return the full-pivoting Householder QR decomposition of \c *this.
  *
  * \sa class FullPivotingHouseholderQR
  */
template<typename Derived>
const FullPivotingHouseholderQR<typename MatrixBase<Derived>::PlainMatrixType>
MatrixBase<Derived>::fullPivotingHouseholderQr() const
{
  return FullPivotingHouseholderQR<PlainMatrixType>(eval());
}

#endif // EIGEN_FULLPIVOTINGHOUSEHOLDERQR_H
