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

#ifndef EIGEN_COLPIVOTINGHOUSEHOLDERQR_H
#define EIGEN_COLPIVOTINGHOUSEHOLDERQR_H

/** \ingroup QR_Module
  * \nonstableyet
  *
  * \class ColPivHouseholderQR
  *
  * \brief Householder rank-revealing QR decomposition of a matrix with column-pivoting
  *
  * \param MatrixType the type of the matrix of which we are computing the QR decomposition
  *
  * This class performs a rank-revealing QR decomposition using Householder transformations.
  *
  * This decomposition performs column pivoting in order to be rank-revealing and improve
  * numerical stability. It is slower than HouseholderQR, and faster than FullPivHouseholderQR.
  *
  * \sa MatrixBase::colPivHouseholderQr()
  */
template<typename _MatrixType> class ColPivHouseholderQR
{
  public:

    typedef _MatrixType MatrixType;
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
    typedef PermutationMatrix<ColsAtCompileTime> PermutationType;
    typedef Matrix<int, 1, ColsAtCompileTime> IntRowVectorType;
    typedef Matrix<Scalar, 1, ColsAtCompileTime> RowVectorType;
    typedef Matrix<RealScalar, 1, ColsAtCompileTime> RealRowVectorType;
    typedef typename HouseholderSequence<MatrixType,HCoeffsType>::ConjugateReturnType HouseholderSequenceType;

    /**
    * \brief Default Constructor.
    *
    * The default constructor is useful in cases in which the user intends to
    * perform decompositions via ColPivHouseholderQR::compute(const MatrixType&).
    */
    ColPivHouseholderQR() : m_qr(), m_hCoeffs(), m_isInitialized(false) {}

    ColPivHouseholderQR(const MatrixType& matrix)
      : m_qr(matrix.rows(), matrix.cols()),
        m_hCoeffs(std::min(matrix.rows(),matrix.cols())),
        m_isInitialized(false)
    {
      compute(matrix);
    }

    /** This method finds a solution x to the equation Ax=b, where A is the matrix of which
      * *this is the QR decomposition, if any exists.
      *
      * \param b the right-hand-side of the equation to solve.
      *
      * \returns a solution.
      *
      * \note The case where b is a matrix is not yet implemented. Also, this
      *       code is space inefficient.
      *
      * \note_about_checking_solutions
      *
      * \note_about_arbitrary_choice_of_solution
      *
      * Example: \include ColPivHouseholderQR_solve.cpp
      * Output: \verbinclude ColPivHouseholderQR_solve.out
      */
    template<typename Rhs>
    inline const ei_solve_retval<ColPivHouseholderQR, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return ei_solve_retval<ColPivHouseholderQR, Rhs>(*this, b.derived());
    }

    HouseholderSequenceType matrixQ(void) const;

    /** \returns a reference to the matrix where the Householder QR decomposition is stored
      */
    const MatrixType& matrixQR() const
    {
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return m_qr;
    }

    ColPivHouseholderQR& compute(const MatrixType& matrix);

    const PermutationType& colsPermutation() const
    {
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return m_cols_permutation;
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
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return m_rank;
    }

    /** \returns the dimension of the kernel of the matrix of which *this is the QR decomposition.
      *
      * \note Since the rank is computed at the time of the construction of the QR decomposition, this
      *       method almost does not perform any further computation.
      */
    inline int dimensionOfKernel() const
    {
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
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
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
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
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return m_rank == m_qr.rows();
    }

    /** \returns true if the matrix of which *this is the QR decomposition is invertible.
      *
      * \note Since the rank is computed at the time of the construction of the QR decomposition, this
      *       method almost does not perform any further computation.
      */
    inline bool isInvertible() const
    {
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return isInjective() && isSurjective();
    }

    /** \returns the inverse of the matrix of which *this is the QR decomposition.
      *
      * \note If this matrix is not invertible, the returned matrix has undefined coefficients.
      *       Use isInvertible() to first determine whether this matrix is invertible.
      */
    inline const
    ei_solve_retval<ColPivHouseholderQR, typename MatrixType::IdentityReturnType>
    inverse() const
    {
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return ei_solve_retval<ColPivHouseholderQR,typename MatrixType::IdentityReturnType>
               (*this, MatrixType::Identity(m_qr.rows(), m_qr.cols()).nestByValue());
    }

    inline int rows() const { return m_qr.rows(); }
    inline int cols() const { return m_qr.cols(); }
    const HCoeffsType& hCoeffs() const { return m_hCoeffs; }

  protected:
    MatrixType m_qr;
    HCoeffsType m_hCoeffs;
    PermutationType m_cols_permutation;
    bool m_isInitialized;
    RealScalar m_precision;
    int m_rank;
    int m_det_pq;
};

#ifndef EIGEN_HIDE_HEAVY_CODE

template<typename MatrixType>
typename MatrixType::RealScalar ColPivHouseholderQR<MatrixType>::absDeterminant() const
{
  ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
  ei_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  return ei_abs(m_qr.diagonal().prod());
}

template<typename MatrixType>
typename MatrixType::RealScalar ColPivHouseholderQR<MatrixType>::logAbsDeterminant() const
{
  ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
  ei_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  return m_qr.diagonal().cwise().abs().cwise().log().sum();
}

template<typename MatrixType>
ColPivHouseholderQR<MatrixType>& ColPivHouseholderQR<MatrixType>::compute(const MatrixType& matrix)
{
  int rows = matrix.rows();
  int cols = matrix.cols();
  int size = std::min(rows,cols);
  m_rank = size;

  m_qr = matrix;
  m_hCoeffs.resize(size);

  RowVectorType temp(cols);

  m_precision = epsilon<Scalar>() * size;

  IntRowVectorType cols_transpositions(matrix.cols());
  int number_of_transpositions = 0;

  RealRowVectorType colSqNorms(cols);
  for(int k = 0; k < cols; ++k)
    colSqNorms.coeffRef(k) = m_qr.col(k).squaredNorm();
  RealScalar biggestColSqNorm = colSqNorms.maxCoeff();

  for (int k = 0; k < size; ++k)
  {
    int biggest_col_in_corner;
    RealScalar biggestColSqNormInCorner = colSqNorms.end(cols-k).maxCoeff(&biggest_col_in_corner);
    biggest_col_in_corner += k;

    // if the corner is negligible, then we have less than full rank, and we can finish early
    if(ei_isMuchSmallerThan(biggestColSqNormInCorner, biggestColSqNorm, m_precision))
    {
      m_rank = k;
      for(int i = k; i < size; i++)
      {
        cols_transpositions.coeffRef(i) = i;
        m_hCoeffs.coeffRef(i) = Scalar(0);
      }
      break;
    }

    cols_transpositions.coeffRef(k) = biggest_col_in_corner;
    if(k != biggest_col_in_corner) {
      m_qr.col(k).swap(m_qr.col(biggest_col_in_corner));
      std::swap(colSqNorms.coeffRef(k), colSqNorms.coeffRef(biggest_col_in_corner));
      ++number_of_transpositions;
    }

    RealScalar beta;
    m_qr.col(k).end(rows-k).makeHouseholderInPlace(m_hCoeffs.coeffRef(k), beta);
    m_qr.coeffRef(k,k) = beta;

    m_qr.corner(BottomRight, rows-k, cols-k-1)
        .applyHouseholderOnTheLeft(m_qr.col(k).end(rows-k-1), m_hCoeffs.coeffRef(k), &temp.coeffRef(k+1));

    colSqNorms.end(cols-k-1) -= m_qr.row(k).end(cols-k-1).cwise().abs2();
  }

  m_cols_permutation.setIdentity(cols);
  for(int k = 0; k < size; ++k)
    m_cols_permutation.applyTranspositionOnTheRight(k, cols_transpositions.coeff(k));

  m_det_pq = (number_of_transpositions%2) ? -1 : 1;
  m_isInitialized = true;

  return *this;
}

template<typename _MatrixType, typename Rhs>
struct ei_solve_retval<ColPivHouseholderQR<_MatrixType>, Rhs>
  : ei_solve_retval_base<ColPivHouseholderQR<_MatrixType>, Rhs>
{
  EIGEN_MAKE_SOLVE_HELPERS(ColPivHouseholderQR<_MatrixType>,Rhs)
  
  template<typename Dest> void evalTo(Dest& dst) const
  {
    const int rows = dec().rows(), cols = dec().cols();
    dst.resize(cols, rhs().cols());
    ei_assert(rhs().rows() == rows);

    // FIXME introduce nonzeroPivots() and use it here. and more generally,
    // make the same improvements in this dec as in FullPivLU.
    if(dec().rank()==0)
    {
      dst.setZero();
      return;
    }

    typename Rhs::PlainMatrixType c(rhs());

    // Note that the matrix Q = H_0^* H_1^*... so its inverse is Q^* = (H_0 H_1 ...)^T
    c.applyOnTheLeft(householderSequence(
      dec().matrixQR().corner(TopLeft,rows,dec().rank()),
      dec().hCoeffs().start(dec().rank())).transpose()
    );

    if(!dec().isSurjective())
    {
      // is c is in the image of R ?
      RealScalar biggest_in_upper_part_of_c = c.corner(TopLeft, dec().rank(), c.cols()).cwise().abs().maxCoeff();
      RealScalar biggest_in_lower_part_of_c = c.corner(BottomLeft, rows-dec().rank(), c.cols()).cwise().abs().maxCoeff();
      // FIXME brain dead
      const RealScalar m_precision = epsilon<Scalar>() * std::min(rows,cols);
      if(!ei_isMuchSmallerThan(biggest_in_lower_part_of_c, biggest_in_upper_part_of_c, m_precision*4))
        return;
    }

    dec().matrixQR()
       .corner(TopLeft, dec().rank(), dec().rank())
       .template triangularView<UpperTriangular>()
       .solveInPlace(c.corner(TopLeft, dec().rank(), c.cols()));

    for(int i = 0; i < dec().rank(); ++i) dst.row(dec().colsPermutation().indices().coeff(i)) = c.row(i);
    for(int i = dec().rank(); i < cols; ++i) dst.row(dec().colsPermutation().indices().coeff(i)).setZero();
  }
};

/** \returns the matrix Q as a sequence of householder transformations */
template<typename MatrixType>
typename ColPivHouseholderQR<MatrixType>::HouseholderSequenceType ColPivHouseholderQR<MatrixType>::matrixQ() const
{
  ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
  return HouseholderSequenceType(m_qr, m_hCoeffs.conjugate());
}

#endif // EIGEN_HIDE_HEAVY_CODE

/** \return the column-pivoting Householder QR decomposition of \c *this.
  *
  * \sa class ColPivHouseholderQR
  */
template<typename Derived>
const ColPivHouseholderQR<typename MatrixBase<Derived>::PlainMatrixType>
MatrixBase<Derived>::colPivHouseholderQr() const
{
  return ColPivHouseholderQR<PlainMatrixType>(eval());
}


#endif // EIGEN_COLPIVOTINGHOUSEHOLDERQR_H
