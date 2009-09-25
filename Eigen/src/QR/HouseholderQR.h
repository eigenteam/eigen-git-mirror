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

#ifndef EIGEN_QR_H
#define EIGEN_QR_H

/** \ingroup QR_Module
  * \nonstableyet
  *
  * \class HouseholderQR
  *
  * \brief Householder QR decomposition of a matrix
  *
  * \param MatrixType the type of the matrix of which we are computing the QR decomposition
  *
  * This class performs a QR decomposition using Householder transformations. The result is
  * stored in a compact way compatible with LAPACK.
  *
  * Note that no pivoting is performed. This is \b not a rank-revealing decomposition.
  * If you want that feature, use FullPivotingHouseholderQR or ColPivotingHouseholderQR instead.
  *
  * This Householder QR decomposition is faster, but less numerically stable and less feature-full than
  * FullPivotingHouseholderQR or ColPivotingHouseholderQR.
  *
  * \sa MatrixBase::householderQr()
  */
template<typename MatrixType> class HouseholderQR
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
    typedef Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime, AutoAlign | (ei_traits<MatrixType>::Flags&RowMajorBit ? RowMajor : ColMajor)> MatrixQType;
    typedef Matrix<Scalar, DiagSizeAtCompileTime, 1> HCoeffsType;
    typedef Matrix<Scalar, 1, ColsAtCompileTime> RowVectorType;
    typedef typename HouseholderSequence<MatrixQType,HCoeffsType>::ConjugateReturnType HouseholderSequenceType;

    /**
    * \brief Default Constructor.
    *
    * The default constructor is useful in cases in which the user intends to
    * perform decompositions via HouseholderQR::compute(const MatrixType&).
    */
    HouseholderQR() : m_qr(), m_hCoeffs(), m_isInitialized(false) {}

    HouseholderQR(const MatrixType& matrix)
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
      * \param result a pointer to the vector/matrix in which to store the solution, if any exists.
      *          Resized if necessary, so that result->rows()==A.cols() and result->cols()==b.cols().
      *          If no solution exists, *result is left with undefined coefficients.
      *
      * \note The case where b is a matrix is not yet implemented. Also, this
      *       code is space inefficient.
      *
      * Example: \include HouseholderQR_solve.cpp
      * Output: \verbinclude HouseholderQR_solve.out
      */
    template<typename OtherDerived, typename ResultType>
    void solve(const MatrixBase<OtherDerived>& b, ResultType *result) const;

    MatrixQType matrixQ() const;

    HouseholderSequenceType matrixQAsHouseholderSequence() const
    {
      return HouseholderSequenceType(m_qr, m_hCoeffs.conjugate());
    }

    /** \returns a reference to the matrix where the Householder QR decomposition is stored
      * in a LAPACK-compatible way.
      */
    const MatrixType& matrixQR() const
    {
        ei_assert(m_isInitialized && "HouseholderQR is not initialized.");
        return m_qr;
    }

    HouseholderQR& compute(const MatrixType& matrix);

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

  protected:
    MatrixType m_qr;
    HCoeffsType m_hCoeffs;
    bool m_isInitialized;
};

#ifndef EIGEN_HIDE_HEAVY_CODE

template<typename MatrixType>
typename MatrixType::RealScalar HouseholderQR<MatrixType>::absDeterminant() const
{
  ei_assert(m_isInitialized && "HouseholderQR is not initialized.");
  ei_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  return ei_abs(m_qr.diagonal().prod());
}

template<typename MatrixType>
typename MatrixType::RealScalar HouseholderQR<MatrixType>::logAbsDeterminant() const
{
  ei_assert(m_isInitialized && "HouseholderQR is not initialized.");
  ei_assert(m_qr.rows() == m_qr.cols() && "You can't take the determinant of a non-square matrix!");
  return m_qr.diagonal().cwise().abs().cwise().log().sum();
}

template<typename MatrixType>
HouseholderQR<MatrixType>& HouseholderQR<MatrixType>::compute(const MatrixType& matrix)
{
  int rows = matrix.rows();
  int cols = matrix.cols();
  int size = std::min(rows,cols);

  m_qr = matrix;
  m_hCoeffs.resize(size);

  RowVectorType temp(cols);

  for (int k = 0; k < size; ++k)
  {
    int remainingRows = rows - k;
    int remainingCols = cols - k - 1;

    RealScalar beta;
    m_qr.col(k).end(remainingRows).makeHouseholderInPlace(&m_hCoeffs.coeffRef(k), &beta);
    m_qr.coeffRef(k,k) = beta;

    // apply H to remaining part of m_qr from the left
    m_qr.corner(BottomRight, remainingRows, remainingCols)
        .applyHouseholderOnTheLeft(m_qr.col(k).end(remainingRows-1), m_hCoeffs.coeffRef(k), &temp.coeffRef(k+1));
  }
  m_isInitialized = true;
  return *this;
}

template<typename MatrixType>
template<typename OtherDerived, typename ResultType>
void HouseholderQR<MatrixType>::solve(
  const MatrixBase<OtherDerived>& b,
  ResultType *result
) const
{
  ei_assert(m_isInitialized && "HouseholderQR is not initialized.");
  const int rows = m_qr.rows();
  const int cols = b.cols();
  ei_assert(b.rows() == rows);
  result->resize(rows, cols);

  *result = b;
  result->applyOnTheLeft(matrixQAsHouseholderSequence().inverse());

  const int rank = std::min(result->rows(), result->cols());
  m_qr.corner(TopLeft, rank, rank)
      .template triangularView<UpperTriangular>()
      .solveInPlace(result->corner(TopLeft, rank, result->cols()));
}

/** \returns the matrix Q */
template<typename MatrixType>
typename HouseholderQR<MatrixType>::MatrixQType HouseholderQR<MatrixType>::matrixQ() const
{
  ei_assert(m_isInitialized && "HouseholderQR is not initialized.");
  return matrixQAsHouseholderSequence();
}

#endif // EIGEN_HIDE_HEAVY_CODE

/** \return the Householder QR decomposition of \c *this.
  *
  * \sa class HouseholderQR
  */
template<typename Derived>
const HouseholderQR<typename MatrixBase<Derived>::PlainMatrixType>
MatrixBase<Derived>::householderQr() const
{
  return HouseholderQR<PlainMatrixType>(eval());
}


#endif // EIGEN_QR_H
