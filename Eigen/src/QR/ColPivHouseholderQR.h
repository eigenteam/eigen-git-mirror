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
        m_isInitialized(false),
        m_usePrescribedThreshold(false)
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

    HouseholderSequenceType householderQ(void) const;

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
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline int rank() const
    {
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      RealScalar premultiplied_threshold = ei_abs(m_maxpivot) * threshold();
      int result = 0;
      for(int i = 0; i < m_nonzero_pivots; ++i)
        result += (ei_abs(m_qr.coeff(i,i)) > premultiplied_threshold);
      return result;
    }

    /** \returns the dimension of the kernel of the matrix of which *this is the QR decomposition.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline int dimensionOfKernel() const
    {
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return cols() - rank();
    }

    /** \returns true if the matrix of which *this is the QR decomposition represents an injective
      *          linear map, i.e. has trivial kernel; false otherwise.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline bool isInjective() const
    {
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return rank() == cols();
    }

    /** \returns true if the matrix of which *this is the QR decomposition represents a surjective
      *          linear map; false otherwise.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline bool isSurjective() const
    {
      ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
      return rank() == rows();
    }

    /** \returns true if the matrix of which *this is the QR decomposition is invertible.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
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
               (*this, MatrixType::Identity(m_qr.rows(), m_qr.cols()));
    }

    inline int rows() const { return m_qr.rows(); }
    inline int cols() const { return m_qr.cols(); }
    const HCoeffsType& hCoeffs() const { return m_hCoeffs; }

    /** Allows to prescribe a threshold to be used by certain methods, such as rank(),
      * who need to determine when pivots are to be considered nonzero. This is not used for the
      * QR decomposition itself.
      *
      * When it needs to get the threshold value, Eigen calls threshold(). By default, this
      * uses a formula to automatically determine a reasonable threshold.
      * Once you have called the present method setThreshold(const RealScalar&),
      * your value is used instead.
      *
      * \param threshold The new value to use as the threshold.
      *
      * A pivot will be considered nonzero if its absolute value is strictly greater than
      *  \f$ \vert pivot \vert \leqslant threshold \times \vert maxpivot \vert \f$
      * where maxpivot is the biggest pivot.
      *
      * If you want to come back to the default behavior, call setThreshold(Default_t)
      */
    ColPivHouseholderQR& setThreshold(const RealScalar& threshold)
    {
      m_usePrescribedThreshold = true;
      m_prescribedThreshold = threshold;
    }

    /** Allows to come back to the default behavior, letting Eigen use its default formula for
      * determining the threshold.
      *
      * You should pass the special object Eigen::Default as parameter here.
      * \code qr.setThreshold(Eigen::Default); \endcode
      *
      * See the documentation of setThreshold(const RealScalar&).
      */
    ColPivHouseholderQR& setThreshold(Default_t)
    {
      m_usePrescribedThreshold = false;
    }

    /** Returns the threshold that will be used by certain methods such as rank().
      *
      * See the documentation of setThreshold(const RealScalar&).
      */
    RealScalar threshold() const
    {
      ei_assert(m_isInitialized || m_usePrescribedThreshold);
      return m_usePrescribedThreshold ? m_prescribedThreshold
      // this formula comes from experimenting (see "LU precision tuning" thread on the list)
      // and turns out to be identical to Higham's formula used already in LDLt.
                                      : epsilon<Scalar>() * m_qr.diagonalSize();
    }

    /** \returns the number of nonzero pivots in the QR decomposition.
      * Here nonzero is meant in the exact sense, not in a fuzzy sense.
      * So that notion isn't really intrinsically interesting, but it is
      * still useful when implementing algorithms.
      *
      * \sa rank()
      */
    inline int nonzeroPivots() const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      return m_nonzero_pivots;
    }

    /** \returns the absolute value of the biggest pivot, i.e. the biggest
      *          diagonal coefficient of U.
      */
    RealScalar maxPivot() const { return m_maxpivot; }

  protected:
    MatrixType m_qr;
    HCoeffsType m_hCoeffs;
    PermutationType m_cols_permutation;
    bool m_isInitialized, m_usePrescribedThreshold;
    RealScalar m_prescribedThreshold, m_maxpivot;
    int m_nonzero_pivots;
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
  return m_qr.diagonal().cwiseAbs().array().log().sum();
}

template<typename MatrixType>
ColPivHouseholderQR<MatrixType>& ColPivHouseholderQR<MatrixType>::compute(const MatrixType& matrix)
{
  int rows = matrix.rows();
  int cols = matrix.cols();
  int size = matrix.diagonalSize();

  m_qr = matrix;
  m_hCoeffs.resize(size);

  RowVectorType temp(cols);

  IntRowVectorType cols_transpositions(matrix.cols());
  int number_of_transpositions = 0;

  RealRowVectorType colSqNorms(cols);
  for(int k = 0; k < cols; ++k)
    colSqNorms.coeffRef(k) = m_qr.col(k).squaredNorm();

  RealScalar threshold_helper = colSqNorms.maxCoeff() * ei_abs2(epsilon<Scalar>()) / rows;

  m_nonzero_pivots = size; // the generic case is that in which all pivots are nonzero (invertible case)
  m_maxpivot = RealScalar(0);

  for(int k = 0; k < size; ++k)
  {
    // first, we look up in our table colSqNorms which column has the biggest squared norm
    int biggest_col_index;
    RealScalar biggest_col_sq_norm = colSqNorms.end(cols-k).maxCoeff(&biggest_col_index);
    biggest_col_index += k;

    // since our table colSqNorms accumulates imprecision at every step, we must now recompute
    // the actual squared norm of the selected column.
    // Note that not doing so does result in solve() sometimes returning inf/nan values
    // when running the unit test with 1000 repetitions.
    biggest_col_sq_norm = m_qr.col(biggest_col_index).end(rows-k).squaredNorm();

    // we store that back into our table: it can't hurt to correct our table.
    colSqNorms.coeffRef(biggest_col_index) = biggest_col_sq_norm;

    // if the current biggest column is smaller than epsilon times the initial biggest column,
    // terminate to avoid generating nan/inf values.
    // Note that here, if we test instead for "biggest == 0", we get a failure every 1000 (or so)
    // repetitions of the unit test, with the result of solve() filled with large values of the order
    // of 1/(size*epsilon).
    if(biggest_col_sq_norm < threshold_helper * (rows-k))
    {
      m_nonzero_pivots = k;
      m_hCoeffs.end(size-k).setZero();
      m_qr.corner(BottomRight,rows-k,cols-k)
          .template triangularView<StrictlyLowerTriangular>()
          .setZero();
      break;
    }

    // apply the transposition to the columns
    cols_transpositions.coeffRef(k) = biggest_col_index;
    if(k != biggest_col_index) {
      m_qr.col(k).swap(m_qr.col(biggest_col_index));
      std::swap(colSqNorms.coeffRef(k), colSqNorms.coeffRef(biggest_col_index));
      ++number_of_transpositions;
    }

    // generate the householder vector, store it below the diagonal
    RealScalar beta;
    m_qr.col(k).end(rows-k).makeHouseholderInPlace(m_hCoeffs.coeffRef(k), beta);

    // apply the householder transformation to the diagonal coefficient
    m_qr.coeffRef(k,k) = beta;

    // remember the maximum absolute value of diagonal coefficients
    if(ei_abs(beta) > m_maxpivot) m_maxpivot = ei_abs(beta);

    // apply the householder transformation
    m_qr.corner(BottomRight, rows-k, cols-k-1)
        .applyHouseholderOnTheLeft(m_qr.col(k).end(rows-k-1), m_hCoeffs.coeffRef(k), &temp.coeffRef(k+1));

    // update our table of squared norms of the columns
    colSqNorms.end(cols-k-1) -= m_qr.row(k).end(cols-k-1).cwiseAbs2();
  }

  m_cols_permutation.setIdentity(cols);
  for(int k = 0; k < m_nonzero_pivots; ++k)
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
    const int rows = dec().rows(), cols = dec().cols(),
              nonzero_pivots = dec().nonzeroPivots();
    dst.resize(cols, rhs().cols());
    ei_assert(rhs().rows() == rows);

    if(nonzero_pivots == 0)
    {
      dst.setZero();
      return;
    }

    typename Rhs::PlainMatrixType c(rhs());

    // Note that the matrix Q = H_0^* H_1^*... so its inverse is Q^* = (H_0 H_1 ...)^T
    c.applyOnTheLeft(householderSequence(
      dec().matrixQR(),
      dec().hCoeffs(),
      true,
      dec().nonzeroPivots()
    ));

    dec().matrixQR()
       .corner(TopLeft, nonzero_pivots, nonzero_pivots)
       .template triangularView<UpperTriangular>()
       .solveInPlace(c.corner(TopLeft, nonzero_pivots, c.cols()));


    typename Rhs::PlainMatrixType d(c);
    d.corner(TopLeft, nonzero_pivots, c.cols())
      = dec().matrixQR()
       .corner(TopLeft, nonzero_pivots, nonzero_pivots)
       .template triangularView<UpperTriangular>()
       * c.corner(TopLeft, nonzero_pivots, c.cols());

    for(int i = 0; i < nonzero_pivots; ++i) dst.row(dec().colsPermutation().indices().coeff(i)) = c.row(i);
    for(int i = nonzero_pivots; i < cols; ++i) dst.row(dec().colsPermutation().indices().coeff(i)).setZero();
  }
};

/** \returns the matrix Q as a sequence of householder transformations */
template<typename MatrixType>
typename ColPivHouseholderQR<MatrixType>::HouseholderSequenceType ColPivHouseholderQR<MatrixType>
  ::householderQ() const
{
  ei_assert(m_isInitialized && "ColPivHouseholderQR is not initialized.");
  return HouseholderSequenceType(m_qr, m_hCoeffs.conjugate(), false, m_nonzero_pivots);
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
