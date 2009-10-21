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

#ifndef EIGEN_LU_H
#define EIGEN_LU_H

template<typename MatrixType, typename Rhs> struct ei_lu_solve_impl;
template<typename MatrixType> struct ei_lu_kernel_impl;
template<typename MatrixType> struct ei_lu_image_impl;

/** \ingroup LU_Module
  *
  * \class LU
  *
  * \brief LU decomposition of a matrix with complete pivoting, and related features
  *
  * \param MatrixType the type of the matrix of which we are computing the LU decomposition
  *
  * This class represents a LU decomposition of any matrix, with complete pivoting: the matrix A
  * is decomposed as A = PLUQ where L is unit-lower-triangular, U is upper-triangular, and P and Q
  * are permutation matrices. This is a rank-revealing LU decomposition. The eigenvalues (diagonal
  * coefficients) of U are sorted in such a way that any zeros are at the end.
  *
  * This decomposition provides the generic approach to solving systems of linear equations, computing
  * the rank, invertibility, inverse, kernel, and determinant.
  *
  * This LU decomposition is very stable and well tested with large matrices. However there are use cases where the SVD
  * decomposition is inherently more stable and/or flexible. For example, when computing the kernel of a matrix,
  * working with the SVD allows to select the smallest singular values of the matrix, something that
  * the LU decomposition doesn't see.
  *
  * The data of the LU decomposition can be directly accessed through the methods matrixLU(),
  * permutationP(), permutationQ().
  *
  * As an exemple, here is how the original matrix can be retrieved:
  * \include class_LU.cpp
  * Output: \verbinclude class_LU.out
  *
  * \sa MatrixBase::lu(), MatrixBase::determinant(), MatrixBase::inverse()
  */
template<typename MatrixType> class LU
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

    /**
    * \brief Default Constructor.
    *
    * The default constructor is useful in cases in which the user intends to
    * perform decompositions via LU::compute(const MatrixType&).
    */
    LU();

    /** Constructor.
      *
      * \param matrix the matrix of which to compute the LU decomposition.
      *               It is required to be nonzero.
      */
    LU(const MatrixType& matrix);

    /** Computes the LU decomposition of the given matrix.
      *
      * \param matrix the matrix of which to compute the LU decomposition.
      *               It is required to be nonzero.
      *
      * \returns a reference to *this
      */
    LU& compute(const MatrixType& matrix);
    
    /** \returns the LU decomposition matrix: the upper-triangular part is U, the
      * unit-lower-triangular part is L (at least for square matrices; in the non-square
      * case, special care is needed, see the documentation of class LU).
      *
      * \sa matrixL(), matrixU()
      */
    inline const MatrixType& matrixLU() const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      return m_lu;
    }
    
    /** \returns the number of nonzero pivots in the LU decomposition.
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
    
    /** \returns a vector of integers, whose size is the number of rows of the matrix being decomposed,
      * representing the P permutation i.e. the permutation of the rows. For its precise meaning,
      * see the examples given in the documentation of class LU.
      *
      * \sa permutationQ()
      */
    inline const IntColVectorType& permutationP() const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      return m_p;
    }

    /** \returns a vector of integers, whose size is the number of columns of the matrix being
      * decomposed, representing the Q permutation i.e. the permutation of the columns.
      * For its precise meaning, see the examples given in the documentation of class LU.
      *
      * \sa permutationP()
      */
    inline const IntRowVectorType& permutationQ() const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      return m_q;
    }

    /** \returns the kernel of the matrix, also called its null-space. The columns of the returned matrix
      * will form a basis of the kernel.
      *
      * \note If the kernel has dimension zero, then the returned matrix is a column-vector filled with zeros.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      *
      * Example: \include LU_kernel.cpp
      * Output: \verbinclude LU_kernel.out
      *
      * \sa image()
      */
    inline const ei_lu_kernel_impl<MatrixType> kernel() const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      return ei_lu_kernel_impl<MatrixType>(*this);
    }

    /** \returns the image of the matrix, also called its column-space. The columns of the returned matrix
      * will form a basis of the kernel.
      *
      * \param originalMatrix the original matrix, of which *this is the LU decomposition.
      *                       The reason why it is needed to pass it here, is that this allows
      *                       a large optimization, as otherwise this method would need to reconstruct it
      *                       from the LU decomposition.
      *
      * \note If the image has dimension zero, then the returned matrix is a column-vector filled with zeros.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      *
      * Example: \include LU_image.cpp
      * Output: \verbinclude LU_image.out
      *
      * \sa kernel()
      */
    template<typename OriginalMatrixType>
    inline const ei_lu_image_impl<MatrixType>
      image(const MatrixBase<OriginalMatrixType>& originalMatrix) const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      return ei_lu_image_impl<MatrixType>(*this, originalMatrix.derived());
    }

    /** This method returns a solution x to the equation Ax=b, where A is the matrix of which
      * *this is the LU decomposition.
      *
      * \param b the right-hand-side of the equation to solve. Can be a vector or a matrix,
      *          the only requirement in order for the equation to make sense is that
      *          b.rows()==A.rows(), where A is the matrix of which *this is the LU decomposition.
      *
      * \returns a solution.
      *
      * \note_about_checking_solutions
      *
      * \note_about_arbitrary_choice_of_solution
      * \note_about_using_kernel_to_study_multiple_solutions
      *
      * Example: \include LU_solve.cpp
      * Output: \verbinclude LU_solve.out
      *
      * \sa TriangularView::solve(), kernel(), inverse()
      */
    template<typename Rhs>
    inline const ei_lu_solve_impl<MatrixType, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      return ei_lu_solve_impl<MatrixType, Rhs>(*this, b.derived());
    }

    /** \returns the determinant of the matrix of which
      * *this is the LU decomposition. It has only linear complexity
      * (that is, O(n) where n is the dimension of the square matrix)
      * as the LU decomposition has already been computed.
      *
      * \note This is only for square matrices.
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

    /** Allows to prescribe a threshold to be used by certain methods, such as rank(),
      * who need to determine when pivots are to be considered nonzero. This is not used for the
      * LU decomposition itself.
      *
      * When it needs to get the threshold value, Eigen calls threshold(). By default, this calls
      * defaultThreshold(). Once you have called the present method setThreshold(const RealScalar&),
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
    LU& setThreshold(const RealScalar& threshold)
    {
      m_usePrescribedThreshold = true;
      m_prescribedThreshold = threshold;
    }
    
    /** Allows to come back to the default behavior, letting Eigen use its default formula for
      * determining the threshold.
      *
      * You should pass the special object Eigen::Default as parameter here.
      * \code lu.setThreshold(Eigen::Default); \endcode
      *
      * See the documentation of setThreshold(const RealScalar&).
      */
    LU& setThreshold(Default_t)
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
                                      : epsilon<Scalar>() * m_lu.diagonalSize();
    }
    
    /** \returns the rank of the matrix of which *this is the LU decomposition.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline int rank() const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      RealScalar premultiplied_threshold = ei_abs(m_maxpivot) * threshold();
      int result = 0;
      for(int i = 0; i < m_nonzero_pivots; ++i)
        result += (ei_abs(m_lu.coeff(i,i)) > premultiplied_threshold);
      return result;
    }
    
    /** \returns the dimension of the kernel of the matrix of which *this is the LU decomposition.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline int dimensionOfKernel() const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      return m_lu.cols() - rank();
    }

    /** \returns true if the matrix of which *this is the LU decomposition represents an injective
      *          linear map, i.e. has trivial kernel; false otherwise.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline bool isInjective() const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      return rank() == m_lu.cols();
    }

    /** \returns true if the matrix of which *this is the LU decomposition represents a surjective
      *          linear map; false otherwise.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline bool isSurjective() const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      return rank() == m_lu.rows();
    }

    /** \returns true if the matrix of which *this is the LU decomposition is invertible.
      *
      * \note This method has to determine which pivots should be considered nonzero.
      *       For that, it uses the threshold value that you can control by calling
      *       setThreshold(const RealScalar&).
      */
    inline bool isInvertible() const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      return isInjective() && (m_lu.rows() == m_lu.cols());
    }

    /** \returns the inverse of the matrix of which *this is the LU decomposition.
      *
      * \note If this matrix is not invertible, the returned matrix has undefined coefficients.
      *       Use isInvertible() to first determine whether this matrix is invertible.
      *
      * \sa MatrixBase::inverse()
      */
    inline const ei_lu_solve_impl<MatrixType,NestByValue<typename MatrixType::IdentityReturnType> > inverse() const
    {
      ei_assert(m_isInitialized && "LU is not initialized.");
      ei_assert(m_lu.rows() == m_lu.cols() && "You can't take the inverse of a non-square matrix!");
      return ei_lu_solve_impl<MatrixType,NestByValue<typename MatrixType::IdentityReturnType> >
               (*this, MatrixType::Identity(m_lu.rows(), m_lu.cols()).nestByValue());
    }

  protected:
    MatrixType m_lu;
    IntColVectorType m_p;
    IntRowVectorType m_q;
    int m_det_pq, m_nonzero_pivots;
    RealScalar m_maxpivot, m_prescribedThreshold;
    bool m_isInitialized, m_usePrescribedThreshold;
};

template<typename MatrixType>
LU<MatrixType>::LU()
  : m_isInitialized(false), m_usePrescribedThreshold(false)
{
}

template<typename MatrixType>
LU<MatrixType>::LU(const MatrixType& matrix)
  : m_isInitialized(false), m_usePrescribedThreshold(false)
{
  compute(matrix);
}

template<typename MatrixType>
LU<MatrixType>& LU<MatrixType>::compute(const MatrixType& matrix)
{
  m_isInitialized = true;
  m_lu = matrix;
  m_p.resize(matrix.rows());
  m_q.resize(matrix.cols());

  const int size = matrix.diagonalSize();
  const int rows = matrix.rows();
  const int cols = matrix.cols();

  // will store the transpositions, before we accumulate them at the end.
  // can't accumulate on-the-fly because that will be done in reverse order for the rows.
  IntColVectorType rows_transpositions(matrix.rows());
  IntRowVectorType cols_transpositions(matrix.cols());
  int number_of_transpositions = 0; // number of NONTRIVIAL transpositions, i.e. rows_transpositions[i]!=i

  m_nonzero_pivots = size; // the generic case is that in which all pivots are nonzero (invertible case)
  m_maxpivot = RealScalar(0);
  for(int k = 0; k < size; ++k)
  {
    // First, we need to find the pivot.

    // biggest coefficient in the remaining bottom-right corner (starting at row k, col k)
    int row_of_biggest_in_corner, col_of_biggest_in_corner;
    RealScalar biggest_in_corner;
    biggest_in_corner = m_lu.corner(Eigen::BottomRight, rows-k, cols-k)
                        .cwise().abs()
                        .maxCoeff(&row_of_biggest_in_corner, &col_of_biggest_in_corner);
    row_of_biggest_in_corner += k; // correct the values! since they were computed in the corner,
    col_of_biggest_in_corner += k; // need to add k to them.

    // if the pivot (hence the corner) is exactly zero, terminate to avoid generating nan/inf values
    if(biggest_in_corner == RealScalar(0))
    {
      // before exiting, make sure to initialize the still uninitialized row_transpositions
      // in a sane state without destroying what we already have.
      m_nonzero_pivots = k;
      for(int i = k; i < size; i++)
      {
        rows_transpositions.coeffRef(i) = i;
        cols_transpositions.coeffRef(i) = i;
      }
      break;
    }

    if(biggest_in_corner > m_maxpivot) m_maxpivot = biggest_in_corner;

    // Now that we've found the pivot, we need to apply the row/col swaps to
    // bring it to the location (k,k).

    rows_transpositions.coeffRef(k) = row_of_biggest_in_corner;
    cols_transpositions.coeffRef(k) = col_of_biggest_in_corner;
    if(k != row_of_biggest_in_corner) {
      m_lu.row(k).swap(m_lu.row(row_of_biggest_in_corner));
      ++number_of_transpositions;
    }
    if(k != col_of_biggest_in_corner) {
      m_lu.col(k).swap(m_lu.col(col_of_biggest_in_corner));
      ++number_of_transpositions;
    }

    // Now that the pivot is at the right location, we update the remaining
    // bottom-right corner by Gaussian elimination.
    
    if(k<rows-1)
      m_lu.col(k).end(rows-k-1) /= m_lu.coeff(k,k);
    if(k<size-1)
      m_lu.block(k+1,k+1,rows-k-1,cols-k-1).noalias() -= m_lu.col(k).end(rows-k-1) * m_lu.row(k).end(cols-k-1);
  }

  // the main loop is over, we still have to accumulate the transpositions to find the
  // permutations P and Q

  for(int k = 0; k < matrix.rows(); ++k) m_p.coeffRef(k) = k;
  for(int k = size-1; k >= 0; --k)
    std::swap(m_p.coeffRef(k), m_p.coeffRef(rows_transpositions.coeff(k)));

  for(int k = 0; k < matrix.cols(); ++k) m_q.coeffRef(k) = k;
  for(int k = 0; k < size; ++k)
    std::swap(m_q.coeffRef(k), m_q.coeffRef(cols_transpositions.coeff(k)));

  m_det_pq = (number_of_transpositions%2) ? -1 : 1;
  return *this;
}

template<typename MatrixType>
typename ei_traits<MatrixType>::Scalar LU<MatrixType>::determinant() const
{
  ei_assert(m_isInitialized && "LU is not initialized.");
  ei_assert(m_lu.rows() == m_lu.cols() && "You can't take the determinant of a non-square matrix!");
  return Scalar(m_det_pq) * m_lu.diagonal().prod();
}

/********* Implementation of kernel() **************************************************/

template<typename MatrixType>
struct ei_traits<ei_lu_kernel_impl<MatrixType> >
{
  typedef Matrix<
    typename MatrixType::Scalar,
    MatrixType::ColsAtCompileTime, // the number of rows in the "kernel matrix"
                                   // is the number of cols of the original matrix
                                   // so that the product "matrix * kernel = zero" makes sense
    Dynamic,                       // we don't know at compile-time the dimension of the kernel
    MatrixType::Options,
    MatrixType::MaxColsAtCompileTime, // see explanation for 2nd template parameter
    MatrixType::MaxColsAtCompileTime // the kernel is a subspace of the domain space,
                                     // whose dimension is the number of columns of the original matrix
  > ReturnMatrixType;
};

template<typename MatrixType>
struct ei_lu_kernel_impl : public ReturnByValue<ei_lu_kernel_impl<MatrixType> >
{
  typedef LU<MatrixType> LUType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  const LUType& m_lu;
  int m_rank, m_cols;
  
  ei_lu_kernel_impl(const LUType& lu)
    : m_lu(lu),
      m_rank(lu.rank()),
      m_cols(m_rank==lu.matrixLU().cols() ? 1 : lu.matrixLU().cols() - m_rank){}

  inline int rows() const { return m_lu.matrixLU().cols(); }
  inline int cols() const { return m_cols; }

  template<typename Dest> void evalTo(Dest& dst) const
  {
    const int cols = m_lu.matrixLU().cols(), dimker = cols - m_rank;
    if(dimker == 0)
    {
      // The Kernel is just {0}, so it doesn't have a basis properly speaking, but let's
      // avoid crashing/asserting as that depends on floating point calculations. Let's
      // just return a single column vector filled with zeros.
      dst.setZero();
      return;
    }
    
    /* Let us use the following lemma:
      *
      * Lemma: If the matrix A has the LU decomposition PAQ = LU,
      * then Ker A = Q(Ker U).
      *
      * Proof: trivial: just keep in mind that P, Q, L are invertible.
      */

    /* Thus, all we need to do is to compute Ker U, and then apply Q.
      *
      * U is upper triangular, with eigenvalues sorted so that any zeros appear at the end.
      * Thus, the diagonal of U ends with exactly
      * m_dimKer zero's. Let us use that to construct dimKer linearly
      * independent vectors in Ker U.
      */

    Matrix<int, Dynamic, 1, 0, LUType::MaxSmallDimAtCompileTime, 1> pivots(m_rank);
    RealScalar premultiplied_threshold = m_lu.maxPivot() * m_lu.threshold();
    int p = 0;
    for(int i = 0; i < m_lu.nonzeroPivots(); ++i)
      if(ei_abs(m_lu.matrixLU().coeff(i,i)) > premultiplied_threshold)
        pivots.coeffRef(p++) = i;
    ei_assert(p == m_rank && "You hit a bug in Eigen! Please report (backtrace and matrix)!");

    // we construct a temporaty trapezoid matrix m, by taking the U matrix and
    // permuting the rows and cols to bring the nonnegligible pivots to the top of
    // the main diagonal. We need that to be able to apply our triangular solvers.
    // FIXME when we get triangularView-for-rectangular-matrices, this can be simplified
    Matrix<typename MatrixType::Scalar, Dynamic, Dynamic, MatrixType::Options,
           LUType::MaxSmallDimAtCompileTime, MatrixType::MaxColsAtCompileTime>
      m(m_lu.matrixLU().block(0, 0, m_rank, cols));
    for(int i = 0; i < m_rank; ++i)
    {
      if(i) m.row(i).start(i).setZero();
      m.row(i).end(cols-i) = m_lu.matrixLU().row(pivots.coeff(i)).end(cols-i);
    }
    m.block(0, 0, m_rank, m_rank).template triangularView<StrictlyLowerTriangular>().setZero();
    for(int i = 0; i < m_rank; ++i)
      m.col(i).swap(m.col(pivots.coeff(i)));

    // ok, we have our trapezoid matrix, we can apply the triangular solver.
    // notice that the math behind this suggests that we should apply this to the
    // negative of the RHS, but for performance we just put the negative sign elsewhere, see below.
    m.corner(TopLeft, m_rank, m_rank)
     .template triangularView<UpperTriangular>().solveInPlace(
        m.corner(TopRight, m_rank, dimker)
      );

    // now we must undo the column permutation that we had applied!
    for(int i = m_rank-1; i >= 0; --i)
      m.col(i).swap(m.col(pivots.coeff(i)));

    // see the negative sign in the next line, that's what we were talking about above.
    for(int i = 0; i < m_rank; ++i) dst.row(m_lu.permutationQ().coeff(i)) = -m.row(i).end(dimker);
    for(int i = m_rank; i < cols; ++i) dst.row(m_lu.permutationQ().coeff(i)).setZero();
    for(int k = 0; k < dimker; ++k) dst.coeffRef(m_lu.permutationQ().coeff(m_rank+k), k) = Scalar(1);
  }
};

/***** Implementation of image() *****************************************************/

template<typename MatrixType>
struct ei_traits<ei_lu_image_impl<MatrixType> >
{
  typedef Matrix<
    typename MatrixType::Scalar,
    MatrixType::RowsAtCompileTime, // the image is a subspace of the destination space, whose 
                                   // dimension is the number of rows of the original matrix
    Dynamic,                       // we don't know at compile time the dimension of the image (the rank)
    MatrixType::Options,
    MatrixType::MaxRowsAtCompileTime, // the image matrix will consist of columns from the original matrix,
    MatrixType::MaxColsAtCompileTime  // so it has the same number of rows and at most as many columns.
  > ReturnMatrixType;
};

template<typename MatrixType>
struct ei_lu_image_impl : public ReturnByValue<ei_lu_image_impl<MatrixType> >
{
  typedef LU<MatrixType> LUType;
  typedef typename MatrixType::RealScalar RealScalar;
  const LUType& m_lu;
  int m_rank, m_cols;
  const MatrixType& m_originalMatrix;
  
  ei_lu_image_impl(const LUType& lu, const MatrixType& originalMatrix)
    : m_lu(lu), m_rank(lu.rank()),
      m_cols(m_rank == 0 ? 1 : m_rank),
      m_originalMatrix(originalMatrix) {}

  inline int rows() const { return m_lu.matrixLU().rows(); }
  inline int cols() const { return m_cols; }

  template<typename Dest> void evalTo(Dest& dst) const
  {
    if(m_rank == 0)
    {
      // The Image is just {0}, so it doesn't have a basis properly speaking, but let's
      // avoid crashing/asserting as that depends on floating point calculations. Let's
      // just return a single column vector filled with zeros.
      dst.setZero();
      return;
    }
    
    Matrix<int, Dynamic, 1, 0, LUType::MaxSmallDimAtCompileTime, 1> pivots(m_rank);
    RealScalar premultiplied_threshold = m_lu.maxPivot() * m_lu.threshold();
    int p = 0;
    for(int i = 0; i < m_lu.nonzeroPivots(); ++i)
      if(ei_abs(m_lu.matrixLU().coeff(i,i)) > premultiplied_threshold)
        pivots.coeffRef(p++) = i;
    ei_assert(p == m_rank && "You hit a bug in Eigen! Please report (backtrace and matrix)!");
    
    for(int i = 0; i < m_rank; ++i)
      dst.col(i) = m_originalMatrix.col(m_lu.permutationQ().coeff(pivots.coeff(i)));
  }
};

/***** Implementation of solve() *****************************************************/

template<typename MatrixType,typename Rhs>
struct ei_traits<ei_lu_solve_impl<MatrixType,Rhs> >
{
  typedef Matrix<typename Rhs::Scalar,
                 MatrixType::ColsAtCompileTime,
                 Rhs::ColsAtCompileTime,
                 Rhs::PlainMatrixType::Options,
                 MatrixType::MaxColsAtCompileTime,
                 Rhs::MaxColsAtCompileTime> ReturnMatrixType;
};

template<typename MatrixType, typename Rhs>
struct ei_lu_solve_impl : public ReturnByValue<ei_lu_solve_impl<MatrixType, Rhs> >
{
  typedef typename ei_cleantype<typename Rhs::Nested>::type RhsNested;
  typedef LU<MatrixType> LUType;
  const LUType& m_lu;
  const typename Rhs::Nested m_rhs;
  
  ei_lu_solve_impl(const LUType& lu, const Rhs& rhs)
    : m_lu(lu), m_rhs(rhs)
  {}

  inline int rows() const { return m_lu.matrixLU().cols(); }
  inline int cols() const { return m_rhs.cols(); }

  template<typename Dest> void evalTo(Dest& dst) const
  {
    /* The decomposition PAQ = LU can be rewritten as A = P^{-1} L U Q^{-1}.
    * So we proceed as follows:
    * Step 1: compute c = P * rhs.
    * Step 2: replace c by the solution x to Lx = c. Exists because L is invertible.
    * Step 3: replace c by the solution x to Ux = c. May or may not exist.
    * Step 4: result = Q * c;
    */

    const int rows = m_lu.matrixLU().rows(),
              cols = m_lu.matrixLU().cols(),
              nonzero_pivots = m_lu.nonzeroPivots();
    ei_assert(m_rhs.rows() == rows);
    const int smalldim = std::min(rows, cols);

    if(nonzero_pivots == 0)
    {
      dst.setZero();
      return;
    }
    
    typename Rhs::PlainMatrixType c(m_rhs.rows(), m_rhs.cols());

    // Step 1
    for(int i = 0; i < rows; ++i)
      c.row(m_lu.permutationP().coeff(i)) = m_rhs.row(i);

    // Step 2
    m_lu.matrixLU()
        .corner(Eigen::TopLeft,smalldim,smalldim)
        .template triangularView<UnitLowerTriangular>()
        .solveInPlace(c.corner(Eigen::TopLeft, smalldim, c.cols()));
    if(rows>cols)
    {
      c.corner(Eigen::BottomLeft, rows-cols, c.cols())
        -= m_lu.matrixLU().corner(Eigen::BottomLeft, rows-cols, cols)
         * c.corner(Eigen::TopLeft, cols, c.cols());
    }

    // Step 3
    m_lu.matrixLU()
        .corner(TopLeft, nonzero_pivots, nonzero_pivots)
        .template triangularView<UpperTriangular>()
        .solveInPlace(c.corner(TopLeft, nonzero_pivots, c.cols()));

    // Step 4
    for(int i = 0; i < nonzero_pivots; ++i)
      dst.row(m_lu.permutationQ().coeff(i)) = c.row(i);
    for(int i = nonzero_pivots; i < m_lu.matrixLU().cols(); ++i)
      dst.row(m_lu.permutationQ().coeff(i)).setZero();
  }
};

/******* MatrixBase methods *****************************************************************/

/** \lu_module
  *
  * \return the LU decomposition of \c *this.
  *
  * \sa class LU
  */
template<typename Derived>
inline const LU<typename MatrixBase<Derived>::PlainMatrixType>
MatrixBase<Derived>::lu() const
{
  return LU<PlainMatrixType>(eval());
}

#endif // EIGEN_LU_H
