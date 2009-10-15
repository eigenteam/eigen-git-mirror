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
  * coefficients) of U are sorted in such a way that any zeros are at the end, so that the rank
  * of A is the index of the first zero on the diagonal of U (with indices starting at 0) if any.
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
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
      return m_lu;
    }
    
    /** \returns a pointer to the matrix of which *this is the LU decomposition.
      */
    inline const MatrixType* originalMatrix() const
    {
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
      return m_originalMatrix;
    }
    
    /** \returns a vector of integers, whose size is the number of rows of the matrix being decomposed,
      * representing the P permutation i.e. the permutation of the rows. For its precise meaning,
      * see the examples given in the documentation of class LU.
      *
      * \sa permutationQ()
      */
    inline const IntColVectorType& permutationP() const
    {
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
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
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
      return m_q;
    }

    /** \returns the kernel of the matrix, also called its null-space. The columns of the returned matrix
      * will form a basis of the kernel.
      *
      * \note If the kernel has dimension zero, then the returned matrix is a column-vector filled with zeros.
      *
      * Example: \include LU_kernel.cpp
      * Output: \verbinclude LU_kernel.out
      *
      * \sa image()
      */
    inline const ei_lu_kernel_impl<MatrixType> kernel() const
    {
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
      return ei_lu_kernel_impl<MatrixType>(*this);
    }

    /** \returns the image of the matrix, also called its column-space. The columns of the returned matrix
      * will form a basis of the kernel.
      *
      * \note If the image has dimension zero, then the returned matrix is a column-vector filled with zeros.
      *
      * Example: \include LU_image.cpp
      * Output: \verbinclude LU_image.out
      *
      * \sa kernel()
      */
    inline const ei_lu_image_impl<MatrixType> image() const
    {
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
      return ei_lu_image_impl<MatrixType>(*this);
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
      * \note_about_inexistant_solutions
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
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
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

    /** \returns the rank of the matrix of which *this is the LU decomposition.
      *
      * \note This is computed at the time of the construction of the LU decomposition. This
      *       method does not perform any further computation.
      */
    inline int rank() const
    {
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
      return m_rank;
    }

    /** \returns the dimension of the kernel of the matrix of which *this is the LU decomposition.
      *
      * \note Since the rank is computed at the time of the construction of the LU decomposition, this
      *       method almost does not perform any further computation.
      */
    inline int dimensionOfKernel() const
    {
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
      return m_lu.cols() - m_rank;
    }

    /** \returns true if the matrix of which *this is the LU decomposition represents an injective
      *          linear map, i.e. has trivial kernel; false otherwise.
      *
      * \note Since the rank is computed at the time of the construction of the LU decomposition, this
      *       method almost does not perform any further computation.
      */
    inline bool isInjective() const
    {
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
      return m_rank == m_lu.cols();
    }

    /** \returns true if the matrix of which *this is the LU decomposition represents a surjective
      *          linear map; false otherwise.
      *
      * \note Since the rank is computed at the time of the construction of the LU decomposition, this
      *       method almost does not perform any further computation.
      */
    inline bool isSurjective() const
    {
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
      return m_rank == m_lu.rows();
    }

    /** \returns true if the matrix of which *this is the LU decomposition is invertible.
      *
      * \note Since the rank is computed at the time of the construction of the LU decomposition, this
      *       method almost does not perform any further computation.
      */
    inline bool isInvertible() const
    {
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
      return isInjective() && isSurjective();
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
      ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
      ei_assert(m_lu.rows() == m_lu.cols() && "You can't take the inverse of a non-square matrix!");
      return ei_lu_solve_impl<MatrixType,NestByValue<typename MatrixType::IdentityReturnType> >
               (*this, MatrixType::Identity(m_lu.rows(), m_lu.cols()).nestByValue());
    }

  protected:
    const MatrixType* m_originalMatrix;
    MatrixType m_lu;
    IntColVectorType m_p;
    IntRowVectorType m_q;
    int m_det_pq;
    int m_rank;
    RealScalar m_precision;
};

template<typename MatrixType>
LU<MatrixType>::LU()
  : m_originalMatrix(0),
    m_lu(),
    m_p(),
    m_q(),
    m_det_pq(0),
    m_rank(-1),
    m_precision(precision<RealScalar>())
{
}

template<typename MatrixType>
LU<MatrixType>::LU(const MatrixType& matrix)
  : m_originalMatrix(0),
    m_lu(),
    m_p(),
    m_q(),
    m_det_pq(0),
    m_rank(-1),
    m_precision(precision<RealScalar>())
{
  compute(matrix);
}

template<typename MatrixType>
LU<MatrixType>& LU<MatrixType>::compute(const MatrixType& matrix)
{
  m_originalMatrix = &matrix;
  m_lu = matrix;
  m_p.resize(matrix.rows());
  m_q.resize(matrix.cols());

  const int size = matrix.diagonalSize();
  const int rows = matrix.rows();
  const int cols = matrix.cols();

  // this formula comes from experimenting (see "LU precision tuning" thread on the list)
  // and turns out to be identical to Higham's formula used already in LDLt.
  m_precision = epsilon<Scalar>() * size;

  IntColVectorType rows_transpositions(matrix.rows());
  IntRowVectorType cols_transpositions(matrix.cols());
  int number_of_transpositions = 0;

  RealScalar biggest = RealScalar(0);
  m_rank = size;
  for(int k = 0; k < size; ++k)
  {
    int row_of_biggest_in_corner, col_of_biggest_in_corner;
    RealScalar biggest_in_corner;

    biggest_in_corner = m_lu.corner(Eigen::BottomRight, rows-k, cols-k)
                        .cwise().abs()
                        .maxCoeff(&row_of_biggest_in_corner, &col_of_biggest_in_corner);
    row_of_biggest_in_corner += k;
    col_of_biggest_in_corner += k;
    if(k==0) biggest = biggest_in_corner;

    // if the corner is exactly zero, terminate to avoid generating NaN values
    if(biggest_in_corner == RealScalar(0))
    {
      m_rank = k;
      for(int i = k; i < size; i++)
      {
        rows_transpositions.coeffRef(i) = i;
        cols_transpositions.coeffRef(i) = i;
      }
      break;
    }

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
    if(k<rows-1)
      m_lu.col(k).end(rows-k-1) /= m_lu.coeff(k,k);
    if(k<size-1)
      m_lu.block(k+1,k+1,rows-k-1,cols-k-1).noalias() -= m_lu.col(k).end(rows-k-1) * m_lu.row(k).end(cols-k-1);
  }

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
  ei_assert(m_originalMatrix != 0 && "LU is not initialized.");
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
  const LUType& m_lu;
  int m_dimKer;
  
  ei_lu_kernel_impl(const LUType& lu) : m_lu(lu), m_dimKer(lu.dimensionOfKernel()) {}

  inline int rows() const { return m_lu.matrixLU().cols(); }
  inline int cols() const { return m_dimKer; }

  template<typename Dest> void evalTo(Dest& dst) const
  {
    typedef typename MatrixType::Scalar Scalar;
    const int rank = m_lu.rank(),
              cols = m_lu.matrixLU().cols();
    if(m_dimKer == 0)
    {
      // The Kernel is just {0}, so it doesn't have a basis properly speaking, but let's
      // avoid crashing/asserting as that depends on floating point calculations. Let's
      // just return a single column vector filled with zeros.
      dst.resize(cols,1);
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

    dst.resize(cols, m_dimKer);

    Matrix<typename MatrixType::Scalar, Dynamic, Dynamic, MatrixType::Options,
           MatrixType::MaxColsAtCompileTime, MatrixType::MaxColsAtCompileTime>
      y(-m_lu.matrixLU().corner(TopRight, rank, m_dimKer));

    m_lu.matrixLU()
        .corner(TopLeft, rank, rank)
        .template triangularView<UpperTriangular>().solveInPlace(y);

    for(int i = 0; i < rank; ++i) dst.row(m_lu.permutationQ().coeff(i)) = y.row(i);
    for(int i = rank; i < cols; ++i) dst.row(m_lu.permutationQ().coeff(i)).setZero();
    for(int k = 0; k < m_dimKer; ++k) dst.coeffRef(m_lu.permutationQ().coeff(rank+k), k) = Scalar(1);
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
  const LUType& m_lu;
  
  ei_lu_image_impl(const LUType& lu) : m_lu(lu) {}

  inline int rows() const { return m_lu.matrixLU().cols(); }
  inline int cols() const { return m_lu.rank(); }

  template<typename Dest> void evalTo(Dest& dst) const
  {
    int rank = m_lu.rank();
    if(rank == 0)
    {
      // The Image is just {0}, so it doesn't have a basis properly speaking, but let's
      // avoid crashing/asserting as that depends on floating point calculations. Let's
      // just return a single column vector filled with zeros.
      dst.resize(m_lu.originalMatrix()->rows(), 1);
      dst.setZero();
      return;
    }
    
    dst.resize(m_lu.originalMatrix()->rows(), rank);
    for(int i = 0; i < rank; ++i)
      dst.col(i) = m_lu.originalMatrix()->col(m_lu.permutationQ().coeff(i));
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
    dst.resize(m_lu.matrixLU().cols(), m_rhs.cols());
    if(m_lu.rank()==0)
    {
      dst.setZero();
      return;
    }
    
    /* The decomposition PAQ = LU can be rewritten as A = P^{-1} L U Q^{-1}.
    * So we proceed as follows:
    * Step 1: compute c = P * rhs.
    * Step 2: replace c by the solution x to Lx = c. Exists because L is invertible.
    * Step 3: replace c by the solution x to Ux = c. May or may not exist.
    * Step 4: result = Q * c;
    */

    const int rows = m_lu.matrixLU().rows(),
              cols = m_lu.matrixLU().cols(),
              rank = m_lu.rank();
    ei_assert(m_rhs.rows() == rows);
    const int smalldim = std::min(rows, cols);

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
        .corner(TopLeft, rank, rank)
        .template triangularView<UpperTriangular>()
        .solveInPlace(c.corner(TopLeft, rank, c.cols()));

    // Step 4
    for(int i = 0; i < rank; ++i)
      dst.row(m_lu.permutationQ().coeff(i)) = c.row(i);
    for(int i = rank; i < m_lu.matrixLU().cols(); ++i)
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
