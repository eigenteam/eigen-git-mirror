// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

    /**
    * \brief Default Constructor.
    *
    * The default constructor is useful in cases in which the user intends to
    * perform decompositions via PartialLU::compute(const MatrixType&).
    */
    PartialLU();

    /** Constructor.
      *
      * \param matrix the matrix of which to compute the LU decomposition.
      *
      * \warning The matrix should have full rank (e.g. if it's square, it should be invertible).
      * If you need to deal with non-full rank, use class LU instead.
      */
    PartialLU(const MatrixType& matrix);

    PartialLU& compute(const MatrixType& matrix);

    /** \returns the LU decomposition matrix: the upper-triangular part is U, the
      * unit-lower-triangular part is L (at least for square matrices; in the non-square
      * case, special care is needed, see the documentation of class LU).
      *
      * \sa matrixL(), matrixU()
      */
    inline const MatrixType& matrixLU() const
    {
      ei_assert(m_isInitialized && "PartialLU is not initialized.");
      return m_lu;
    }

    /** \returns a vector of integers, whose size is the number of rows of the matrix being decomposed,
      * representing the P permutation i.e. the permutation of the rows. For its precise meaning,
      * see the examples given in the documentation of class LU.
      */
    inline const IntColVectorType& permutationP() const
    {
      ei_assert(m_isInitialized && "PartialLU is not initialized.");
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
      * \sa TriangularView::solve(), inverse(), computeInverse()
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
    MatrixType m_lu;
    IntColVectorType m_p;
    int m_det_p;
    bool m_isInitialized;
};

template<typename MatrixType>
PartialLU<MatrixType>::PartialLU()
  : m_lu(),
    m_p(),
    m_det_p(0),
    m_isInitialized(false)
{
}

template<typename MatrixType>
PartialLU<MatrixType>::PartialLU(const MatrixType& matrix)
  : m_lu(),
    m_p(),
    m_det_p(0),
    m_isInitialized(false)
{
  compute(matrix);
}



/** This is the blocked version of ei_lu_unblocked() */
template<typename Scalar, int StorageOrder>
struct ei_partial_lu_impl
{
  // FIXME add a stride to Map, so that the following mapping becomes easier,
  // another option would be to create an expression being able to automatically
  // warp any Map, Matrix, and Block expressions as a unique type, but since that's exactly
  // a Map + stride, why not adding a stride to Map, and convenient ctors from a Matrix,
  // and Block.
  typedef Map<Matrix<Scalar, Dynamic, Dynamic, StorageOrder> > MapLU;
  typedef Block<MapLU, Dynamic, Dynamic> MatrixType;
  typedef Block<MatrixType,Dynamic,Dynamic> BlockType;
  typedef typename MatrixType::RealScalar RealScalar;

  /** \internal performs the LU decomposition in-place of the matrix \a lu
    * using an unblocked algorithm.
    *
    * In addition, this function returns the row transpositions in the
    * vector \a row_transpositions which must have a size equal to the number
    * of columns of the matrix \a lu, and an integer \a nb_transpositions
    * which returns the actual number of transpositions.
    *
    * \returns false if some pivot is exactly zero, in which case the matrix is left with
    *          undefined coefficients (to avoid generating inf/nan values). Returns true
    *          otherwise.
    */
  static bool unblocked_lu(MatrixType& lu, int* row_transpositions, int& nb_transpositions)
  {
    const int rows = lu.rows();
    const int size = std::min(lu.rows(),lu.cols());
    nb_transpositions = 0;
    for(int k = 0; k < size; ++k)
    {
      int row_of_biggest_in_col;
      RealScalar biggest_in_corner
        = lu.col(k).end(rows-k).cwise().abs().maxCoeff(&row_of_biggest_in_col);
      row_of_biggest_in_col += k;

      if(biggest_in_corner == 0) // the pivot is exactly zero: the matrix is singular
      {
        // end quickly, avoid generating inf/nan values. Although in this unblocked_lu case
        // the result is still valid, there's no need to boast about it because
        // the blocked_lu code can't guarantee the same.
        // before exiting, make sure to initialize the still uninitialized row_transpositions
        // in a sane state without destroying what we already have.
        for(int i = k; i < size; i++)
          row_transpositions[i] = i;
        return false;
      }

      row_transpositions[k] = row_of_biggest_in_col;

      if(k != row_of_biggest_in_col)
      {
        lu.row(k).swap(lu.row(row_of_biggest_in_col));
        ++nb_transpositions;
      }

      if(k<rows-1)
      {
        int rrows = rows-k-1;
        int rsize = size-k-1;
        lu.col(k).end(rrows) /= lu.coeff(k,k);
        lu.corner(BottomRight,rrows,rsize).noalias() -= lu.col(k).end(rrows) * lu.row(k).end(rsize);
      }
    }
    return true;
  }

  /** \internal performs the LU decomposition in-place of the matrix represented
    * by the variables \a rows, \a cols, \a lu_data, and \a lu_stride using a
    * recursive, blocked algorithm.
    *
    * In addition, this function returns the row transpositions in the
    * vector \a row_transpositions which must have a size equal to the number
    * of columns of the matrix \a lu, and an integer \a nb_transpositions
    * which returns the actual number of transpositions.
    *
    * \returns false if some pivot is exactly zero, in which case the matrix is left with
    *          undefined coefficients (to avoid generating inf/nan values). Returns true
    *          otherwise.
    *
    * \note This very low level interface using pointers, etc. is to:
    *   1 - reduce the number of instanciations to the strict minimum
    *   2 - avoid infinite recursion of the instanciations with Block<Block<Block<...> > >
    */
  static bool blocked_lu(int rows, int cols, Scalar* lu_data, int luStride, int* row_transpositions, int& nb_transpositions, int maxBlockSize=256)
  {
    MapLU lu1(lu_data,StorageOrder==RowMajor?rows:luStride,StorageOrder==RowMajor?luStride:cols);
    MatrixType lu(lu1,0,0,rows,cols);

    const int size = std::min(rows,cols);

    // if the matrix is too small, no blocking:
    if(size<=16)
    {
      return unblocked_lu(lu, row_transpositions, nb_transpositions);
    }

    // automatically adjust the number of subdivisions to the size
    // of the matrix so that there is enough sub blocks:
    int blockSize;
    {
      blockSize = size/8;
      blockSize = (blockSize/16)*16;
      blockSize = std::min(std::max(blockSize,8), maxBlockSize);
    }

    nb_transpositions = 0;
    for(int k = 0; k < size; k+=blockSize)
    {
      int bs = std::min(size-k,blockSize); // actual size of the block
      int trows = rows - k - bs; // trailing rows
      int tsize = size - k - bs; // trailing size

      // partition the matrix:
      //        A00 | A01 | A02
      // lu  =  A10 | A11 | A12
      //        A20 | A21 | A22
      BlockType A_0(lu,0,0,rows,k);
      BlockType A_2(lu,0,k+bs,rows,tsize);
      BlockType A11(lu,k,k,bs,bs);
      BlockType A12(lu,k,k+bs,bs,tsize);
      BlockType A21(lu,k+bs,k,trows,bs);
      BlockType A22(lu,k+bs,k+bs,trows,tsize);

      int nb_transpositions_in_panel;
      // recursively calls the blocked LU algorithm with a very small
      // blocking size:
      if(!blocked_lu(trows+bs, bs, &lu.coeffRef(k,k), luStride,
                   row_transpositions+k, nb_transpositions_in_panel, 16))
      {
        // end quickly with undefined coefficients, just avoid generating inf/nan values.
        // before exiting, make sure to initialize the still uninitialized row_transpositions
        // in a sane state without destroying what we already have.
        for(int i=k; i<size; ++i)
          row_transpositions[i] = i;
        return false;
      }
      nb_transpositions += nb_transpositions_in_panel;

      // update permutations and apply them to A10
      for(int i=k; i<k+bs; ++i)
      {
        int piv = (row_transpositions[i] += k);
        A_0.row(i).swap(A_0.row(piv));
      }

      if(trows)
      {
        // apply permutations to A_2
        for(int i=k;i<k+bs; ++i)
          A_2.row(i).swap(A_2.row(row_transpositions[i]));

        // A12 = A11^-1 A12
        A11.template triangularView<UnitLowerTriangular>().solveInPlace(A12);

        A22 -= A21 * A12;
      }
    }
    return true;
  }
};

/** \internal performs the LU decomposition with partial pivoting in-place.
  */
template<typename MatrixType, typename IntVector>
void ei_partial_lu_inplace(MatrixType& lu, IntVector& row_transpositions, int& nb_transpositions)
{
  ei_assert(lu.cols() == row_transpositions.size());
  ei_assert((&row_transpositions.coeffRef(1)-&row_transpositions.coeffRef(0)) == 1);

  ei_partial_lu_impl
    <typename MatrixType::Scalar, MatrixType::Flags&RowMajorBit?RowMajor:ColMajor>
    ::blocked_lu(lu.rows(), lu.cols(), &lu.coeffRef(0,0), lu.stride(), &row_transpositions.coeffRef(0), nb_transpositions);
}

template<typename MatrixType>
PartialLU<MatrixType>& PartialLU<MatrixType>::compute(const MatrixType& matrix)
{
  m_lu = matrix;
  m_p.resize(matrix.rows());

  ei_assert(matrix.rows() == matrix.cols() && "PartialLU is only for square (and moreover invertible) matrices");
  const int size = matrix.rows();

  IntColVectorType rows_transpositions(size);

  int nb_transpositions;
  ei_partial_lu_inplace(m_lu, rows_transpositions, nb_transpositions);
  m_det_p = (nb_transpositions%2) ? -1 : 1;

  for(int k = 0; k < size; ++k) m_p.coeffRef(k) = k;
  for(int k = size-1; k >= 0; --k)
    std::swap(m_p.coeffRef(k), m_p.coeffRef(rows_transpositions.coeff(k)));

  m_isInitialized = true;
  return *this;
}

template<typename MatrixType>
typename ei_traits<MatrixType>::Scalar PartialLU<MatrixType>::determinant() const
{
  ei_assert(m_isInitialized && "PartialLU is not initialized.");
  return Scalar(m_det_p) * m_lu.diagonal().prod();
}

template<typename MatrixType>
template<typename OtherDerived, typename ResultType>
void PartialLU<MatrixType>::solve(
  const MatrixBase<OtherDerived>& b,
  ResultType *result
) const
{
  ei_assert(m_isInitialized && "PartialLU is not initialized.");

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
  m_lu.template triangularView<UnitLowerTriangular>().solveInPlace(*result);

  // Step 3
  m_lu.template triangularView<UpperTriangular>().solveInPlace(*result);
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
