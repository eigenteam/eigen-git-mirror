// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

/** \ingroup LU_Module
  *
  * \class LU
  *
  * \brief LU decomposition of a matrix with complete pivoting, and associated features
  *
  * \param MatrixType the type of the matrix of which we are computing the LU decomposition
  *
  * This class performs a LU decomposition of any matrix, with complete pivoting: the matrix A
  * is decomposed as A = PLUQ where L is unit-lower-triangular, U is upper-triangular, and P and Q
  * are permutation matrices.
  *
  * This decomposition provides the generic approach to solving systems of linear equations, computing
  * the rank, invertibility, inverse, and determinant. However for the case when invertibility is
  * assumed, we have a specialized variant (see MatrixBase::inverse()) achieving better performance.
  *
  * \sa MatrixBase::lu(), MatrixBase::determinant(), MatrixBase::rank(), MatrixBase::kernelDim(),
  *     MatrixBase::kernelBasis(), MatrixBase::solve(), MatrixBase::isInvertible(),
  *     MatrixBase::inverse(), MatrixBase::computeInverse()
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
             MatrixType::MaxRowsAtCompileTime),
           SmallDimAtCompileTime = EIGEN_ENUM_MIN(
             MatrixType::ColsAtCompileTime,
             MatrixType::RowsAtCompileTime),
           MaxBigDimAtCompileTime = EIGEN_ENUM_MAX(
             MatrixType::MaxColsAtCompileTime,
             MatrixType::MaxRowsAtCompileTime),
           BigDimAtCompileTime = EIGEN_ENUM_MAX(
             MatrixType::ColsAtCompileTime,
             MatrixType::RowsAtCompileTime)
    };

    LU(const MatrixType& matrix);

    inline const MatrixType& matrixLU() const
    {
      return m_lu;
    }

    inline const Part<MatrixType, UnitLower> matrixL() const
    {
      return m_lu;
    }

    inline const Part<MatrixType, Upper> matrixU() const
    {
      return m_lu;
    }

    inline const IntColVectorType& permutationP() const
    {
      return m_p;
    }

    inline const IntRowVectorType& permutationQ() const
    {
      return m_q;
    }

    void computeKernel(Matrix<typename MatrixType::Scalar,
                              MatrixType::ColsAtCompileTime, Dynamic,
                              MatrixType::MaxColsAtCompileTime,
                              LU<MatrixType>::MaxSmallDimAtCompileTime
                             > *result) const;

    const Matrix<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, Dynamic,
                 MatrixType::MaxColsAtCompileTime,
                 LU<MatrixType>::MaxSmallDimAtCompileTime> kernel() const;

    template<typename OtherDerived>
    Matrix<typename MatrixType::Scalar,
           MatrixType::ColsAtCompileTime, OtherDerived::ColsAtCompileTime,
           MatrixType::MaxColsAtCompileTime, OtherDerived::MaxColsAtCompileTime
    > solve(MatrixBase<OtherDerived> *b) const;

    /**
      * This method returns the determinant of the matrix of which
      * *this is the LU decomposition. It has only linear complexity
      * (that is, O(n) where n is the dimension of the square matrix)
      * as the LU decomposition has already been computed.
      *
      * Warning: a determinant can be very big or small, so for matrices
      * of large enough dimension (like a 50-by-50 matrix) there is a risk of
      * overflow/underflow.
      */
    typename ei_traits<MatrixType>::Scalar determinant() const;

    inline int rank() const
    {
      return m_rank;
    }

    inline int dimensionOfKernel() const
    {
      return m_lu.cols() - m_rank;
    }

    inline bool isInjective() const
    {
      return m_rank == m_lu.cols();
    }

    inline bool isSurjective() const
    {
      return m_rank == m_lu.rows();
    }

    inline bool isInvertible() const
    {
      return isInjective() && isSurjective();
    }

  protected:
    MatrixType m_lu;
    IntColVectorType m_p;
    IntRowVectorType m_q;
    int m_det_pq;
    int m_rank;
};

template<typename MatrixType>
LU<MatrixType>::LU(const MatrixType& matrix)
  : m_lu(matrix),
    m_p(matrix.rows()),
    m_q(matrix.cols())
{
  const int size = matrix.diagonal().size();
  const int rows = matrix.rows();
  const int cols = matrix.cols();

  IntColVectorType rows_transpositions(matrix.rows());
  IntRowVectorType cols_transpositions(matrix.cols());
  int number_of_transpositions = 0;

  RealScalar biggest;
  for(int k = 0; k < size; k++)
  {
    int row_of_biggest_in_corner, col_of_biggest_in_corner;
    RealScalar biggest_in_corner;

    biggest_in_corner = m_lu.corner(Eigen::BottomRight, rows-k, cols-k)
                  .cwise().abs()
                  .maxCoeff(&row_of_biggest_in_corner, &col_of_biggest_in_corner);
    row_of_biggest_in_corner += k;
    col_of_biggest_in_corner += k;
    rows_transpositions.coeffRef(k) = row_of_biggest_in_corner;
    cols_transpositions.coeffRef(k) = col_of_biggest_in_corner;
    if(k != row_of_biggest_in_corner) {
      m_lu.row(k).swap(m_lu.row(row_of_biggest_in_corner));
      number_of_transpositions++;
    }
    if(k != col_of_biggest_in_corner) {
      m_lu.col(k).swap(m_lu.col(col_of_biggest_in_corner));
      number_of_transpositions++;
    }

    if(k==0) biggest = biggest_in_corner;
    const Scalar lu_k_k = m_lu.coeff(k,k);
    if(ei_isMuchSmallerThan(lu_k_k, biggest)) continue;
    if(k<rows-1)
      m_lu.col(k).end(rows-k-1) /= lu_k_k;
    if(k<size-1)
      for( int col = k + 1; col < cols; col++ )
        m_lu.col(col).end(rows-k-1) -= m_lu.col(k).end(rows-k-1) * m_lu.coeff(k,col);
  }

  for(int k = 0; k < matrix.rows(); k++) m_p.coeffRef(k) = k;
  for(int k = size-1; k >= 0; k--)
    std::swap(m_p.coeffRef(k), m_p.coeffRef(rows_transpositions.coeff(k)));

  for(int k = 0; k < matrix.cols(); k++) m_q.coeffRef(k) = k;
  for(int k = 0; k < size; k++)
    std::swap(m_q.coeffRef(k), m_q.coeffRef(cols_transpositions.coeff(k)));

  m_det_pq = (number_of_transpositions%2) ? -1 : 1;

  for(m_rank = 0; m_rank < size; m_rank++)
    if(ei_isMuchSmallerThan(m_lu.diagonal().coeff(m_rank), m_lu.diagonal().coeff(0)))
      break;
}

template<typename MatrixType>
typename ei_traits<MatrixType>::Scalar LU<MatrixType>::determinant() const
{
  return Scalar(m_det_pq) * m_lu.diagonal().redux(ei_scalar_product_op<Scalar>());
}

template<typename MatrixType>
void LU<MatrixType>::computeKernel(Matrix<typename MatrixType::Scalar,
                                          MatrixType::ColsAtCompileTime, Dynamic,
                                          MatrixType::MaxColsAtCompileTime,
                                          LU<MatrixType>::MaxSmallDimAtCompileTime
                                   > *result) const
{
  ei_assert(!isInvertible());
  const int dimker = dimensionOfKernel(), rows = m_lu.rows(), cols = m_lu.cols();
  result->resize(cols, dimker);

  /* Let us use the following lemma:
    *
    * Lemma: If the matrix A has the LU decomposition PAQ = LU,
    * then Ker A = Q( Ker U ).
    *
    * Proof: trivial: just keep in mind that P, Q, L are invertible.
    */

  /* Thus, all we need to do is to compute Ker U, and then apply Q.
    *
    * U is upper triangular, with eigenvalues sorted in decreasing order of
    * absolute value. Thus, the diagonal of U ends with exactly
    * m_dimKer zero's. Let us use that to construct m_dimKer linearly
    * independent vectors in Ker U.
    */

  Matrix<Scalar, Dynamic, Dynamic, MatrixType::MaxColsAtCompileTime, MaxSmallDimAtCompileTime>
    y(-m_lu.corner(TopRight, m_rank, dimker));

  m_lu.corner(TopLeft, m_rank, m_rank)
      .template marked<Upper>()
      .inverseProductInPlace(y);

  for(int i = 0; i < m_rank; i++)
    result->row(m_q.coeff(i)) = y.row(i);
  for(int i = m_rank; i < cols; i++) result->row(m_q.coeff(i)).setZero();
  for(int k = 0; k < dimker; k++) result->coeffRef(m_q.coeff(m_rank+k), k) = Scalar(1);
}

template<typename MatrixType>
const Matrix<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, Dynamic,
                    MatrixType::MaxColsAtCompileTime,
                    LU<MatrixType>::MaxSmallDimAtCompileTime>
LU<MatrixType>::kernel() const
{
  Matrix<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, Dynamic,
                    MatrixType::MaxColsAtCompileTime,
                    LU<MatrixType>::MaxSmallDimAtCompileTime> result(m_lu.cols(), dimensionOfKernel());
  computeKernel(&result);
  return result;
}

#if 0
template<typename MatrixType>
template<typename OtherDerived>
bool LU<MatrixType>::solve(
  const MatrixBase<OtherDerived>& b,
  Matrix<typename MatrixType::Scalar,
             MatrixType::ColsAtCompileTime, OtherDerived::ColsAtCompileTime,
             MatrixType::MaxColsAtCompileTime, OtherDerived::MaxColsAtCompileTime> *result
) const
{
  /* The decomposition PAQ = LU can be rewritten as A = P^{-1} L U Q^{-1}.
   * So we proceed as follows:
   * Step 1: compute c = Pb.
   * Step 2: replace c by the solution x to Lx = c. Exists because L is invertible.
   * Step 3: compute d such that Ud = c. Check if such d really exists.
   * Step 4: result = Qd;
   */

  typename OtherDerived::Eval c(b.rows(), b.cols());
  Matrix<typename MatrixType::Scalar,
         MatrixType::ColsAtCompileTime, OtherDerived::ColsAtCompileTime,
         MatrixType::MaxColsAtCompileTime, OtherDerived::MaxColsAtCompileTime>
    d(m_lu.cols(), b.cols());

  // Step 1
  for(int i = 0; i < dim(); i++) c.row(m_p.coeff(i)) = b.row(i);

  // Step 2
  Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime,
         MatrixType::MaxRowsAtCompileTime,
         MatrixType::MaxRowsAtCompileTime> l(m_lu.rows(), m_lu.rows());
  l.setIdentity();
  l.corner(Eigen::TopLeft,HEIGHT,SIZE) = lu.matrixL().corner(Eigen::TopLeft,HEIGHT,SIZE);
  l.template marked<UnitLower>.solveInPlace(c);

  // Step 3
  const int bigdim = std::max(m_lu.rows(), m_lu.cols());
  const int smalldim = std::min(m_lu.rows(), m_lu.cols());
  Matrix<Scalar, MatrixType::BigDimAtCompileTime, MatrixType::BigDimAtCompileTime,
         MatrixType::MaxBigDimAtCompileTime,
         MatrixType::MaxBigDimAtCompileTime> u(bigdim, bigdim);
  u.setZero();
  u.corner(TopLeft, smalldim, smalldim) = m_lu.corner(TopLeft, smalldim, smalldim)
                                              .template part<Upper>();
  if(m_lu.cols() > m_lu.rows())
    u.corner(BottomLeft, m_lu.cols()-m_lu.rows(), m_lu.cols()).setZero();
  const int size = std::min(m_lu.rows(), m_lu.cols());
  for(int i = size-1; i >= m_rank; i--)
  {
    if(c.row(i).isMuchSmallerThan(ei_abs(m_lu.coeff(0,0))))
    {
      d.row(i).setConstant(Scalar(1));
    }
    else return false;
  }
  for(int i = m_rank-1; i >= 0; i--)
  {
    d.row(i) = c.row(i);
    for( int j = i + 1; j <= dim() - 1; j++ )
    {
        rowptr += dim();
        b[i] -= b[j] * (*rowptr);
    }
    b[i] /= *denomptr;
  }
}
#endif

/** \return the LU decomposition of \c *this.
  *
  * \sa class LU
  */
template<typename Derived>
const LU<typename MatrixBase<Derived>::EvalType>
MatrixBase<Derived>::lu() const
{
  return eval();
}

#endif // EIGEN_LU_H
