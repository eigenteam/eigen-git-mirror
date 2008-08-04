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
    typedef Matrix<int, MatrixType::ColsAtCompileTime, 1> IntRowVectorType;
    typedef Matrix<int, MatrixType::RowsAtCompileTime, 1> IntColVectorType;

    LU(const MatrixType& matrix);

    const MatrixType& matrixLU() const
    {
      return m_lu;
    }

    const Part<MatrixType, UnitLower> matrixL() const
    {
      return m_lu;
    }

    const Part<MatrixType, Upper> matrixU() const
    {
      return m_lu;
    }

    const IntColVectorType& permutationP() const
    {
      return m_p;
    }

    const IntRowVectorType& permutationQ() const
    {
      return m_q;
    }

    template<typename OtherDerived>
    typename ProductReturnType<Transpose<MatrixType>, OtherDerived>::Type::Eval
    solve(const MatrixBase<MatrixType> &b) const;

    /**
      * This method returns the determinant of the matrix of which
      * *this is the LU decomposition. It has only linear complexity
      * (that is, O(n) where n is the dimension of the square matrix)
      * as the LU decomposition has already been computed.
      *
      * Warning: a determinant can be very big or small, so for matrices
      * of large dimension (like a 50-by-50 matrix) there can be a risk of
      * overflow/underflow.
      */
    typename ei_traits<MatrixType>::Scalar determinant() const;

  protected:
    MatrixType m_lu;
    IntColVectorType m_p;
    IntRowVectorType m_q;
    int m_det_pq;
    Scalar m_biggest_eigenvalue_of_u;
    int m_dimension_of_kernel;
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

  for(int k = 0; k < size; k++)
  {
    int row_of_biggest, col_of_biggest;
    const Scalar biggest = m_lu.corner(Eigen::BottomRight, rows-k, cols-k)
                               .cwise().abs()
                               .maxCoeff(&row_of_biggest, &col_of_biggest);
    row_of_biggest += k;
    col_of_biggest += k;
    rows_transpositions.coeffRef(k) = row_of_biggest;
    cols_transpositions.coeffRef(k) = col_of_biggest;
    if(k != row_of_biggest) {
      m_lu.row(k).swap(m_lu.row(row_of_biggest));
      number_of_transpositions++;
    }
    if(k != col_of_biggest) {
      m_lu.col(k).swap(m_lu.col(col_of_biggest));
      number_of_transpositions++;
    }
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

  int index_of_biggest;
  m_lu.diagonal().cwise().abs().maxCoeff(&index_of_biggest);
  m_biggest_eigenvalue_of_u = m_lu.diagonal().coeff(index_of_biggest);

  m_dimension_of_kernel = 0;
  for(int k = 0; k < size; k++)
    m_dimension_of_kernel += ei_isMuchSmallerThan(m_lu.diagonal().coeff(k), m_biggest_eigenvalue_of_u);
}

template<typename MatrixType>
typename ei_traits<MatrixType>::Scalar LU<MatrixType>::determinant() const
{
  Scalar res = m_det_pq;
  for(int k = 0; k < m_lu.diagonal().size(); k++) res *= m_lu.diagonal().coeff(k);
  return res;
}

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
