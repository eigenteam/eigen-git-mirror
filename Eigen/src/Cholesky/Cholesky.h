// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_CHOLESKY_H
#define EIGEN_CHOLESKY_H

/** \class Cholesky
  *
  * \brief Standard Cholesky decomposition of a matrix and associated features
  *
  * \param MatrixType the type of the matrix of which we are computing the Cholesky decomposition
  *
  * This class performs a standard Cholesky decomposition of a symmetric, positive definite
  * matrix A such that A = U'U = LL', where U is upper triangular.
  *
  * While the Cholesky decomposition is particularly useful to solve selfadjoint problems like  A'A x = b,
  * for that purpose, we recommend the Cholesky decomposition without square root which is more stable
  * and even faster. Nevertheless, this standard Cholesky decomposition remains useful in many other
  * situation like generalised eigen problem with hermitian matrices.
  *
  * \sa class CholeskyWithoutSquareRoot
  */
template<typename MatrixType> class Cholesky
{
  public:

    typedef typename MatrixType::Scalar Scalar;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;

    Cholesky(const MatrixType& matrix)
      : m_matrix(matrix.rows(), matrix.cols())
    {
      compute(matrix);
    }

    Triangular<Upper, Temporary<Transpose<MatrixType> > > matrixU(void) const
    {
      return m_matrix.transpose().temporary().upper();
    }

    Triangular<Lower, MatrixType> matrixL(void) const
    {
      return m_matrix.lower();
    }

    bool isPositiveDefinite(void) const { return m_isPositiveDefinite; }

    template<typename DerivedVec>
    typename DerivedVec::Eval solve(MatrixBase<DerivedVec> &vecB);

    /** Compute / recompute the Cholesky decomposition A = U'U = LL' of \a matrix
      */
    void compute(const MatrixType& matrix);

  protected:
    /** \internal
      * Used to compute and store the cholesky decomposition.
      * The strict upper part correspond to the coefficients of the input
      * symmetric matrix, while the lower part store U'=L.
      */
    MatrixType m_matrix;
    bool m_isPositiveDefinite;
};

template<typename MatrixType>
void Cholesky<MatrixType>::compute(const MatrixType& matrix)
{
  assert(matrix.rows()==matrix.cols());
  const int size = matrix.rows();
  m_matrix = matrix;

  #if 1
  // this version looks faster for large matrices
  m_isPositiveDefinite = m_matrix(0,0) > Scalar(0);
  m_matrix(0,0) = ei_sqrt(m_matrix(0,0));
  m_matrix.col(0).end(size-1) = m_matrix.row(0).end(size-1) / m_matrix(0,0);
  for (int j = 1; j < size; ++j)
  {
    Scalar tmp = m_matrix(j,j) - m_matrix.row(j).start(j).norm2();
    m_isPositiveDefinite = m_isPositiveDefinite && tmp > Scalar(0);
    m_matrix(j,j) = ei_sqrt(tmp<Scalar(0) ? Scalar(0) : tmp);
    tmp = Scalar(1) / m_matrix(j,j);
    for (int i = j+1; i < size; ++i)
      m_matrix(i,j) = tmp * (m_matrix(j,i) -
          (m_matrix.row(i).start(j) * m_matrix.row(j).start(j).transpose())(0,0) );
  }
  #else
  m_isPositiveDefinite = true;
  for (int i = 0; i < size; ++i)
  {
    m_isPositiveDefinite = m_isPositiveDefinite && m_matrix(i,i) > Scalar(0);
    m_matrix(i,i) = ei_sqrt(m_matrix(i,i));
    if (i+1<size)
      m_matrix.col(i).end(size-i-1) /= m_matrix(i,i);
    for (int j = i+1; j < size; ++j)
    {
      m_matrix.col(j).end(size-j) -= m_matrix(j,i) * m_matrix.col(i).end(size-j);
    }
  }
  #endif
}

/** Solve A*x = b with A symmeric positive definite using the available Cholesky decomposition.
 */
template<typename MatrixType>
template<typename DerivedVec>
typename DerivedVec::Eval Cholesky<MatrixType>::solve(MatrixBase<DerivedVec> &vecB)
{
  const int size = m_matrix.rows();
  ei_assert(size==vecB.size());

  // FIXME .inverseProduct creates a temporary that is not nice since it is called twice
  // add a .inverseProductInPlace ??
  return m_matrix.transpose().upper()
    .inverseProduct(m_matrix.lower().inverseProduct(vecB));
}


#endif // EIGEN_CHOLESKY_H
