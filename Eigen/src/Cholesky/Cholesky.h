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
  * matrix A such that A = LL^* = U^*U, where L is lower triangular.
  *
  * While the Cholesky decomposition is particularly useful to solve selfadjoint problems like  D^*D x = b,
  * for that purpose, we recommend the Cholesky decomposition without square root which is more stable
  * and even faster. Nevertheless, this standard Cholesky decomposition remains useful in many other
  * situations like generalised eigen problems with hermitian matrices.
  *
  * Note that during the decomposition, only the upper triangular part of A is considered. Therefore,
  * the strict lower part does not have to store correct values.
  *
  * \sa class CholeskyWithoutSquareRoot
  */
template<typename MatrixType> class Cholesky
{
  public:

    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;

    Cholesky(const MatrixType& matrix)
      : m_matrix(matrix.rows(), matrix.cols())
    {
      compute(matrix);
    }

    Triangular<Lower, MatrixType> matrixL(void) const
    {
      return m_matrix.lower();
    }

    bool isPositiveDefinite(void) const { return m_isPositiveDefinite; }

    template<typename Derived>
    typename Derived::Eval solve(MatrixBase<Derived> &b);

    void compute(const MatrixType& matrix);

  protected:
    /** \internal
      * Used to compute and store L
      * The strict upper part is not used and even not initialized.
      */
    MatrixType m_matrix;
    bool m_isPositiveDefinite;
};

/** Compute / recompute the Cholesky decomposition A = LL^* = U^*U of \a matrix
  */
template<typename MatrixType>
void Cholesky<MatrixType>::compute(const MatrixType& a)
{
  assert(a.rows()==a.cols());
  const int size = a.rows();
  m_matrix.resize(size, size);

  RealScalar x;
  x = ei_real(a.coeff(0,0));
  m_isPositiveDefinite = x > RealScalar(0) && ei_isMuchSmallerThan(ei_imag(m_matrix.coeff(0,0)), RealScalar(1));
  m_matrix.coeffRef(0,0) = ei_sqrt(x);
  m_matrix.col(0).end(size-1) = a.row(0).end(size-1).adjoint() / m_matrix.coeff(0,0);
  for (int j = 1; j < size; ++j)
  {
    Scalar tmp = ei_real(a.coeff(j,j)) - m_matrix.row(j).start(j).norm2();
    x = ei_real(tmp);
    m_isPositiveDefinite = m_isPositiveDefinite && x > RealScalar(0) && ei_isMuchSmallerThan(ei_imag(tmp), RealScalar(1));
    m_matrix.coeffRef(j,j) = x = ei_sqrt(x);

    int endSize = size-j-1;
    if (endSize>0)
      m_matrix.col(j).end(endSize) = (a.row(j).end(endSize).adjoint()
        - m_matrix.block(j+1, 0, endSize, j) * m_matrix.row(j).start(j).adjoint()) / x;
  }
}

/** \returns the solution of A x = \a b using the current decomposition of A.
  * In other words, it returns \code A^-1 b \endcode computing
  * \code L^-*  L^1 b \endcode from right to left.
  */
template<typename MatrixType>
template<typename Derived>
typename Derived::Eval Cholesky<MatrixType>::solve(MatrixBase<Derived> &b)
{
  const int size = m_matrix.rows();
  ei_assert(size==b.size());

  return m_matrix.adjoint().upper().inverseProduct(m_matrix.lower().inverseProduct(b));
}


#endif // EIGEN_CHOLESKY_H
