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

#ifndef EIGEN_CHOLESKY_WITHOUT_SQUARE_ROOT_H
#define EIGEN_CHOLESKY_WITHOUT_SQUARE_ROOT_H

/** \class CholeskyWithoutSquareRoot
  *
  * \brief Robust Cholesky decomposition of a matrix and associated features
  *
  * \param MatrixType the type of the matrix of which we are computing the Cholesky decomposition
  *
  * This class performs a Cholesky decomposition without square root of a symmetric, positive definite
  * matrix A such that A = L D L^* = U^* D U, where L is lower triangular with a unit diagonal and D is a diagonal
  * matrix.
  *
  * Compared to a standard Cholesky decomposition, avoiding the square roots allows for faster and more
  * stable computation.
  *
  * Note that during the decomposition, only the upper triangular part of A is considered. Therefore,
  * the strict lower part does not have to store correct values.
  *
  * \sa class Cholesky
  */
template<typename MatrixType> class CholeskyWithoutSquareRoot
{
  public:

    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;

    CholeskyWithoutSquareRoot(const MatrixType& matrix)
      : m_matrix(matrix.rows(), matrix.cols())
    {
      compute(matrix);
    }

    /** \returns the lower triangular matrix L */
    Triangular<Lower|UnitDiagBit, MatrixType > matrixL(void) const
    {
      return m_matrix.lowerWithUnitDiag();
    }

    /** \returns the coefficients of the diagonal matrix D */
    DiagonalCoeffs<MatrixType> vectorD(void) const
    {
      return m_matrix.diagonal();
    }

    /** \returns whether the matrix is positive definite */
    bool isPositiveDefinite(void) const
    {
      return m_matrix.diagonal().minCoeff() > Scalar(0);
    }

    template<typename Derived>
    typename Derived::Eval solve(MatrixBase<Derived> &b);

    void compute(const MatrixType& matrix);

  protected:
    /** \internal
      * Used to compute and store the cholesky decomposition A = L D L^* = U^* D U.
      * The strict upper part is used during the decomposition, the strict lower
      * part correspond to the coefficients of L (its diagonal is equal to 1 and
      * is not stored), and the diagonal entries correspond to D.
      */
    MatrixType m_matrix;
};

/** Compute / recompute the Cholesky decomposition A = L D L^* = U^* D U of \a matrix
  */
template<typename MatrixType>
void CholeskyWithoutSquareRoot<MatrixType>::compute(const MatrixType& a)
{
  assert(a.rows()==a.cols());
  const int size = a.rows();
  m_matrix.resize(size, size);

  // Note that, in this algorithm the rows of the strict upper part of m_matrix is used to store
  // column vector, thus the strange .conjugate() and .transpose()...

  m_matrix.row(0) = a.row(0).conjugate();
  m_matrix.col(0).end(size-1) = m_matrix.row(0).end(size-1) / m_matrix.coeff(0,0);
  for (int j = 1; j < size; ++j)
  {
    RealScalar tmp = ei_real(a.coeff(j,j) - (m_matrix.row(j).start(j) * m_matrix.col(j).start(j).conjugate()).coeff(0,0));
    m_matrix.coeffRef(j,j) = tmp;

    int endSize = size-j-1;
    if (endSize>0)
    {
      m_matrix.row(j).end(endSize) = a.row(j).end(endSize).conjugate()
        - (m_matrix.block(j+1,0, endSize, j) * m_matrix.col(j).start(j).conjugate()).transpose();
      m_matrix.col(j).end(endSize) = m_matrix.row(j).end(endSize) / tmp;
    }
  }
}

/** \returns the solution of A x = \a b using the current decomposition of A.
  * In other words, it returns \code A^-1 b \endcode computing
  * \code (L^-*) (D^-1) (L^-1) b \code from right to left.
  */
template<typename MatrixType>
template<typename Derived>
typename Derived::Eval CholeskyWithoutSquareRoot<MatrixType>::solve(MatrixBase<Derived> &vecB)
{
  const int size = m_matrix.rows();
  ei_assert(size==vecB.size());

  return m_matrix.adjoint().upperWithUnitDiag()
    .inverseProduct(
      (m_matrix.lowerWithUnitDiag()
        .inverseProduct(vecB))
        .cwiseQuotient(m_matrix.diagonal())
      );
}


#endif // EIGEN_CHOLESKY_WITHOUT_SQUARE_ROOT_H
