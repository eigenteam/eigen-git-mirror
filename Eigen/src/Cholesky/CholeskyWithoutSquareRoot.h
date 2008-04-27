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
  * matrix A such that A = U' D U = L D L', where U is upper triangular with a unit diagonal and D is a diagonal
  * matrix.
  *
  * Compared to a standard Cholesky decomposition, avoiding the square roots allows for faster and more
  * stable computation.
  *
  * \todo what about complex matrices ?
  *
  * \sa class Cholesky
  */
template<typename MatrixType> class CholeskyWithoutSquareRoot
{
  public:

    typedef typename MatrixType::Scalar Scalar;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;

    CholeskyWithoutSquareRoot(const MatrixType& matrix)
      : m_matrix(matrix.rows(), matrix.cols())
    {
      compute(matrix);
    }

    Triangular<Lower|UnitDiagBit, MatrixType > matrixL(void) const
    {
      return m_matrix.lowerWithUnitDiag();
    }

    DiagonalCoeffs<MatrixType> vectorD(void) const
    {
      return m_matrix.diagonal();
    }

    bool isPositiveDefinite(void) const
    {
      return m_matrix.diagonal().minCoeff() > Scalar(0);
    }

    template<typename DerivedVec>
    typename DerivedVec::Eval solve(MatrixBase<DerivedVec> &vecB);

    /** Compute / recompute the Cholesky decomposition A = U'DU = LDL' of \a matrix
      */
    void compute(const MatrixType& matrix);

  protected:
    /** \internal
      * Used to compute and store the cholesky decomposition A = U'DU = LDL'.
      * The strict upper part is used during the decomposition, the strict lower
      * part correspond to the coefficients of U'=L (its diagonal is equal to 1 and
      * is not stored), and the diagonal entries correspond to D.
      */
    MatrixType m_matrix;
};

template<typename MatrixType>
void CholeskyWithoutSquareRoot<MatrixType>::compute(const MatrixType& matrix)
{
  assert(matrix.rows()==matrix.cols());
  const int size = matrix.rows();
  m_matrix = matrix.conjugate();
  #if 0
  for (int i = 0; i < size; ++i)
  {
    Scalar tmp = Scalar(1) / m_matrix(i,i);
    for (int j = i+1; j < size; ++j)
    {
      m_matrix(j,i) = m_matrix(i,j) * tmp;
      m_matrix.row(j).end(size-j) -= m_matrix(j,i) * m_matrix.row(i).end(size-j);
    }
  }
  #else
  // this version looks faster for large matrices
  m_matrix.col(0).end(size-1) = m_matrix.row(0).end(size-1) / m_matrix(0,0);
  for (int j = 1; j < size; ++j)
  {
    Scalar tmp = m_matrix(j,j) - (m_matrix.row(j).start(j) * m_matrix.col(j).start(j).conjugate())(0,0);
    m_matrix(j,j) = tmp;
    tmp = Scalar(1) / tmp;
    for (int i = j+1; i < size; ++i)
    {
      m_matrix(j,i) = (m_matrix(j,i) - (m_matrix.row(i).start(j) * m_matrix.col(j).start(j).conjugate())(0,0) );
      m_matrix(i,j) = tmp * m_matrix(j,i);
    }
  }
  #endif
}

/** Solve A*x = b with A symmeric positive definite using the available Cholesky decomposition.
 */
template<typename MatrixType>
template<typename DerivedVec>
typename DerivedVec::Eval CholeskyWithoutSquareRoot<MatrixType>::solve(MatrixBase<DerivedVec> &vecB)
{
  const int size = m_matrix.rows();
  ei_assert(size==vecB.size());

  // FIXME .inverseProduct creates a temporary that is not nice since it is called twice
  // maybe add a .inverseProductInPlace() ??
  return m_matrix.adjoint().upperWithUnitDiag()
    .inverseProduct(
      (m_matrix.lowerWithUnitDiag()
        .inverseProduct(vecB))
        .cwiseQuotient(m_matrix.diagonal())
      );

//   return m_matrix.adjoint().upperWithUnitDiag()
//     .inverseProduct(
//       (m_matrix.lowerWithUnitDiag() * (m_matrix.diagonal().asDiagonal())).lower().inverseProduct(vecB));
}


#endif // EIGEN_CHOLESKY_WITHOUT_SQUARE_ROOT_H
