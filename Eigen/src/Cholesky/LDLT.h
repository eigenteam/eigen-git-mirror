// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2009 Keir Mierle <mierle@gmail.com>
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

#ifndef EIGEN_LDLT_H
#define EIGEN_LDLT_H

/** \ingroup cholesky_Module
  *
  * \class LDLT
  *
  * \brief Robust Cholesky decomposition of a matrix
  *
  * \param MatrixType the type of the matrix of which to compute the LDL^T Cholesky decomposition
  *
  * Perform a robust Cholesky decomposition of a positive semidefinite or negative semidefinite
  * matrix \f$ A \f$ such that \f$ A =  P^TLDL^*P \f$, where P is a permutation matrix, L
  * is lower triangular with a unit diagonal and D is a diagonal matrix.
  *
  * The decomposition uses pivoting to ensure stability, so that L will have
  * zeros in the bottom right rank(A) - n submatrix. Avoiding the square root
  * on D also stabilizes the computation.
  *
  * \sa MatrixBase::ldlt(), class LLT
  */
 /* THIS PART OF THE DOX IS CURRENTLY DISABLED BECAUSE INACCURATE BECAUSE OF BUG IN THE DECOMPOSITION CODE
  * Note that during the decomposition, only the upper triangular part of A is considered. Therefore,
  * the strict lower part does not have to store correct values.
  */
template<typename MatrixType> class LDLT
{
  public:

    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;
    typedef Matrix<int, MatrixType::RowsAtCompileTime, 1> IntColVectorType;
    typedef Matrix<int, 1, MatrixType::RowsAtCompileTime> IntRowVectorType;

    LDLT(const MatrixType& matrix)
      : m_matrix(matrix.rows(), matrix.cols()),
        m_p(matrix.rows()),
        m_transpositions(matrix.rows())
    {
      compute(matrix);
    }

    /** \returns the lower triangular matrix L */
    inline Part<MatrixType, UnitLowerTriangular> matrixL(void) const { return m_matrix; }

    /** \returns a vector of integers, whose size is the number of rows of the matrix being decomposed,
      * representing the P permutation i.e. the permutation of the rows. For its precise meaning,
      * see the examples given in the documentation of class LU.
      */
    inline const IntColVectorType& permutationP() const
    {
      return m_p;
    }

    /** \returns the coefficients of the diagonal matrix D */
    inline DiagonalCoeffs<MatrixType> vectorD(void) const { return m_matrix.diagonal(); }

    /** \returns true if the matrix is positive (semidefinite) */
    inline bool isPositive(void) const { return m_sign == 1; }

    /** \returns true if the matrix is negative (semidefinite) */
    inline bool isNegative(void) const { return m_sign == -1; }

    /** \returns true if the matrix is invertible */
    inline bool isInvertible(void) const { return m_rank == m_matrix.rows(); }

    /** \returns true if the matrix is positive definite */
    inline bool isPositiveDefinite(void) const { return isPositive() && isInvertible(); }

    /** \returns true if the matrix is negative definite */
    inline bool isNegativeDefinite(void) const { return isNegative() && isInvertible(); }

    /** \returns the rank of the matrix of which *this is the LDLT decomposition.
      *
      * \note This is computed at the time of the construction of the LDLT decomposition. This
      *       method does not perform any further computation.
      */
    inline int rank() const
    {
      return m_rank;
    }

    template<typename RhsDerived, typename ResDerived>
    bool solve(const MatrixBase<RhsDerived> &b, MatrixBase<ResDerived> *result) const;

    template<typename Derived>
    bool solveInPlace(MatrixBase<Derived> &bAndX) const;

    void compute(const MatrixType& matrix);

  protected:
    /** \internal
      * Used to compute and store the Cholesky decomposition A = L D L^* = U^* D U.
      * The strict upper part is used during the decomposition, the strict lower
      * part correspond to the coefficients of L (its diagonal is equal to 1 and
      * is not stored), and the diagonal entries correspond to D.
      */
    MatrixType m_matrix;
    IntColVectorType m_p;
    IntColVectorType m_transpositions;
    int m_rank, m_sign;
};

/** Compute / recompute the LDLT decomposition A = L D L^* = U^* D U of \a matrix
  */
template<typename MatrixType>
void LDLT<MatrixType>::compute(const MatrixType& a)
{
  ei_assert(a.rows()==a.cols());
  const int size = a.rows();
  m_rank = size;

  m_matrix = a;

  if (size <= 1) {
    m_p.setZero();
    m_transpositions.setZero();
    m_sign = ei_real(a.coeff(0,0))>0 ? 1:-1;
    return;
  }

  RealScalar cutoff = 0, biggest_in_corner;

  // By using a temorary, packet-aligned products are guarenteed. In the LLT
  // case this is unnecessary because the diagonal is included and will always
  // have optimal alignment.
  Matrix<Scalar,MatrixType::RowsAtCompileTime,1> _temporary(size);

  for (int j = 0; j < size; ++j)
  {
    // Find largest diagonal element
    int index_of_biggest_in_corner;
    biggest_in_corner = m_matrix.diagonal().end(size-j).cwise().abs()
                       .maxCoeff(&index_of_biggest_in_corner);
    index_of_biggest_in_corner += j;

    if(j == 0)
    {
      // The biggest overall is the point of reference to which further diagonals
      // are compared; if any diagonal is negligible compared
      // to the largest overall, the algorithm bails.  This cutoff is suggested
      // in "Analysis of the Cholesky Decomposition of a Semi-definite Matrix" by
      // Nicholas J. Higham. Also see "Accuracy and Stability of Numerical
      // Algorithms" page 208, also by Higham.
      cutoff = ei_abs(precision<RealScalar>() * size * biggest_in_corner);

      m_sign = ei_real(m_matrix.diagonal().coeff(index_of_biggest_in_corner)) > 0 ? 1 : -1;
    }

    // Finish early if the matrix is not full rank.
    if(biggest_in_corner < cutoff)
    {
      for(int i = j; i < size; i++) m_transpositions.coeffRef(i) = i;
      m_rank = j;
      break;
    }

    m_transpositions.coeffRef(j) = index_of_biggest_in_corner;
    if(j != index_of_biggest_in_corner)
    {
      m_matrix.row(j).swap(m_matrix.row(index_of_biggest_in_corner));
      m_matrix.col(j).swap(m_matrix.col(index_of_biggest_in_corner));
    }

    if (j == 0) {
      m_matrix.row(0) = m_matrix.row(0).conjugate();
      m_matrix.col(0).end(size-1) = m_matrix.row(0).end(size-1) / m_matrix.coeff(0,0);
      continue;
    }

    RealScalar Djj = ei_real(m_matrix.coeff(j,j) - (m_matrix.row(j).start(j)
                                                  * m_matrix.col(j).start(j).conjugate()).coeff(0,0));
    m_matrix.coeffRef(j,j) = Djj;

    // Finish early if the matrix is not full rank.
    if(ei_abs(Djj) < cutoff) // i made experiments, this is better than isMuchSmallerThan(biggest_in_corner), and of course
                             // much better than plain sign comparison as used to be done before.
    {
      for(int i = j; i < size; i++) m_transpositions.coeffRef(i) = i;
      m_rank = j;
      break;
    }

    int endSize = size - j - 1;
    if (endSize > 0) {
      _temporary.end(endSize) = ( m_matrix.block(j+1,0, endSize, j)
                                * m_matrix.col(j).start(j).conjugate() ).lazy();

      m_matrix.row(j).end(endSize) = m_matrix.row(j).end(endSize).conjugate()
                                   - _temporary.end(endSize).transpose();

      m_matrix.col(j).end(endSize) = m_matrix.row(j).end(endSize) / Djj;
    }
  }

  // Reverse applied swaps to get P matrix.
  for(int k = 0; k < size; ++k) m_p.coeffRef(k) = k;
  for(int k = size-1; k >= 0; --k) {
    std::swap(m_p.coeffRef(k), m_p.coeffRef(m_transpositions.coeff(k)));
  }
}

/** Computes the solution x of \f$ A x = b \f$ using the current decomposition of A.
  * The result is stored in \a result
  *
  * \returns true in case of success, false otherwise.
  *
  * In other words, it computes \f$ b = A^{-1} b \f$ with
  * \f$ P^T{L^{*}}^{-1} D^{-1} L^{-1} P b \f$ from right to left.
  *
  * \sa LDLT::solveInPlace(), MatrixBase::ldlt()
  */
template<typename MatrixType>
template<typename RhsDerived, typename ResDerived>
bool LDLT<MatrixType>
::solve(const MatrixBase<RhsDerived> &b, MatrixBase<ResDerived> *result) const
{
  const int size = m_matrix.rows();
  ei_assert(size==b.rows() && "LDLT::solve(): invalid number of rows of the right hand side matrix b");
  *result = b;
  return solveInPlace(*result);
}

/** This is the \em in-place version of solve().
  *
  * \param bAndX represents both the right-hand side matrix b and result x.
  *
  * This version avoids a copy when the right hand side matrix b is not
  * needed anymore.
  *
  * \sa LDLT::solve(), MatrixBase::ldlt()
  */
template<typename MatrixType>
template<typename Derived>
bool LDLT<MatrixType>::solveInPlace(MatrixBase<Derived> &bAndX) const
{
  const int size = m_matrix.rows();
  ei_assert(size == bAndX.rows());

  if (m_rank != size) return false;

  // z = P b
  for(int i = 0; i < size; ++i) bAndX.row(m_transpositions.coeff(i)).swap(bAndX.row(i));

  // y = L^-1 z
  matrixL().solveTriangularInPlace(bAndX);

  // w = D^-1 y
  bAndX = (m_matrix.diagonal().cwise().inverse().asDiagonal() * bAndX).lazy();

  // u = L^-T w
  m_matrix.adjoint().template part<UnitUpperTriangular>().solveTriangularInPlace(bAndX);

  // x = P^T u
  for (int i = size-1; i >= 0; --i) bAndX.row(m_transpositions.coeff(i)).swap(bAndX.row(i));

  return true;
}

/** \cholesky_module
  * \returns the Cholesky decomposition with full pivoting without square root of \c *this
  */
template<typename Derived>
inline const LDLT<typename MatrixBase<Derived>::PlainMatrixType>
MatrixBase<Derived>::ldlt() const
{
  return derived();
}

#endif // EIGEN_LDLT_H
