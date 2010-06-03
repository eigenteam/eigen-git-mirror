// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
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
  * \brief Robust Cholesky decomposition of a matrix with pivoting
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
  * Remember that Cholesky decompositions are not rank-revealing. Also, do not use a Cholesky
	* decomposition to determine whether a system of equations has a solution.
  *
  * \sa MatrixBase::ldlt(), class LLT
  */
 /* THIS PART OF THE DOX IS CURRENTLY DISABLED BECAUSE INACCURATE BECAUSE OF BUG IN THE DECOMPOSITION CODE
  * Note that during the decomposition, only the upper triangular part of A is considered. Therefore,
  * the strict lower part does not have to store correct values.
  */
template<typename _MatrixType> class LDLT
{
  public:
    typedef _MatrixType MatrixType;
    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      Options = MatrixType::Options,
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef typename MatrixType::Index Index;
    typedef typename ei_plain_col_type<MatrixType, Index>::type IntColVectorType;
    typedef Matrix<Scalar, RowsAtCompileTime, 1, Options, MaxRowsAtCompileTime, 1> TmpMatrixType;

    /** \brief Default Constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via LDLT::compute(const MatrixType&).
      */
    LDLT() : m_matrix(), m_p(), m_transpositions(), m_isInitialized(false) {}

    /** \brief Default Constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa LDLT()
      */
    LDLT(Index size)
      : m_matrix(size, size),
        m_p(size),
        m_transpositions(size),
        m_temporary(size),
        m_isInitialized(false)
    {}

    LDLT(const MatrixType& matrix)
      : m_matrix(matrix.rows(), matrix.cols()),
        m_p(matrix.rows()),
        m_transpositions(matrix.rows()),
        m_temporary(matrix.rows()),
        m_isInitialized(false)
    {
      compute(matrix);
    }

    /** \returns an expression of the lower triangular matrix L */
    inline TriangularView<MatrixType, UnitLower> matrixL(void) const
    {
      ei_assert(m_isInitialized && "LDLT is not initialized.");
      return m_matrix;
    }

    /** \returns a vector of integers, whose size is the number of rows of the matrix being decomposed,
      * representing the P permutation i.e. the permutation of the rows. For its precise meaning,
      * see the examples given in the documentation of class FullPivLU.
      */
    inline const IntColVectorType& permutationP() const
    {
      ei_assert(m_isInitialized && "LDLT is not initialized.");
      return m_p;
    }

    /** \returns the coefficients of the diagonal matrix D */
    inline Diagonal<MatrixType,0> vectorD(void) const
    {
      ei_assert(m_isInitialized && "LDLT is not initialized.");
      return m_matrix.diagonal();
    }

    /** \returns true if the matrix is positive (semidefinite) */
    inline bool isPositive(void) const
    {
      ei_assert(m_isInitialized && "LDLT is not initialized.");
      return m_sign == 1;
    }

    /** \returns true if the matrix is negative (semidefinite) */
    inline bool isNegative(void) const
    {
      ei_assert(m_isInitialized && "LDLT is not initialized.");
      return m_sign == -1;
    }

    /** \returns a solution x of \f$ A x = b \f$ using the current decomposition of A.
      *
      * \note_about_checking_solutions
      *
      * \sa solveInPlace(), MatrixBase::ldlt()
      */
    template<typename Rhs>
    inline const ei_solve_retval<LDLT, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      ei_assert(m_isInitialized && "LDLT is not initialized.");
      ei_assert(m_matrix.rows()==b.rows()
                && "LDLT::solve(): invalid number of rows of the right hand side matrix b");
      return ei_solve_retval<LDLT, Rhs>(*this, b.derived());
    }

    template<typename Derived>
    bool solveInPlace(MatrixBase<Derived> &bAndX) const;

    LDLT& compute(const MatrixType& matrix);

    /** \returns the internal LDLT decomposition matrix
      *
      * TODO: document the storage layout
      */
    inline const MatrixType& matrixLDLT() const
    {
      ei_assert(m_isInitialized && "LDLT is not initialized.");
      return m_matrix;
    }

    MatrixType reconstructedMatrix() const;

    inline Index rows() const { return m_matrix.rows(); }
    inline Index cols() const { return m_matrix.cols(); }

  protected:

    /** \internal
      * Used to compute and store the Cholesky decomposition A = L D L^* = U^* D U.
      * The strict upper part is used during the decomposition, the strict lower
      * part correspond to the coefficients of L (its diagonal is equal to 1 and
      * is not stored), and the diagonal entries correspond to D.
      */
    MatrixType m_matrix;
    IntColVectorType m_p;
    IntColVectorType m_transpositions; // FIXME do we really need to store permanently the transpositions?
    TmpMatrixType m_temporary;
    Index m_sign;
    bool m_isInitialized;
};

/** Compute / recompute the LDLT decomposition A = L D L^* = U^* D U of \a matrix
  */
template<typename MatrixType>
LDLT<MatrixType>& LDLT<MatrixType>::compute(const MatrixType& a)
{
  ei_assert(a.rows()==a.cols());
  const Index size = a.rows();

  m_matrix = a;

  m_p.resize(size);
  m_transpositions.resize(size);
  m_isInitialized = false;

  if (size <= 1)
  {
    m_p.setZero();
    m_transpositions.setZero();
    m_sign = ei_real(a.coeff(0,0))>0 ? 1:-1;
    m_isInitialized = true;
    return *this;
  }

  RealScalar cutoff = 0, biggest_in_corner;

  // By using a temporary, packet-aligned products are guarenteed. In the LLT
  // case this is unnecessary because the diagonal is included and will always
  // have optimal alignment.
  m_temporary.resize(size);
  for (Index j = 0; j < size; ++j)
  {

    // Find largest diagonal element
    Index index_of_biggest_in_corner;
    biggest_in_corner = m_matrix.diagonal().tail(size-j).cwiseAbs().maxCoeff(&index_of_biggest_in_corner);
    index_of_biggest_in_corner += j;

    if(j == 0)
    {
      // The biggest overall is the point of reference to which further diagonals
      // are compared; if any diagonal is negligible compared
      // to the largest overall, the algorithm bails.
      cutoff = ei_abs(NumTraits<Scalar>::epsilon() * biggest_in_corner);

      m_sign = ei_real(m_matrix.diagonal().coeff(index_of_biggest_in_corner)) > 0 ? 1 : -1;
    }

    // Finish early if the matrix is not full rank.
    if(biggest_in_corner < cutoff)
    {
      for(Index i = j; i < size; i++) m_transpositions.coeffRef(i) = i;
      break;
    }

    m_transpositions.coeffRef(j) = index_of_biggest_in_corner;
    if(j != index_of_biggest_in_corner)
    {
      m_matrix.row(j).swap(m_matrix.row(index_of_biggest_in_corner));
      m_matrix.col(j).swap(m_matrix.col(index_of_biggest_in_corner));
    }

    Index rs = size - j - 1;
    Block<MatrixType,Dynamic,1> A21(m_matrix,j+1,j,rs,1);
    Block<MatrixType,1,Dynamic> A10(m_matrix,j,0,1,j);
    Block<MatrixType,Dynamic,Dynamic> A20(m_matrix,j+1,0,rs,j);

    if(j>0)
    {
      m_temporary.head(j) = m_matrix.diagonal().head(j).asDiagonal() * A10.adjoint();
      m_matrix.coeffRef(j,j) -= (A10 * m_temporary.head(j)).value();
      if(rs>0)
        A21.noalias() -= A20 * m_temporary.head(j);
    }
    if((rs>0) && (ei_abs(m_matrix.coeffRef(j,j)) > cutoff))
      A21 /= m_matrix.coeffRef(j,j);
  }

  // Reverse applied swaps to get P matrix.
  for(Index k = 0; k < size; ++k) m_p.coeffRef(k) = k;
  for(Index k = size-1; k >= 0; --k) {
    std::swap(m_p.coeffRef(k), m_p.coeffRef(m_transpositions.coeff(k)));
  }

  m_isInitialized = true;
  return *this;
}

template<typename _MatrixType, typename Rhs>
struct ei_solve_retval<LDLT<_MatrixType>, Rhs>
  : ei_solve_retval_base<LDLT<_MatrixType>, Rhs>
{
  EIGEN_MAKE_SOLVE_HELPERS(LDLT<_MatrixType>,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dst = rhs();
    dec().solveInPlace(dst);
  }
};

/** This is the \em in-place version of solve().
  *
  * \param bAndX represents both the right-hand side matrix b and result x.
  *
  * \returns true always! If you need to check for existence of solutions, use another decomposition like LU, QR, or SVD.
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
  ei_assert(m_isInitialized && "LDLT is not initialized.");
  const Index size = m_matrix.rows();
  ei_assert(size == bAndX.rows());

  // z = P b
  for(Index i = 0; i < size; ++i) bAndX.row(m_transpositions.coeff(i)).swap(bAndX.row(i));

  // y = L^-1 z
  //matrixL().solveInPlace(bAndX);
  m_matrix.template triangularView<UnitLower>().solveInPlace(bAndX);

  // w = D^-1 y
  bAndX = m_matrix.diagonal().asDiagonal().inverse() * bAndX;

  // u = L^-T w
  m_matrix.adjoint().template triangularView<UnitUpper>().solveInPlace(bAndX);

  // x = P^T u
  for (Index i = size-1; i >= 0; --i) bAndX.row(m_transpositions.coeff(i)).swap(bAndX.row(i));

  return true;
}

/** \returns the matrix represented by the decomposition,
 * i.e., it returns the product: P^T L D L^* P.
 * This function is provided for debug purpose. */
template<typename MatrixType>
MatrixType LDLT<MatrixType>::reconstructedMatrix() const
{
  ei_assert(m_isInitialized && "LDLT is not initialized.");
  const Index size = m_matrix.rows();
  MatrixType res(size,size);
  res.setIdentity();

  // PI
  for(Index i = 0; i < size; ++i) res.row(m_transpositions.coeff(i)).swap(res.row(i));
  // L^* P
  res = matrixL().adjoint() * res;
  // D(L^*P)
  res = vectorD().asDiagonal() * res;
  // L(DL^*P)
  res = matrixL() * res;
  // P^T (LDL^*P)
  for (Index i = size-1; i >= 0; --i) res.row(m_transpositions.coeff(i)).swap(res.row(i));

  return res;
}

/** \cholesky_module
  * \returns the Cholesky decomposition with full pivoting without square root of \c *this
  */
template<typename Derived>
inline const LDLT<typename MatrixBase<Derived>::PlainObject>
MatrixBase<Derived>::ldlt() const
{
  return LDLT<PlainObject>(derived());
}

#endif // EIGEN_LDLT_H
