// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_INVERSE_H
#define EIGEN_INVERSE_H

/** \lu_module
  *
  * \class Inverse
  *
  * \brief Inverse of a matrix
  *
  * \param MatrixType the type of the matrix of which we are taking the inverse
  * \param CheckExistence whether or not to check the existence of the inverse while computing it
  *
  * This class represents the inverse of a matrix. It is the return
  * type of MatrixBase::inverse() and most of the time this is the only way it
  * is used.
  *
  * \sa MatrixBase::inverse(), MatrixBase::quickInverse()
  */
template<typename MatrixType, bool CheckExistence>
struct ei_traits<Inverse<MatrixType, CheckExistence> >
{
  typedef typename MatrixType::Scalar Scalar;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    Flags = MatrixType::Flags,
    CoeffReadCost = MatrixType::CoeffReadCost
  };
};

template<typename MatrixType, bool CheckExistence> class Inverse : ei_no_assignment_operator,
  public MatrixBase<Inverse<MatrixType, CheckExistence> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Inverse)

    Inverse(const MatrixType& matrix)
      : m_inverse(MatrixType::identity(matrix.rows(), matrix.cols()))
    {
      if(CheckExistence) m_exists = true;
      ei_assert(matrix.rows() == matrix.cols());
      _compute(matrix);
    }

    /** \returns whether or not the inverse exists.
      *
      * \note This method is only available if CheckExistence is set to true, which is the default value.
      *       For instance, when using quickInverse(), this method is not available.
      */
    bool exists() const { assert(CheckExistence); return m_exists; }

  private:

    int _rows() const { return m_inverse.rows(); }
    int _cols() const { return m_inverse.cols(); }

    const Scalar _coeff(int row, int col) const
    {
      return m_inverse.coeff(row, col);
    }

    template<int LoadMode>
    PacketScalar _packet(int row, int col) const
    {
      return m_inverse.template packet<LoadMode>(row, col);
    }

    enum { _Size = MatrixType::RowsAtCompileTime };
    void _compute(const MatrixType& matrix);
    void _compute_in_general_case(const MatrixType& matrix);
    void _compute_in_size2_case(const MatrixType& matrix);
    void _compute_in_size3_case(const MatrixType& matrix);
    void _compute_in_size4_case(const MatrixType& matrix);

  protected:
    bool m_exists;
    typename MatrixType::Eval m_inverse;
};

template<typename MatrixType, bool CheckExistence>
void Inverse<MatrixType, CheckExistence>
::_compute_in_general_case(const MatrixType& _matrix)
{
  MatrixType matrix(_matrix);
  const RealScalar max = CheckExistence ? matrix.cwiseAbs().maxCoeff()
                                        : static_cast<RealScalar>(0);
  const int size = matrix.rows();
  for(int k = 0; k < size-1; k++)
  {
    int rowOfBiggest;
    const RealScalar max_in_this_col
      = matrix.col(k).end(size-k).cwiseAbs().maxCoeff(&rowOfBiggest);
    if(CheckExistence && ei_isMuchSmallerThan(max_in_this_col, max))
    { m_exists = false; return; }

    m_inverse.row(k).swap(m_inverse.row(k+rowOfBiggest));
    matrix.row(k).swap(matrix.row(k+rowOfBiggest));

    const Scalar d = matrix(k,k);
    m_inverse.block(k+1, 0, size-k-1, size)
      -= matrix.col(k).end(size-k-1) * (m_inverse.row(k) / d);
    matrix.corner(BottomRight, size-k-1, size-k)
      -= matrix.col(k).end(size-k-1) * (matrix.row(k).end(size-k) / d);
  }

  for(int k = 0; k < size-1; k++)
  {
    const Scalar d = static_cast<Scalar>(1)/matrix(k,k);
    matrix.row(k).end(size-k) *= d;
    m_inverse.row(k) *= d;
  }
  if(CheckExistence && ei_isMuchSmallerThan(matrix(size-1,size-1), max))
  { m_exists = false; return; }
  m_inverse.row(size-1) /= matrix(size-1,size-1);

  for(int k = size-1; k >= 1; k--)
  {
    m_inverse.block(0,0,k,size) -= matrix.col(k).start(k) * m_inverse.row(k);
  }
}

template<typename ExpressionType, bool CheckExistence>
bool ei_compute_size2_inverse(const ExpressionType& xpr, typename ExpressionType::Eval* result)
{
  typedef typename ExpressionType::Scalar Scalar;
  const typename ei_nested<ExpressionType, 1+CheckExistence>::type matrix(xpr);
  const Scalar det = matrix.determinant();
  if(CheckExistence && ei_isMuchSmallerThan(det, matrix.cwiseAbs().maxCoeff()))
    return false;
  const Scalar invdet = static_cast<Scalar>(1) / det;
  result->coeffRef(0,0) = matrix.coeff(1,1) * invdet;
  result->coeffRef(1,0) = -matrix.coeff(1,0) * invdet;
  result->coeffRef(0,1) = -matrix.coeff(0,1) * invdet;
  result->coeffRef(1,1) = matrix.coeff(0,0) * invdet;
  return true;
}

template<typename MatrixType, bool CheckExistence>
void Inverse<MatrixType, CheckExistence>::_compute_in_size3_case(const MatrixType& matrix)
{
  const Scalar det_minor00 = matrix.minor(0,0).determinant();
  const Scalar det_minor10 = matrix.minor(1,0).determinant();
  const Scalar det_minor20 = matrix.minor(2,0).determinant();
  const Scalar det = det_minor00 * matrix.coeff(0,0)
                   - det_minor10 * matrix.coeff(1,0)
                   + det_minor20 * matrix.coeff(2,0);
  if(CheckExistence && ei_isMuchSmallerThan(det, matrix.cwiseAbs().maxCoeff()))
    m_exists = false;
  else
  {
    const Scalar invdet = static_cast<Scalar>(1) / det;
    m_inverse.coeffRef(0, 0) = det_minor00 * invdet;
    m_inverse.coeffRef(0, 1) = -det_minor10 * invdet;
    m_inverse.coeffRef(0, 2) = det_minor20 * invdet;
    m_inverse.coeffRef(1, 0) = -matrix.minor(0,1).determinant() * invdet;
    m_inverse.coeffRef(1, 1) = matrix.minor(1,1).determinant() * invdet;
    m_inverse.coeffRef(1, 2) = -matrix.minor(2,1).determinant() * invdet;
    m_inverse.coeffRef(2, 0) = matrix.minor(0,2).determinant() * invdet;
    m_inverse.coeffRef(2, 1) = -matrix.minor(1,2).determinant() * invdet;
    m_inverse.coeffRef(2, 2) = matrix.minor(2,2).determinant() * invdet;
  }
}

template<typename MatrixType, bool CheckExistence>
void Inverse<MatrixType, CheckExistence>::_compute_in_size4_case(const MatrixType& matrix)
{
  /* Let's split M into four 2x2 blocks:
    * (P Q)
    * (R S)
    * If P is invertible, with inverse denoted by P_inverse, and if
    * (S - R*P_inverse*Q) is also invertible, then the inverse of M is
    * (P' Q')
    * (R' S')
    * where
    * S' = (S - R*P_inverse*Q)^(-1)
    * P' = P1 + (P1*Q) * S' *(R*P_inverse)
    * Q' = -(P_inverse*Q) * S'
    * R' = -S' * (R*P_inverse)
    */
  typedef Block<MatrixType,2,2> XprBlock22;
  typedef typename XprBlock22::Eval Block22;
  Block22 P_inverse;

  if(ei_compute_size2_inverse<XprBlock22, true>(matrix.template block<2,2>(0,0), &P_inverse))
  {
    const Block22 Q = matrix.template block<2,2>(0,2);
    const Block22 P_inverse_times_Q = P_inverse * Q;
    const XprBlock22 R = matrix.template block<2,2>(2,0);
    const Block22 R_times_P_inverse = R * P_inverse;
    const Block22 R_times_P_inverse_times_Q = R_times_P_inverse * Q;
    const XprBlock22 S = matrix.template block<2,2>(2,2);
    const Block22 X = S - R_times_P_inverse_times_Q;
    Block22 Y;
    if(ei_compute_size2_inverse<Block22, CheckExistence>(X, &Y))
    {
      m_inverse.template block<2,2>(2,2) = Y;
      m_inverse.template block<2,2>(2,0) = - Y * R_times_P_inverse;
      const Block22 Z = P_inverse_times_Q * Y;
      m_inverse.template block<2,2>(0,2) = - Z;
      m_inverse.template block<2,2>(0,0) = P_inverse + Z * R_times_P_inverse;
    }
    else
    {
      m_exists = false; return;
    }
  }
  else
  {
    _compute_in_general_case(matrix);
  }
}

template<typename MatrixType, bool CheckExistence>
void Inverse<MatrixType, CheckExistence>::_compute(const MatrixType& matrix)
{
  if(_Size == 1)
  {
    const Scalar x = matrix.coeff(0,0);
    if(CheckExistence && x == static_cast<Scalar>(0))
      m_exists = false;
    else
      m_inverse.coeffRef(0,0) = static_cast<Scalar>(1) / x;
  }
  else if(_Size == 2)
  {
    if(CheckExistence)
      m_exists = ei_compute_size2_inverse<MatrixType, true>(matrix, &m_inverse);
    else
      ei_compute_size2_inverse<MatrixType, false>(matrix, &m_inverse);
  }
  else if(_Size == 3) _compute_in_size3_case(matrix);
  else if(_Size == 4) _compute_in_size4_case(matrix);
  else _compute_in_general_case(matrix);
}

/** \lu_module
  *
  * \returns the matrix inverse of \c *this, if it exists.
  *
  * Example: \include MatrixBase_inverse.cpp
  * Output: \verbinclude MatrixBase_inverse.out
  *
  * \sa class Inverse
  */
template<typename Derived>
const Inverse<typename ei_eval<Derived>::type, true>
MatrixBase<Derived>::inverse() const
{
  return Inverse<typename Derived::Eval, true>(eval());
}

/** \lu_module
  *
  * \returns the matrix inverse of \c *this, which is assumed to exist.
  *
  * Example: \include MatrixBase_quickInverse.cpp
  * Output: \verbinclude MatrixBase_quickInverse.out
  *
  * \sa class Inverse
  */
template<typename Derived>
const Inverse<typename ei_eval<Derived>::type, false>
MatrixBase<Derived>::quickInverse() const
{
  return Inverse<typename Derived::Eval, false>(eval());
}

#endif // EIGEN_INVERSE_H
