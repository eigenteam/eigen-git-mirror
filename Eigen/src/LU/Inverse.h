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

/** \class Inverse
  *
  * \brief Inverse of a matrix
  *
  * \param ExpressionType the type of the matrix/expression of which we are taking the inverse
  * \param CheckExistence whether or not to check the existence of the inverse while computing it
  *
  * This class represents the inverse of a matrix. It is the return
  * type of MatrixBase::inverse() and most of the time this is the only way it
  * is used.
  *
  * \sa MatrixBase::inverse(), MatrixBase::quickInverse()
  */
template<typename ExpressionType, bool CheckExistence>
struct ei_traits<Inverse<ExpressionType, CheckExistence> >
{
  typedef typename ExpressionType::Scalar Scalar;
  typedef typename ExpressionType::Eval MatrixType;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    Flags = MatrixType::Flags,
    CoeffReadCost = MatrixType::CoeffReadCost
  };
};

template<typename ExpressionType, bool CheckExistence> class Inverse : ei_no_assignment_operator,
  public MatrixBase<Inverse<ExpressionType, CheckExistence> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Inverse)
    typedef typename ei_traits<Inverse>::MatrixType MatrixType;

    Inverse(const ExpressionType& xpr)
      : m_exists(true),
        m_inverse(MatrixType::identity(xpr.rows(), xpr.cols()))
    {
      ei_assert(xpr.rows() == xpr.cols());
      _compute(xpr);
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

    enum { _Size = MatrixType::RowsAtCompileTime };
    void _compute(const ExpressionType& xpr);
    void _compute_in_general_case(const ExpressionType& xpr);
    void _compute_in_size1_case(const ExpressionType& xpr);
    void _compute_in_size2_case(const ExpressionType& xpr);
    void _compute_in_size3_case(const ExpressionType& xpr);
    void _compute_in_size4_case(const ExpressionType& xpr);

  protected:
    bool m_exists;
    MatrixType m_inverse;
};

template<typename ExpressionType, bool CheckExistence>
void Inverse<ExpressionType, CheckExistence>
::_compute_in_general_case(const ExpressionType& xpr)
{
  MatrixType matrix(xpr);
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

template<typename ExpressionType, typename MatrixType, bool CheckExistence>
bool ei_compute_size2_inverse(const ExpressionType& xpr, MatrixType* result)
{
  typedef typename MatrixType::Scalar Scalar;
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

template<typename ExpressionType, bool CheckExistence>
void Inverse<ExpressionType, CheckExistence>::_compute_in_size3_case(const ExpressionType& xpr)
{
  const typename ei_nested<ExpressionType, 2+CheckExistence>::type matrix(xpr);
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

template<typename ExpressionType, bool CheckExistence>
void Inverse<ExpressionType, CheckExistence>::_compute_in_size4_case(const ExpressionType& xpr)
{
  typedef Block<ExpressionType,2,2> XprBlock22;
  typedef typename XprBlock22::Eval Block22;

  Block22 P_inverse;

  if(ei_compute_size2_inverse<XprBlock22, Block22, true>(xpr.template block<2,2>(0,0), &P_inverse))
  {
    const Block22 Q = xpr.template block<2,2>(0,2);
    const Block22 P_inverse_times_Q = P_inverse * Q;
    const Block22 R = xpr.template block<2,2>(2,0);
    const Block22 R_times_P_inverse = R * P_inverse;
    const Block22 R_times_P_inverse_times_Q = R_times_P_inverse * Q;
    const Block22 S = xpr.template block<2,2>(2,2);
    const Block22 X = S - R_times_P_inverse_times_Q;
    Block22 Y;
    if(ei_compute_size2_inverse<Block22, Block22, true>(X, &Y)) 
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
    _compute_in_general_case(xpr);
  }
}

template<typename ExpressionType, bool CheckExistence>
void Inverse<ExpressionType, CheckExistence>::_compute(const ExpressionType& xpr)
{
  if(_Size == 1)
  {
    const Scalar x = xpr.coeff(0,0);
    if(CheckExistence && x == static_cast<Scalar>(0))
      m_exists = false;
    else
      m_inverse.coeffRef(0,0) = static_cast<Scalar>(1) / x;
  }
  else if(_Size == 2)
  {
    if(CheckExistence)
      m_exists = ei_compute_size2_inverse<ExpressionType, MatrixType, true>(xpr, &m_inverse);
    else
      ei_compute_size2_inverse<ExpressionType, MatrixType, false>(xpr, &m_inverse);
  }
  else if(_Size == 3) _compute_in_size3_case(xpr);
  else if(_Size == 4) _compute_in_size4_case(xpr);
  else _compute_in_general_case(xpr);
}

/** \return the matrix inverse of \c *this, if it exists.
  *
  * Example: \include MatrixBase_inverse.cpp
  * Output: \verbinclude MatrixBase_inverse.out
  *
  * \sa class Inverse
  */
template<typename Derived>
const Inverse<Derived, true>
MatrixBase<Derived>::inverse() const
{
  return Inverse<Derived, true>(derived());
}

/** \return the matrix inverse of \c *this, which is assumed to exist.
  *
  * Example: \include MatrixBase_quickInverse.cpp
  * Output: \verbinclude MatrixBase_quickInverse.out
  *
  * \sa class Inverse
  */
template<typename Derived>
const Inverse<Derived, false>
MatrixBase<Derived>::quickInverse() const
{
  return Inverse<Derived, false>(derived());
}

#endif // EIGEN_INVERSE_H
