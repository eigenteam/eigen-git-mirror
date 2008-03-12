// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_COMMA_INITIALIZER_H
#define EIGEN_COMMA_INITIALIZER_H

/** \internal
  * Helper class to define the MatrixBase::operator<<
  */
template<typename Derived>
struct MatrixBase<Derived>::CommaInitializer
{
  CommaInitializer(Derived& mat, const Scalar& s)
    : m_matrix(mat), m_row(0), m_col(1), m_currentBlockRows(1)
  {
    m_matrix.coeffRef(0,0) = s;
  }

  template<typename OtherDerived>
  CommaInitializer(Derived& mat, const MatrixBase<OtherDerived>& other)
    : m_matrix(mat), m_row(0), m_col(other.cols()), m_currentBlockRows(other.rows())
  {
    m_matrix.block(0, 0, other.rows(), other.cols()) = other;
  }

  CommaInitializer& operator,(const Scalar& s)
  {
    if (m_col==m_matrix.cols())
    {
      m_row+=m_currentBlockRows;
      m_col = 0;
      m_currentBlockRows = 1;
    }
    assert(m_col<m_matrix.cols() && "Too many coefficients passed to Matrix::operator<<");
    assert(m_currentBlockRows==1);
    m_matrix.coeffRef(m_row, m_col++) = s;
    return *this;
  }

  template<typename OtherDerived>
  CommaInitializer& operator,(const MatrixBase<OtherDerived>& other)
  {
    if (m_col==m_matrix.cols())
    {
      m_row+=m_currentBlockRows;
      m_col = 0;
      m_currentBlockRows = other.rows();
    }
    assert(m_col<m_matrix.cols() && "Too many coefficients passed to Matrix::operator<<");
    assert(m_currentBlockRows==other.rows());
    m_matrix.block(m_row, m_col, other.rows(), other.cols()) = other;
    m_col += other.cols();
    return *this;
  }

  ~CommaInitializer(void)
  {
    assert((m_row+m_currentBlockRows) == m_matrix.rows()
         && m_col == m_matrix.cols()
         && "Too few coefficients passed to Matrix::operator<<");
  }

  Derived& m_matrix;
  int m_row; // current row id
  int m_col; // current col id
  int m_currentBlockRows; // current block height
};

/** Convenient operator to set the coefficients of a matrix.
  *
  * The coefficients must be provided in a row major order and exactly match
  * the size of the matrix. Otherwise an assertion is raised.
  *
  * Example: \include MatrixBase_set.cpp
  * Output: \verbinclude MatrixBase_set.out
  */
template<typename Derived>
typename MatrixBase<Derived>::CommaInitializer MatrixBase<Derived>::operator<< (const Scalar& s)
{
  return CommaInitializer(*static_cast<Derived*>(this), s);
}

template<typename Derived>
template<typename OtherDerived>
typename MatrixBase<Derived>::CommaInitializer
MatrixBase<Derived>::operator<<(const MatrixBase<OtherDerived>& other)
{
  return CommaInitializer(*static_cast<Derived *>(this), other);
}

#endif // EIGEN_COMMA_INITIALIZER_H
