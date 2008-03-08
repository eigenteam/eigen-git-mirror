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

template<typename Scalar, typename Derived>
struct MatrixBase<Scalar, Derived>::CommaInitializer
{
  CommaInitializer(Derived& mat) : m_matrix(mat), m_count(1) {}

  CommaInitializer& operator,(const Scalar& s) {
    assert(m_count<m_matrix.size() && "Too many coefficients passed to Matrix::operator<<");
    m_matrix._coeffRef(m_count/m_matrix.cols(), m_count%m_matrix.cols()) = s;
    m_count++;
    return *this;
  }

  ~CommaInitializer(void)
  {
    assert(m_count==m_matrix.size() && "Too few coefficients passed to Matrix::operator<<");
  }

  Derived& m_matrix;
  int m_count;
};


/** Convenient operator to set the coefficients of a matrix.
  *
  * The coefficients must be provided in a row major order and exactly match
  * the size of the matrix. Otherwise an assertion is raised.
  *
  * Example: \include MatrixBase_set.cpp
  * Output: \verbinclude MatrixBase_set.out
  */
template<typename Scalar, typename Derived>
typename MatrixBase<Scalar, Derived>::CommaInitializer MatrixBase<Scalar, Derived>::operator<< (const Scalar& s)
{
  coeffRef(0,0) = s;
  return CommaInitializer(*static_cast<Derived *>(this));
}


#endif // EIGEN_COMMA_INITIALIZER_H
