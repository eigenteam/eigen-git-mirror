// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EIGEN_MINOR_H
#define EIGEN_MINOR_H

namespace Eigen {

template<typename MatrixType> class MatrixMinor
  : public EigenBase<typename MatrixType::Scalar, MatrixMinor<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class EigenBase<Scalar, MatrixMinor<MatrixType> >;
    typedef MatrixMinor Ref;
    
    static const int RowsAtCompileTime = MatrixType::RowsAtCompileTime - 1,
                     ColsAtCompileTime = MatrixType::ColsAtCompileTime - 1;

    MatrixMinor(const MatRef& matrix,
                int row, int col = 0)
      : m_matrix(matrix), m_row(row), m_col(col)
    {
      EIGEN_CHECK_RANGES(matrix, row, col);
    }
    
    MatrixMinor(const MatrixMinor& other)
      : m_matrix(other.m_matrix), m_row(other.m_row), m_col(other.m_col) {}
    
    INHERIT_ASSIGNMENT_OPERATORS(MatrixMinor)
    
  private:
    const Ref& _ref() const { return *this; }
    int _rows() const { return m_matrix.rows() - 1; }
    int _cols() const { return m_matrix.cols() - 1; }
    
    Scalar& _write(int row, int col=0)
    {
      return m_matrix.write(row + (row >= m_row), col + (col >= m_col));
    }
    
    Scalar _read(int row, int col=0) const
    {
      return m_matrix.read(row + (row >= m_row), col + (col >= m_col));
    }
    
  protected:
    MatRef m_matrix;
    const int m_row, m_col;
};

template<typename Scalar, typename Derived>
MatrixMinor<EigenBase<Scalar, Derived> >
EigenBase<Scalar, Derived>::minor(int row, int col)
{
  return MatrixMinor<EigenBase>(ref(), row, col);
}

} // namespace Eigen

#endif // EIGEN_MINOR_H
