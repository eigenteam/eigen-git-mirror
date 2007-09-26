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

#ifndef EIGEN_MATRIXALIAS_H
#define EIGEN_MATRIXALIAS_H

namespace Eigen
{

template<typename MatrixType> class MatrixAlias
  : public EigenBase<typename MatrixType::Scalar, MatrixAlias<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef MatrixRef<MatrixAlias> Ref;
    typedef EigenBase<typename MatrixType::Scalar, MatrixAlias> Base;
    friend class EigenBase<typename MatrixType::Scalar, MatrixAlias>;
    
    MatrixAlias(MatrixType& matrix) : m_aliased(matrix), m_tmp(matrix) {}
    MatrixAlias(const MatrixAlias& other) : m_aliased(other.m_aliased), m_tmp(other.m_tmp) {}
    
    ~MatrixAlias()
    {
      m_aliased = m_tmp;
    }
    
    INHERIT_ASSIGNMENT_OPERATORS(MatrixAlias)

  private:
    Ref _ref() const
    {
      return Ref(*const_cast<MatrixAlias*>(this));
    }
    
    int _rows() const { return m_tmp.rows(); }
    int _cols() const { return m_tmp.cols(); }
    
    Scalar& _write(int row, int col)
    {
      return m_tmp.write(row, col);
    }
    
    Scalar _read(int row, int col) const
    {
      return m_aliased.read(row, col);
    }
    
  protected:
    MatrixRef<MatrixType> m_aliased;
    MatrixType m_tmp;
};

template<typename _Scalar, int _Rows, int _Cols>
typename Matrix<_Scalar, _Rows, _Cols>::Alias
Matrix<_Scalar, _Rows, _Cols>::alias()
{
  return Alias(*this);
}

} // namespace Eigen

#endif // EIGEN_MATRIXALIAS_H
