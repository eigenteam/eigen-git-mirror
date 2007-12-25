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

#ifndef EIGEN_MAP_H
#define EIGEN_MAP_H

template<typename MatrixType> class Map
  : public MatrixBase<typename MatrixType::Scalar, Map<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, Map<MatrixType> >;
    
    Map(const Scalar* data, int rows, int cols) : m_data(data), m_rows(rows), m_cols(cols)
    {
      assert(rows > 0 && cols > 0);
    }
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Map)
    
  private:
    static const int _RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     _ColsAtCompileTime = MatrixType::ColsAtCompileTime;

    const Map& _ref() const { return *this; }
    int _rows() const { return m_rows; }
    int _cols() const { return m_cols; }
    
    const Scalar& _coeff(int row, int col) const
    {
      return m_data[row + col * m_rows];
    }
    
    Scalar& _coeffRef(int row, int col)
    {
      return const_cast<Scalar*>(m_data)[row + col * m_rows];
    }
    
  protected:
    const Scalar* m_data;
    int m_rows, m_cols;
};

template<typename Scalar, typename Derived>
const Map<Derived> MatrixBase<Scalar, Derived>::map(const Scalar* data, int rows, int cols)
{
  return Map<Derived>(data, rows, cols);
}

template<typename Scalar, typename Derived>
const Map<Derived> MatrixBase<Scalar, Derived>::map(const Scalar* data, int size)
{
  assert(IsVectorAtCompileTime);
  if(ColsAtCompileTime == 1)
    return Map<Derived>(data, size, 1);
  else
    return Map<Derived>(data, 1, size);
}

template<typename Scalar, typename Derived>
const Map<Derived> MatrixBase<Scalar, Derived>::map(const Scalar* data)
{
  return Map<Derived>(data, RowsAtCompileTime, ColsAtCompileTime);
}

template<typename Scalar, typename Derived>
Map<Derived> MatrixBase<Scalar, Derived>::map(Scalar* data, int rows, int cols)
{
  return Map<Derived>(data, rows, cols);
}

template<typename Scalar, typename Derived>
Map<Derived> MatrixBase<Scalar, Derived>::map(Scalar* data, int size)
{
  assert(IsVectorAtCompileTime);
  if(ColsAtCompileTime == 1)
    return Map<Derived>(data, size, 1);
  else
    return Map<Derived>(data, 1, size);
}

template<typename Scalar, typename Derived>
Map<Derived> MatrixBase<Scalar, Derived>::map(Scalar* data)
{
  return Map<Derived>(data, RowsAtCompileTime, ColsAtCompileTime);
}

template<typename _Scalar, int _Rows, int _Cols>
Matrix<_Scalar, _Rows, _Cols>
  ::Matrix(const Scalar *data, int rows, int cols)
  : Storage(rows, cols)
{
  *this = map(data, rows, cols);
}

template<typename _Scalar, int _Rows, int _Cols>
Matrix<_Scalar, _Rows, _Cols>
  ::Matrix(const Scalar *data, int size)
  : Storage(size)
{
  *this = map(data, size);
}

template<typename _Scalar, int _Rows, int _Cols>
Matrix<_Scalar, _Rows, _Cols>
  ::Matrix(const Scalar *data)
  : Storage()
{
  *this = map(data);
}

#endif // EIGEN_MAP_H
