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

#ifndef EI_FROMARRAY_H
#define EI_FROMARRAY_H

template<typename MatrixType> class FromArray
  : public Object<typename MatrixType::Scalar, FromArray<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class Object<Scalar, FromArray<MatrixType> >;
    
    static const int RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     ColsAtCompileTime = MatrixType::ColsAtCompileTime;

    FromArray(int rows, int cols, const Scalar* array) : m_rows(rows), m_cols(cols), m_array(array)
    {
      assert(rows > 0 && cols > 0);
    }
    
  private:
    FromArray& _ref() { return *this; }
    const FromArray& _constRef() const { return *this; }
    int _rows() const { return m_rows; }
    int _cols() const { return m_cols; }
    
    const Scalar& _read(int row, int col) const
    {
      return m_array[row + col * m_rows];
    }
    
  protected:
    int m_rows, m_cols;
    const Scalar* m_array;
};

template<typename Scalar, typename Derived>
FromArray<Derived> Object<Scalar, Derived>::fromArray(const Scalar* array, int rows, int cols)
{
  return FromArray<Derived>(rows, cols, array);
}

#endif // EI_FROMARRAY_H
