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

#ifndef EIGEN_MATRIXSTORAGE_H
#define EIGEN_MATRIXSTORAGE_H

template<typename Scalar,
         int      RowsAtCompileTime,
         int      ColsAtCompileTime>
class MatrixStorage
{
  protected:
    Scalar m_array[RowsAtCompileTime * RowsAtCompileTime];
  
    void resize(int rows, int cols)
    { assert(rows == RowsAtCompileTime && cols == ColsAtCompileTime); }
    
    int _rows() const
    { return RowsAtCompileTime; }
    
    int _cols() const
    { return ColsAtCompileTime; }

  public:
    MatrixStorage(int rows, int cols)
    {
      EIGEN_UNUSED(rows);
      EIGEN_UNUSED(cols);
      assert(RowsAtCompileTime > 0 && ColsAtCompileTime > 0);
    }
    
    ~MatrixStorage() {};
};

template<typename Scalar>
class MatrixStorage<Scalar, DynamicSize, 1>
{
  protected:
    int m_rows;
    Scalar *m_array;
    
    void resize(int rows, int cols)
    { assert(rows > 0 && cols == 1);
      if(rows > m_rows) {
        delete[] m_array;
        m_array  = new Scalar[rows];
      }
      m_rows = rows;
    }
    
    int _rows() const
    { return m_rows; }
    
    int _cols() const
    { return 1; }
    
  public:
    MatrixStorage(int rows, int cols) : m_rows(rows)
    {
      assert(m_rows > 0 && cols == 1);
      m_array = new Scalar[m_rows];
    }
    
    ~MatrixStorage()
    { delete[] m_array; }
};

template<typename Scalar>
class MatrixStorage<Scalar, DynamicSize, DynamicSize>
{
  protected:
    int m_rows, m_cols;
    Scalar *m_array;
    
    void resize(int rows, int cols)
    {
      assert(rows > 0 && cols > 0);
      if(rows * cols > m_rows * m_cols)
      {
        delete[] m_array;
        m_array  = new Scalar[rows * cols];
      }
      m_rows = rows;
      m_cols = cols;
    }
    
    int _rows() const
    { return m_rows; }
    
    int _cols() const
    { return m_cols; }
    
  public:
    MatrixStorage(int rows, int cols) : m_rows(rows), m_cols(cols)
    {
      assert(m_rows > 0 && m_cols > 0);
      m_array = new Scalar[m_rows * m_cols];
    }
    
    ~MatrixStorage()
    { delete[] m_array; }
};

#endif // EIGEN_MATRIXSTORAGE_H
