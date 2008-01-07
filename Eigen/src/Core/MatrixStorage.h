// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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
    Scalar m_data[RowsAtCompileTime * ColsAtCompileTime];
  
    void resize(int rows, int cols)
    {
      EIGEN_ONLY_USED_FOR_DEBUG(rows);
      EIGEN_ONLY_USED_FOR_DEBUG(cols);
      assert(rows == RowsAtCompileTime && cols == ColsAtCompileTime);
    }
    
    int _rows() const
    { return RowsAtCompileTime; }
    
    int _cols() const
    { return ColsAtCompileTime; }

  public:
    MatrixStorage() {}
    MatrixStorage(int) {}
    MatrixStorage(int, int) {}
    ~MatrixStorage() {};
};

template<typename Scalar, int ColsAtCompileTime>
class MatrixStorage<Scalar, Dynamic, ColsAtCompileTime>
{
  protected:
    int m_rows;
    Scalar* m_data;
    
    void resize(int rows, int cols)
    {
      EIGEN_ONLY_USED_FOR_DEBUG(cols);
      assert(rows > 0 && cols == ColsAtCompileTime);
      if(rows > m_rows)
      {
        delete[] m_data;
        m_data  = new Scalar[rows * ColsAtCompileTime];
      }
      m_rows = rows;
    }
    
    int _rows() const
    { return m_rows; }
    
    int _cols() const
    { return ColsAtCompileTime; }
    
  public:
    MatrixStorage(int dim = 1) : m_rows(dim)
    {
      m_data = new Scalar[m_rows * ColsAtCompileTime];
    }
  
    MatrixStorage(int rows, int) : m_rows(rows)
    {
      m_data = new Scalar[m_rows * ColsAtCompileTime];
    }
    
    ~MatrixStorage()
    { delete[] m_data; }
};

template<typename Scalar, int RowsAtCompileTime>
class MatrixStorage<Scalar, RowsAtCompileTime, Dynamic>
{
  protected:
    int m_cols;
    Scalar* m_data;
    
    void resize(int rows, int cols)
    {
      EIGEN_ONLY_USED_FOR_DEBUG(rows);
      assert(rows == RowsAtCompileTime && cols > 0);
      if(cols > m_cols)
      {
        delete[] m_data;
        m_data  = new Scalar[cols * RowsAtCompileTime];
      }
      m_cols = cols;
    }
    
    int _rows() const
    { return RowsAtCompileTime; }
    
    int _cols() const
    { return m_cols; }
    
  public:
    MatrixStorage(int dim = 1) : m_cols(dim)
    {
      m_data = new Scalar[m_cols * RowsAtCompileTime];
    }
    
    MatrixStorage(int, int cols) : m_cols(cols)
    {
      m_data = new Scalar[m_cols * RowsAtCompileTime];
    }
    
    ~MatrixStorage()
    { delete[] m_data; }
};

template<typename Scalar>
class MatrixStorage<Scalar, Dynamic, Dynamic>
{
  protected:
    int m_rows, m_cols;
    Scalar* m_data;
    
    void resize(int rows, int cols)
    {
      assert(rows > 0 && cols > 0);
      if(rows * cols > m_rows * m_cols)
      {
        delete[] m_data;
        m_data  = new Scalar[rows * cols];
      }
      m_rows = rows;
      m_cols = cols;
    }
    
    int _rows() const
    { return m_rows; }
    
    int _cols() const
    { return m_cols; }
    
  public:
    
    MatrixStorage(int rows = 1, int cols = 1) : m_rows(rows), m_cols(cols)
    {
      m_data = new Scalar[m_rows * m_cols];
    }
    
    ~MatrixStorage()
    { delete[] m_data; }
};

#endif // EIGEN_MATRIXSTORAGE_H
