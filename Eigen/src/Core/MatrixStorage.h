// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@gmail.com>
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

#ifndef EIGEN_MATRIXSTORAGE_H
#define EIGEN_MATRIXSTORAGE_H


/** \class MatrixStorage
  *
  * \brief Stores the data of a matrix
  *
  * \internal
  *
  * This class stores the data of fixed-size, dynamic-size or mixed matrices
  * in a way as compact as possible.
  *
  * \sa Matrix
  */
template<typename T, int Size, int _Rows, int _Cols> class MatrixStorage;

// purely fixed-size matrix.
template<typename T, int Size, int _Rows, int _Cols> class MatrixStorage
{
    T m_data[Size];
  public:
    MatrixStorage() {}
    MatrixStorage(int,int,int) {}
    static int rows(void) {return _Rows;}
    static int cols(void) {return _Cols;}
    void resize(int,int,int) {}
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

// dynamic-size matrix with fixed-size storage
template<typename T, int Size> class MatrixStorage<T, Size, Dynamic, Dynamic>
{
    T m_data[Size];
    int m_rows;
    int m_cols;
  public:
    MatrixStorage(int, int nbRows, int nbCols) : m_rows(nbRows), m_cols(nbCols) {}
    ~MatrixStorage() {}
    int rows(void) const {return m_rows;}
    int cols(void) const {return m_cols;}
    void resize(int, int nbRows, int nbCols)
    {
      m_rows = nbRows;
      m_cols = nbCols;
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

// dynamic-size matrix with fixed-size storage and fixed width
template<typename T, int Size, int _Cols> class MatrixStorage<T, Size, Dynamic, _Cols>
{
    T m_data[Size];
    int m_rows;
  public:
    MatrixStorage(int, int nbRows, int) : m_rows(nbRows) {}
    ~MatrixStorage() {}
    int rows(void) const {return m_rows;}
    int cols(void) const {return _Cols;}
    void resize(int size, int nbRows, int)
    {
      m_rows = nbRows;
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

// dynamic-size matrix with fixed-size storage and fixed height
template<typename T, int Size, int _Rows> class MatrixStorage<T, Size, _Rows, Dynamic>
{
    T m_data[Size];
    int m_cols;
  public:
    MatrixStorage(int, int nbRows, int nbCols) : m_cols(nbCols) {}
    ~MatrixStorage() {}
    int rows(void) const {return _Rows;}
    int cols(void) const {return m_cols;}
    void resize(int size, int, int nbCols)
    {
      m_cols = nbCols;
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

// purely dynamic matrix.
template<typename T> class MatrixStorage<T, Dynamic, Dynamic, Dynamic>
{
    T *m_data;
    int m_rows;
    int m_cols;
  public:
    MatrixStorage(int size, int nbRows, int nbCols) : m_data(new T[size]), m_rows(nbRows), m_cols(nbCols) {}
    ~MatrixStorage() { delete[] m_data; }
    int rows(void) const {return m_rows;}
    int cols(void) const {return m_cols;}
    void resize(int size, int nbRows, int nbCols)
    {
      if(size != m_rows*m_cols)
      {
        delete[] m_data;
        m_data = new T[size];
      }
      m_rows = nbRows;
      m_cols = nbCols;
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

// matrix with dynamic width and fixed height (so that matrix has dynamic size).
template<typename T, int _Rows> class MatrixStorage<T, Dynamic, _Rows, Dynamic>
{
    T *m_data;
    int m_cols;
  public:
    MatrixStorage(int size, int, int nbCols) : m_data(new T[size]), m_cols(nbCols) {}
    ~MatrixStorage() { delete[] m_data; }
    static int rows(void) {return _Rows;}
    int cols(void) const {return m_cols;}
    void resize(int size, int, int nbCols)
    {
      if(size != _Rows*m_cols)
      {
        delete[] m_data;
        m_data = new T[size];
      }
      m_cols = nbCols;
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

// matrix with dynamic height and fixed width (so that matrix has dynamic size).
template<typename T, int _Cols> class MatrixStorage<T, Dynamic, Dynamic, _Cols>
{
    T *m_data;
    int m_rows;
  public:
    MatrixStorage(int size, int nbRows, int) : m_data(new T[size]), m_rows(nbRows) {}
    ~MatrixStorage() { delete[] m_data; }
    int rows(void) const {return m_rows;}
    static int cols(void) {return _Cols;}
    void resize(int size, int nbRows, int)
    {
      if(size != m_rows*_Cols)
      {
        delete[] m_data;
        m_data = new T[size];
      }
      m_rows = nbRows;
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

#endif // EIGEN_MATRIX_H
