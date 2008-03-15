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

#ifndef EIGEN_MATRIXSTORAGE_H
#define EIGEN_MATRIXSTORAGE_H

/** \internal
  *
  * \class ei_matrix_storage
  *
  * \brief Stores the data of a matrix
  *
  * This class stores the data of fixed-size, dynamic-size or mixed matrices
  * in a way as compact as possible.
  *
  * \sa Matrix
  */
template<typename T, int Size, int _Rows, int _Cols> class ei_matrix_storage;

// purely fixed-size matrix.
template<typename T, int Size, int _Rows, int _Cols> class ei_matrix_storage
{
    T m_data[Size];
  public:
    ei_matrix_storage() {}
    ei_matrix_storage(int,int,int) {}
    static int rows(void) {return _Rows;}
    static int cols(void) {return _Cols;}
    void resize(int,int,int) {}
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

// dynamic-size matrix with fixed-size storage
template<typename T, int Size> class ei_matrix_storage<T, Size, Dynamic, Dynamic>
{
    T m_data[Size];
    int m_rows;
    int m_cols;
  public:
    ei_matrix_storage(int, int rows, int cols) : m_rows(rows), m_cols(cols) {}
    ~ei_matrix_storage() {}
    int rows(void) const {return m_rows;}
    int cols(void) const {return m_cols;}
    void resize(int, int rows, int cols)
    {
      m_rows = rows;
      m_cols = cols;
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

// dynamic-size matrix with fixed-size storage and fixed width
template<typename T, int Size, int _Cols> class ei_matrix_storage<T, Size, Dynamic, _Cols>
{
    T m_data[Size];
    int m_rows;
  public:
    ei_matrix_storage(int, int rows, int) : m_rows(rows) {}
    ~ei_matrix_storage() {}
    int rows(void) const {return m_rows;}
    int cols(void) const {return _Cols;}
    void resize(int size, int rows, int)
    {
      m_rows = rows;
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

// dynamic-size matrix with fixed-size storage and fixed height
template<typename T, int Size, int _Rows> class ei_matrix_storage<T, Size, _Rows, Dynamic>
{
    T m_data[Size];
    int m_cols;
  public:
    ei_matrix_storage(int, int, int cols) : m_cols(cols) {}
    ~ei_matrix_storage() {}
    int rows(void) const {return _Rows;}
    int cols(void) const {return m_cols;}
    void resize(int size, int, int cols)
    {
      m_cols = cols;
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

// purely dynamic matrix.
template<typename T> class ei_matrix_storage<T, Dynamic, Dynamic, Dynamic>
{
    T *m_data;
    int m_rows;
    int m_cols;
  public:
    ei_matrix_storage(int size, int rows, int cols)
      : m_data(new T[size]), m_rows(rows), m_cols(cols) {}
    ~ei_matrix_storage() { delete[] m_data; }
    int rows(void) const {return m_rows;}
    int cols(void) const {return m_cols;}
    void resize(int size, int rows, int cols)
    {
      if(size != m_rows*m_cols)
      {
        delete[] m_data;
        m_data = new T[size];
      }
      m_rows = rows;
      m_cols = cols;
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

// matrix with dynamic width and fixed height (so that matrix has dynamic size).
template<typename T, int _Rows> class ei_matrix_storage<T, Dynamic, _Rows, Dynamic>
{
    T *m_data;
    int m_cols;
  public:
    ei_matrix_storage(int size, int, int cols) : m_data(new T[size]), m_cols(cols) {}
    ~ei_matrix_storage() { delete[] m_data; }
    static int rows(void) {return _Rows;}
    int cols(void) const {return m_cols;}
    void resize(int size, int, int cols)
    {
      if(size != _Rows*m_cols)
      {
        delete[] m_data;
        m_data = new T[size];
      }
      m_cols = cols;
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

// matrix with dynamic height and fixed width (so that matrix has dynamic size).
template<typename T, int _Cols> class ei_matrix_storage<T, Dynamic, Dynamic, _Cols>
{
    T *m_data;
    int m_rows;
  public:
    ei_matrix_storage(int size, int rows, int) : m_data(new T[size]), m_rows(rows) {}
    ~ei_matrix_storage() { delete[] m_data; }
    int rows(void) const {return m_rows;}
    static int cols(void) {return _Cols;}
    void resize(int size, int rows, int)
    {
      if(size != m_rows*_Cols)
      {
        delete[] m_data;
        m_data = new T[size];
      }
      m_rows = rows;
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

#endif // EIGEN_MATRIX_H
