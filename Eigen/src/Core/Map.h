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

/** \class Map
  *
  * \brief A matrix or vector expression mapping an existing array of data.
  *
  * This class represents a matrix or vector expression mapping an existing array of data.
  * It can be used to let Eigen interface without any overhead with non-Eigen data structures,
  * such as plain C arrays or structures from other libraries.
  *
  * This class is the return type of Matrix::map() and most of the time this is the only
  * way it is used.
  *
  * \sa Matrix::map()
  */
template<typename MatrixType> class Map
  : public MatrixBase<typename MatrixType::Scalar, Map<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, Map<MatrixType> >;

    static const TraversalOrder Order = MatrixType::Order;
    static const int RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     ColsAtCompileTime = MatrixType::ColsAtCompileTime;

  private:
    const Map& _ref() const { return *this; }
    int _rows() const { return m_rows; }
    int _cols() const { return m_cols; }
    
    const Scalar& _coeff(int row, int col) const
    {
      if(Order == ColumnMajor)
        return m_data[row + col * m_rows];
      else // RowMajor
        return m_data[col + row * m_cols];
    }
    
    Scalar& _coeffRef(int row, int col)
    {
      if(Order == ColumnMajor)
        return const_cast<Scalar*>(m_data)[row + col * m_rows];
      else // RowMajor
        return const_cast<Scalar*>(m_data)[col + row * m_cols];
    }
  
  public:
    Map(const Scalar* data, int rows, int cols) : m_data(data), m_rows(rows), m_cols(cols)
    {
      assert(rows > 0
          && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
          && cols > 0
          && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
    }
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Map)
    
  protected:
    const Scalar* m_data;
    int m_rows, m_cols;
};

/** This is the const version of map(Scalar*,int,int). */
template<typename _Scalar, int _Rows, int _Cols, TraversalOrder _StorageOrder>
const Map<Matrix<_Scalar, _Rows, _Cols, _StorageOrder> >
Matrix<_Scalar, _Rows, _Cols, _StorageOrder>::map(const Scalar* data, int rows, int cols)
{
  return Map<Matrix>(data, rows, cols);
}

/** This is the const version of map(Scalar*,int). */
template<typename _Scalar, int _Rows, int _Cols, TraversalOrder _StorageOrder>
const Map<Matrix<_Scalar, _Rows, _Cols, _StorageOrder> >
Matrix<_Scalar, _Rows, _Cols, _StorageOrder>::map(const Scalar* data, int size)
{
  assert(_Cols == 1 || _Rows ==1);
  if(_Cols == 1)
    return Map<Matrix>(data, size, 1);
  else
    return Map<Matrix>(data, 1, size);
}

/** This is the const version of map(Scalar*). */
template<typename _Scalar, int _Rows, int _Cols, TraversalOrder _StorageOrder>
const Map<Matrix<_Scalar, _Rows, _Cols, _StorageOrder> >
Matrix<_Scalar, _Rows, _Cols, _StorageOrder>::map(const Scalar* data)
{
  return Map<Matrix>(data, _Rows, _Cols);
}

/** \returns a expression of a matrix or vector mapping the given data.
  *
  * \param data The array of data to map
  * \param rows The number of rows of the expression to construct
  * \param cols The number of columns of the expression to construct
  *
  * Example: \include MatrixBase_map_int_int.cpp
  * Output: \verbinclude MatrixBase_map_int_int.out
  *
  * \sa map(const Scalar*, int, int), map(Scalar*, int), map(Scalar*), class Map
  */
template<typename _Scalar, int _Rows, int _Cols, TraversalOrder _StorageOrder>
Map<Matrix<_Scalar, _Rows, _Cols, _StorageOrder> >
Matrix<_Scalar, _Rows, _Cols, _StorageOrder>::map(Scalar* data, int rows, int cols)
{
  return Map<Matrix>(data, rows, cols);
}

/** \returns a expression of a vector mapping the given data.
  *
  * \param data The array of data to map
  * \param size The size (number of coefficients) of the expression to construct
  *
  * \only_for_vectors
  *
  * Example: \include MatrixBase_map_int.cpp
  * Output: \verbinclude MatrixBase_map_int.out
  *
  * \sa map(const Scalar*, int), map(Scalar*, int, int), map(Scalar*), class Map
  */
template<typename _Scalar, int _Rows, int _Cols, TraversalOrder _StorageOrder>
Map<Matrix<_Scalar, _Rows, _Cols, _StorageOrder> >
Matrix<_Scalar, _Rows, _Cols, _StorageOrder>::map(Scalar* data, int size)
{
  assert(_Cols == 1 || _Rows ==1);
  if(_Cols == 1)
    return Map<Matrix>(data, size, 1);
  else
    return Map<Matrix>(data, 1, size);
}

/** \returns a expression of a fixed-size matrix or vector mapping the given data.
  *
  * \param data The array of data to map
  *
  * Example: \include MatrixBase_map.cpp
  * Output: \verbinclude MatrixBase_map.out
  *
  * \sa map(const Scalar*), map(Scalar*, int), map(Scalar*, int, int), class Map
  */
template<typename _Scalar, int _Rows, int _Cols, TraversalOrder _StorageOrder>
Map<Matrix<_Scalar, _Rows, _Cols, _StorageOrder> >
Matrix<_Scalar, _Rows, _Cols, _StorageOrder>::map(Scalar* data)
{
  return Map<Matrix>(data, _Rows, _Cols);
}

/** Constructor copying an existing array of data. Only useful for dynamic-size matrices:
  * for fixed-size matrices, it is redundant to pass the \a rows and \a cols parameters.
  * \param data The array of data to copy
  * \param rows The number of rows of the matrix to construct
  * \param cols The number of columns of the matrix to construct
  *
  * \sa Matrix(const Scalar *), Matrix::map(const Scalar *, int, int)
  */
template<typename _Scalar, int _Rows, int _Cols, TraversalOrder _StorageOrder>
Matrix<_Scalar, _Rows, _Cols, _StorageOrder>
  ::Matrix(const Scalar *data, int rows, int cols)
  : Storage(rows, cols)
{
  *this = map(data, rows, cols);
}

/** Constructor copying an existing array of data. Only useful for dynamic-size vectors:
  * for fixed-size vectors, it is redundant to pass the \a size parameter.
  *
  * \only_for_vectors
  *
  * \param data The array of data to copy
  * \param size The size of the vector to construct
  *
  * \sa Matrix(const Scalar *), Matrix::map(const Scalar *, int)
  */
template<typename _Scalar, int _Rows, int _Cols, TraversalOrder _StorageOrder>
Matrix<_Scalar, _Rows, _Cols, _StorageOrder>
  ::Matrix(const Scalar *data, int size)
  : Storage(size)
{
  *this = map(data, size);
}

/** Constructor copying an existing array of data.
  * Only for fixed-size matrices and vectors.
  * \param data The array of data to copy
  *
  * For dynamic-size matrices and vectors, see the variants taking additional int parameters
  * for the dimensions.
  *
  * \sa Matrix(const Scalar *, int), Matrix(const Scalar *, int, int),
  * Matrix::map(const Scalar *)
  */
template<typename _Scalar, int _Rows, int _Cols, TraversalOrder _StorageOrder>
Matrix<_Scalar, _Rows, _Cols, _StorageOrder>
  ::Matrix(const Scalar *data)
  : Storage()
{
  *this = map(data);
}

#endif // EIGEN_MAP_H
