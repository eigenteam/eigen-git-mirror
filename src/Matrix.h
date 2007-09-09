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

/** \file Matrix.h
  * \brief Matrix and MatrixX class templates
  */

#ifndef EIGEN_MATRIX_H
#define EIGEN_MATRIX_H

#include "MatrixBase.h"

namespace Eigen
{

template<typename T, int Rows, int Cols>
class Matrix: public MatrixBase< Matrix<T, Rows, Cols> >
{
    friend  class MatrixBase<Matrix<T, Rows, Cols> >;
    typedef class MatrixBase<Matrix<T, Rows, Cols> > Base;
    
  public:
    typedef T Scalar;

  private:

    static bool _hasDynamicNumRows()
    { return false; }

    static bool _hasDynamicNumCols()
    { return false; }

    int _rows() const
    { return Rows; }
    
    int _cols() const
    { return Cols; }

    void _resize( int rows, int cols ) const
    {
      assert(rows == Rows && cols == Cols);
    }

  public:

    Matrix()
    {
      assert(Rows > 0 && Cols > 0);
    }
    
    Matrix(const Matrix& other) : Base()
    {
      *this = other;
    }

    Matrix(int rows, int cols)
    {
      assert(Rows > 0 && Cols > 0 && rows == Rows && cols == Cols);
    }

    void operator=(const Matrix & other)
    { Base::operator=(other); }

    template<typename XprContent>
    void operator=(const MatrixXpr<XprContent> &xpr)
    { Base::operator=(xpr); }

    template<typename XprContent>
    explicit Matrix(const MatrixXpr<XprContent>& xpr)
    {
      *this = xpr;
    }

  protected:

    T m_array[ Rows * Cols ];

};

template<typename T>
class MatrixX : public MatrixBase< MatrixX<T> >
{
    friend  class MatrixBase<MatrixX<T> >;
    typedef class MatrixBase<MatrixX<T> > Base;

  public:

    typedef T Scalar;

    MatrixX(int rows, int cols)
    { _init(rows, cols); }

    MatrixX(const MatrixX& other) : Base()
    {
      _init(other.rows(), other.cols());
      *this = other;
    }

    ~MatrixX()
    { delete[] m_array; }

    void operator=(const MatrixX& other)
    { Base::operator=(other); }

    template<typename XprContent>
    void operator=(const MatrixXpr<XprContent> &xpr)
    { Base::operator=(xpr); }

    template<typename XprContent>
    explicit MatrixX(const MatrixXpr<XprContent>& xpr)
    {
      _init(xpr.rows(), xpr.cols());
      *this = xpr;
    }

  protected:

    int m_rows, m_cols;

    T *m_array;

  private:

    int _rows() const { return m_rows; }
    int _cols() const { return m_cols; }
    
    static bool _hasDynamicNumRows()
    { return true; }

    static bool _hasDynamicNumCols()
    { return true; }

    void _resize( int rows, int cols )
    {
      assert(rows > 0 && cols > 0);
      if(rows * cols > m_rows * m_cols)
      {
        delete[] m_array;
        m_array  = new T[rows * cols];
      }
      m_rows = rows;
      m_cols = cols;
    }
    
    void _init( int rows, int cols )
    {
      assert(rows > 0 && cols > 0);
      m_rows = rows;
      m_cols = cols;
      m_array  = new T[m_rows * m_cols];
    }

};

} // namespace Eigen

#include"MatrixOps.h"
#include"ScalarOps.h"
#include"RowAndCol.h"

#endif // EIGEN_MATRIX_H
