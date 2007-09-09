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

#ifndef EIGEN_VECTOR_H
#define EIGEN_VECTOR_H

#include "MatrixBase.h"

namespace Eigen
{

template<typename T, int Size>
class Vector: public MatrixBase<Vector<T, Size> >
{
    friend  class MatrixBase<Vector<T, Size> >;
    typedef class MatrixBase<Vector<T, Size> > Base;
    
  public:
    typedef T Scalar;

  private:

    static bool _hasDynamicNumRows()
    { return false; }

    static bool _hasDynamicNumCols()
    { return false; }

    int _rows() const
    { return Size; }
    
    int _cols() const
    { return 1; }

    void _resize( int rows, int cols ) const
    {
      assert( rows == Size && cols == 1 );
    }

  public:

    Vector()
    {
      assert(Size > 0);
    }

    explicit Vector(int rows, int cols = 1)
    {
      assert(Size > 0 && rows == Size && cols == 1);
    }
    
    Vector(const Vector& other) : Base()
    {
      *this = other;
    }

    void operator=(const Vector & other)
    { Base::operator=(other); }
    
    template<typename XprContent>
    void operator=(const MatrixXpr<XprContent> &xpr)
    {
      Base::operator=(xpr);
    }

    template<typename XprContent>
    explicit Vector(const MatrixXpr<XprContent>& xpr)
    {
      *this = xpr;
    }
    
    int size() const { return _rows(); }

  protected:

    T m_array[Size];

};

template<typename T>
class VectorX : public MatrixBase<VectorX<T> >
{
    friend  class MatrixBase<VectorX<T> >;
    typedef class MatrixBase<VectorX<T> > Base;

  public:

    typedef T Scalar;

    explicit VectorX(int rows, int cols = 1)
    {
      assert(cols == 1);
      _init(rows);
    }
    
    VectorX(const VectorX& other) : Base()
    {
      _init(other.size());
      *this = other;
    }
    
    void operator=(const VectorX& other)
    {
      Base::operator=(other);
    }
    
    template<typename XprContent>
    void operator=(const MatrixXpr<XprContent> &xpr)
    {
      Base::operator=(xpr);
    }

    template<typename XprContent>
    explicit VectorX(const MatrixXpr<XprContent>& xpr)
    {
      _init(xpr.rows());
      *this = xpr;
    }

    ~VectorX()
    {
      delete[] m_array; }

    int size() const { return _rows(); }

  protected:
    
    int m_size;
    T *m_array;

  private:

    int _rows() const { return m_size; }
    int _cols() const { return 1; }
    
    static bool _hasDynamicNumRows()
    { return true; }

    static bool _hasDynamicNumCols()
    { return false; }

    void _resize(int rows, int cols)
    {
      assert(rows > 0 && cols == 1);
      if(rows > m_size)
      {
        delete[] m_array;
        m_array  = new T[rows];
      }
      m_size = rows;
    }

    void _init(int size)
    {
      assert(size > 0);
      m_size = size;
      m_array = new T[m_size];
    }

};

} // namespace Eigen

#endif // EIGEN_VECTOR_H
