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

#ifndef EIGEN_MATRIXBASE_H
#define EIGEN_MATRIXBASE_H

#include"Util.h"
#include"MatrixXpr.h"

namespace Eigen
{

template<typename MatrixType> class MatrixConstRef
{
  public:
    typedef typename ForwardDecl<MatrixType>::Scalar Scalar;
    
    MatrixConstRef(const MatrixType& matrix) : m_matrix(matrix) {}
    MatrixConstRef(const MatrixConstRef& other) : m_matrix(other.m_matrix) {}
    ~MatrixConstRef() {}

    static bool hasDynamicNumRows()
    {
      return MatrixType::hasDynamicNumRows();
    }

    static bool hasDynamicNumCols()
    {
      return MatrixType::hasDynamicNumCols();
    }
    
    int rows() const { return m_matrix.rows(); }
    int cols() const { return m_matrix.cols(); }

    const Scalar& operator()(int row, int col) const
    {
      return m_matrix(row, col);
    }

  protected:
    const MatrixType& m_matrix;
};

template<typename MatrixType> class MatrixRef
{
  public:
    typedef typename ForwardDecl<MatrixType>::Scalar Scalar;
    typedef MatrixXpr<MatrixRef<MatrixType> > Xpr;
    
    MatrixRef(MatrixType& matrix) : m_matrix(matrix) {}
    MatrixRef(const MatrixRef& other) : m_matrix(other.m_matrix) {}
    ~MatrixRef() {}

    static bool hasDynamicNumRows()
    {
      return MatrixType::hasDynamicNumRows();
    }

    static bool hasDynamicNumCols()
    {
      return MatrixType::hasDynamicNumCols();
    }
    
    int rows() const { return m_matrix.rows(); }
    int cols() const { return m_matrix.cols(); }

    Scalar& operator()(int row, int col)
    {
      return m_matrix(row, col);
    }
    
    MatrixType& matrix() { return m_matrix; }

    Xpr xpr()
    {
      return Xpr(*this);
    }

  protected:
    MatrixType& m_matrix;
};

template<typename Derived>
class MatrixBase
{
  public:

    typedef typename ForwardDecl<Derived>::Scalar Scalar;
    typedef MatrixConstRef<MatrixBase<Derived> > ConstRef;
    typedef MatrixRef<MatrixBase<Derived> > Ref;
    typedef MatrixConstXpr<ConstRef> ConstXpr;
    typedef MatrixXpr<Ref> Xpr;
    typedef MatrixAlias<Derived> Alias;

    Ref ref()
    {
      return Ref(*this);
    }
    
    ConstRef constRef() const
    {
      return ConstRef(*this);
    }
    
    Xpr xpr()
    {
      return Xpr(ref());
    }
    
    ConstXpr constXpr() const
    {
      return ConstXpr(constRef());
    }
    
    Alias alias();
    
    static bool hasDynamicNumRows()
    {
      return Derived::_hasDynamicNumRows();
    }

    static bool hasDynamicNumCols()
    {
      return Derived::_hasDynamicNumCols();
    }
    
    int rows() const
    {
      return static_cast<const Derived*>(this)->_rows();
    }
    
    int cols() const
    {
      return static_cast<const Derived*>(this)->_cols();
    }
    
    void resize(int rows, int cols)
    {
      static_cast<Derived*>(this)->_resize(rows, cols);
    }

    const Scalar* array() const
    {
      return static_cast<const Derived*>(this)->m_array;
    }

    Scalar* array()
    {
      return static_cast<Derived*>(this)->m_array;
    }

    const Scalar& operator()(int row, int col = 0) const
    {
      EIGEN_CHECK_RANGES(*this, row, col);
      return array()[row + col * rows()];
    }

    Scalar& operator()(int row, int col = 0)
    {
      EIGEN_CHECK_RANGES(*this, row, col);
      return array()[row + col * rows()];
    }
    
    template<typename XprContent> 
    MatrixBase& operator=(const MatrixConstXpr<XprContent> &otherXpr)
    {
      resize(otherXpr.rows(), otherXpr.cols());
      xpr() = otherXpr;
      return *this;
    }
    
    MatrixBase& operator=(const MatrixBase &other)
    {
      return *this = other.constXpr();
    }
    
    MatrixConstXpr<MatrixRow<const ConstRef> > row(int i) const;
    MatrixConstXpr<MatrixCol<const ConstRef> > col(int i) const;
    MatrixConstXpr<MatrixMinor<const ConstRef> > minor(int row, int col) const;
    MatrixConstXpr<MatrixBlock<const ConstRef> >
      block(int startRow, int endRow, int startCol = 0, int endCol = 0) const;
    
    template<typename Content>
    MatrixBase& operator+=(const MatrixConstXpr<Content> &xpr);
    template<typename Content>
    MatrixBase& operator-=(const MatrixConstXpr<Content> &xpr);
    template<typename Derived2>
    MatrixBase& operator+=(const MatrixBase<Derived2> &other);
    template<typename Derived2>
    MatrixBase& operator-=(const MatrixBase<Derived2> &other);

    
  protected:
  
    MatrixBase() {};
};

template<typename Content>
template<typename Derived>
MatrixXpr<Content>& MatrixXpr<Content>::operator=(const MatrixBase<Derived>& matrix)
{
  assert(rows() == matrix.rows() && cols() == matrix.cols());
  for(int i = 0; i < rows(); i++)
    for(int j = 0; j < cols(); j++)
      this->operator()(i, j) = matrix(i, j);
  return *this;
}

template<typename Derived>
std::ostream & operator <<
( std::ostream & s,
  const MatrixBase<Derived> & m )
{
  for( int i = 0; i < m.rows(); i++ )
  {
    s << m( i, 0 );
    for (int j = 1; j < m.cols(); j++ )
      s << " " << m( i, j );
    if( i < m.rows() - 1)
      s << std::endl;
  }
  return s;
}

template<typename Content>
std::ostream & operator << (std::ostream & s,
                            const MatrixConstXpr<Content>& m)
{
  for( int i = 0; i < m.rows(); i++ )
  {
    s << m( i, 0 );
    for (int j = 1; j < m.cols(); j++ )
      s << " " << m( i, j );
    if( i < m.rows() - 1)
      s << std::endl;
  }
  return s;
}

template<typename Derived> class MatrixAlias
{
  public:
    typedef typename Derived::Scalar Scalar;
    typedef MatrixRef<MatrixAlias<Derived> > Ref;
    typedef MatrixXpr<Ref> Xpr;
    
    MatrixAlias(Derived& matrix) : m_ref(matrix), m_tmp(matrix) {}
    MatrixAlias(const MatrixAlias& other) : m_ref(other.m_ref), m_tmp(other.m_tmp) {}
    
    ~MatrixAlias()
    {
      m_ref.xpr() = m_tmp;
    }
    
    Xpr xpr()
    {
      return Xpr(ref());
    }
    
    static bool hasDynamicNumRows()
    {
      return MatrixBase<Derived>::hasDynamicNumRows();
    }

    static bool hasDynamicNumCols()
    {
      return MatrixBase<Derived>::hasDynamicNumCols();
    }
    
    int rows() const { return m_tmp.rows(); }
    int cols() const { return m_tmp.cols(); }
    
    Scalar& operator()(int row, int col)
    {
      return m_tmp(row, col);
    }
    
    Ref ref()
    {
      return Ref(*this);
    }
    
    template<typename XprContent> 
    void operator=(const MatrixConstXpr<XprContent> &xpr)
    {
      ref().xpr() = xpr;
    }
    
  protected:
    MatrixRef<MatrixBase<Derived> > m_ref;
    Derived m_tmp;
};

template<typename Derived>
typename MatrixBase<Derived>::Alias
MatrixBase<Derived>::alias()
{
  return Alias(*static_cast<Derived*>(this));
}

} // namespace Eigen

#endif // EIGEN_MATRIXBASE_H
