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

#include "Util.h"
#include "MatrixXpr.h"
#include "MatrixRef.h"

namespace Eigen
{

template<typename Derived>
class MatrixBase
{
  public:

    typedef typename ForwardDecl<Derived>::Scalar Scalar;
    typedef MatrixRef<MatrixBase<Derived> > Ref;
    typedef MatrixXpr<Ref> Xpr;
    typedef MatrixAlias<Derived> Alias;

    Ref ref()
    {
      return Ref(*this);
    }
   
    Xpr xpr()
    {
      return Xpr(ref());
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

    const Scalar& read(int row, int col = 0) const
    {
      EIGEN_CHECK_RANGES(*this, row, col);
      return array()[row + col * rows()];
    }

    const Scalar& operator()(int row, int col = 0) const
    {
      return read(row, col);
    }

    Scalar& write(int row, int col = 0)
    {
      EIGEN_CHECK_RANGES(*this, row, col);
      return array()[row + col * rows()];
    }
    
    Scalar& operator()(int row, int col = 0)
    {
      return write(row, col);
    }
    
    template<typename XprContent> 
    MatrixBase& operator=(const MatrixXpr<XprContent> &otherXpr)
    {
      resize(otherXpr.rows(), otherXpr.cols());
      xpr() = otherXpr;
      return *this;
    }
    
    MatrixBase& operator=(const MatrixBase &other)
    {
      resize(other.rows(), other.cols());
      for(int i = 0; i < rows(); i++)
        for(int j = 0; j < cols(); j++)
          this->operator()(i, j) = other(i, j);
      return *this;
    }
    
    MatrixXpr<MatrixRow<Ref> > row(int i);
    MatrixXpr<MatrixCol<Ref> > col(int i);
    MatrixXpr<MatrixMinor<Ref> > minor(int row, int col);
    MatrixXpr<MatrixBlock<Ref> >
      block(int startRow, int endRow, int startCol = 0, int endCol = 0);
    
    template<typename Content>
    MatrixBase& operator+=(const MatrixXpr<Content> &xpr);
    template<typename Content>
    MatrixBase& operator-=(const MatrixXpr<Content> &xpr);
    template<typename Derived2>
    MatrixBase& operator+=(MatrixBase<Derived2> &other);
    template<typename Derived2>
    MatrixBase& operator-=(MatrixBase<Derived2> &other);
    
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
      write(i, j) = matrix(i, j);
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
                            const MatrixXpr<Content>& xpr)
{
  for( int i = 0; i < xpr.rows(); i++ )
  {
    s << xpr.read(i, 0);
    for (int j = 1; j < xpr.cols(); j++ )
      s << " " << xpr.read(i, j);
    if( i < xpr.rows() - 1)
      s << std::endl;
  }
  return s;
}

} // namespace Eigen

#include "MatrixAlias.h"
#include "MatrixOps.h"
#include "ScalarOps.h"
#include "RowAndCol.h"

#endif // EIGEN_MATRIXBASE_H
