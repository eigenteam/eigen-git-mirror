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

#ifndef EIGEN_EIGENBASE_H
#define EIGEN_EIGENBASE_H

#include "Util.h"

namespace Eigen {

template<typename _Scalar, typename Derived> class EigenBase
{
    static const int RowsAtCompileTime = Derived::RowsAtCompileTime,
                     ColsAtCompileTime = Derived::ColsAtCompileTime;
  public:
    typedef typename ForwardDecl<Derived>::Ref Ref;
    typedef _Scalar Scalar;
  
    int rows() const { return static_cast<const Derived *>(this)->_rows(); }
    int cols() const { return static_cast<const Derived *>(this)->_cols(); }
    int size() const { return rows() * cols(); }
    
    Ref ref()
    { return static_cast<Derived *>(this)->_ref(); }
    
    Ref ref() const
    { return static_cast<const Derived *>(this)->_ref(); }
    
    Scalar& write(int row, int col)
    {
      return static_cast<Derived *>(this)->_write(row, col);
    }
    
    Scalar read(int row, int col) const
    {
      return static_cast<const Derived *>(this)->_read(row, col);
    }
    
    template<typename OtherDerived>
    Derived& operator=(const EigenBase<Scalar, OtherDerived>& other)
    {
      assert(rows() == other.rows() && cols() == other.cols());
      for(int i = 0; i < rows(); i++)
        for(int j = 0; j < cols(); j++)
          write(i, j) = other.read(i, j);
      return *static_cast<Derived*>(this);
    }
    
    //special case of the above template operator=. Strangely, g++ 4.1 failed to use
    //that template when OtherDerived == Derived
    Derived& operator=(const EigenBase& other)
    {
      assert(rows() == other.rows() && cols() == other.cols());
      for(int i = 0; i < rows(); i++)
        for(int j = 0; j < cols(); j++)
          write(i, j) = other.read(i, j);
      return *static_cast<Derived*>(this);
    }
    
    MatrixRow<EigenBase> row(int i);
    MatrixCol<EigenBase> col(int i);
    MatrixMinor<EigenBase> minor(int row, int col);
    MatrixBlock<EigenBase>
      block(int startRow, int endRow, int startCol= 0, int endCol = 0);
    
    template<typename OtherDerived>
    Derived& operator+=(const EigenBase<Scalar, OtherDerived>& other);
    template<typename OtherDerived>
    Derived& operator-=(const EigenBase<Scalar, OtherDerived>& other);

    Scalar operator()(int row, int col = 0) const
    { return read(row, col); }
    
    Scalar& operator()(int row, int col = 0)
    { return write(row, col); }
};

template<typename Scalar, typename Derived>
std::ostream & operator <<
( std::ostream & s,
  const EigenBase<Scalar, Derived> & m )
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

} // namespace Eigen

#endif // EIGEN_EIGENBASE_H
