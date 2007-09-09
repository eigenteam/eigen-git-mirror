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

#ifndef EIGEN_MATRIXALIAS_H
#define EIGEN_MATRIXALIAS_H

namespace Eigen
{

template<typename Derived> class MatrixAlias
{
  public:
    typedef typename Derived::Scalar Scalar;
    typedef MatrixRef<MatrixAlias<Derived> > Ref;
    typedef MatrixXpr<Ref> Xpr;
    
    MatrixAlias(Derived& matrix) : m_aliased(matrix), m_tmp(matrix) {}
    MatrixAlias(const MatrixAlias& other) : m_aliased(other.m_aliased), m_tmp(other.m_tmp) {}
    
    ~MatrixAlias()
    {
      m_aliased.xpr() = m_tmp;
    }
    
    Ref ref()
    {
      return Ref(*this);
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
    
    Scalar& write(int row, int col)
    {
      return m_tmp.write(row, col);
    }
    
    MatrixXpr<MatrixRow<Xpr> > row(int i) { return xpr().row(i); };
    MatrixXpr<MatrixCol<Xpr> > col(int i) { return xpr().col(i); };
    MatrixXpr<MatrixMinor<Xpr> > minor(int row, int col) { return xpr().minor(row, col); };
    MatrixXpr<MatrixBlock<Xpr> >
    block(int startRow, int endRow, int startCol = 0, int endCol = 0)
    {
      return xpr().block(startRow, endRow, startCol, endCol);
    }
    
    template<typename XprContent> 
    void operator=(const MatrixXpr<XprContent> &other)
    {
      xpr() = other;
    }
    
    template<typename XprContent> 
    void operator+=(const MatrixXpr<XprContent> &other)
    {
      xpr() += other;
    }
    
    template<typename XprContent> 
    void operator-=(const MatrixXpr<XprContent> &other)
    {
      xpr() -= other;
    }
    
  protected:
    MatrixRef<MatrixBase<Derived> > m_aliased;
    Derived m_tmp;
};

template<typename Derived>
typename MatrixBase<Derived>::Alias
MatrixBase<Derived>::alias()
{
  return Alias(*static_cast<Derived*>(this));
}

} // namespace Eigen

#endif // EIGEN_MATRIXALIAS_H
