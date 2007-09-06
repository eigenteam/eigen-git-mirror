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

#ifndef EIGEN_MATRIXXPR_H
#define EIGEN_MATRIXXPR_H

namespace Eigen {

//forward declarations
template<typename MatrixType> class MatrixRow;
template<typename MatrixType> class MatrixCol;
template<typename MatrixType> class MatrixMinor;
template<typename MatrixType> class MatrixBlock;

template<typename Content> class MatrixConstXpr
{
  public:
    typedef typename Content::Scalar Scalar;
  
    MatrixConstXpr(const Content& content)
      : m_content(content) {}
    
    MatrixConstXpr(const MatrixConstXpr& other)
      : m_content(other.m_content) {}
    
    ~MatrixConstXpr() {}
    
    static bool hasDynamicSize()
    {
      return Content::hasDynamicSize();
    }
    
    int rows() const { return m_content.rows(); }
    int cols() const { return m_content.cols(); }
    
    Scalar operator()(int row, int col) const
    {
      return m_content(row, col);
    }
    
    MatrixConstXpr<MatrixRow<const MatrixConstXpr<Content> > > row(int i) const;
    MatrixConstXpr<MatrixCol<const MatrixConstXpr<Content> > > col(int i) const;
    MatrixConstXpr<MatrixMinor<const MatrixConstXpr<Content> > > minor(int row, int col) const;
    MatrixConstXpr<MatrixBlock<const MatrixConstXpr<Content> > >
      block(int startRow, int endRow, int startCol= 0, int endCol = 0) const;
    
  protected:
    const Content m_content;
};

template<typename Content> class MatrixXpr
{
  public:
    typedef typename Content::Scalar Scalar;
  
    MatrixXpr(const Content& content)
      : m_content(content) {}
    
    MatrixXpr(const MatrixXpr& other)
      : m_content(other.m_content) {}
    
    ~MatrixXpr() {}
    
    static bool hasDynamicSize()
    {
      return Content::hasDynamicSize();
    }
    
    int rows() const { return m_content.rows(); }
    int cols() const { return m_content.cols(); }
    
    Scalar& operator()(int row, int col)
    {
      return m_content(row, col);
    }
    
    template<typename OtherContent>
    MatrixXpr& operator=(const MatrixConstXpr<OtherContent> &other)
    {
      assert(rows() == other.rows() && cols() == other.cols());
      for(int i = 0; i < rows(); i++)
        for(int j = 0; j < cols(); j++)
          this->operator()(i, j) = other(i, j);
      return *this;
    }
    
    template<typename Derived>
    MatrixXpr& operator=(const MatrixBase<Derived>& matrix);
    
    MatrixXpr<MatrixRow<MatrixXpr<Content> > > row(int i);
    MatrixXpr<MatrixCol<MatrixXpr<Content> > > col(int i);
    MatrixXpr<MatrixMinor<MatrixXpr<Content> > > minor(int row, int col);
    MatrixXpr<MatrixBlock<MatrixXpr<Content> > >
      block(int startRow, int endRow, int startCol= 0, int endCol = 0);
  
  private:
    void operator=(const MatrixXpr &other)
    {}
    
  protected:
    Content m_content;
};

} // namespace Eigen

#endif // EIGEN_MATRIXXPR_H
