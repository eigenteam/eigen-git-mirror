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

#ifndef EI_ROWANDCOL_H
#define EI_ROWANDCOL_H

template<typename MatrixType> class EiRow
  : public EiObject<typename MatrixType::Scalar, EiRow<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class EiObject<Scalar, EiRow<MatrixType> >;
    typedef EiRow Ref;

    static const int RowsAtCompileTime = MatrixType::ColsAtCompileTime,
                     ColsAtCompileTime = 1;

    EiRow(const MatRef& matrix, int row)
      : m_matrix(matrix), m_row(row)
    {
      EI_CHECK_ROW_RANGE(matrix, row);
    }
    
    EiRow(const EiRow& other)
      : m_matrix(other.m_matrix), m_row(other.m_row) {}
    
    template<typename OtherDerived>
    EiRow& operator=(const EiObject<Scalar, OtherDerived>& other)
    {
      return EiObject<Scalar, EiRow<MatrixType> >::operator=(other);
    }
    
    EI_INHERIT_ASSIGNMENT_OPERATORS(EiRow)
    
  private:
    const Ref& _ref() const { return *this; }
    
    int _rows() const { return m_matrix.cols(); }
    int _cols() const { return 1; }
    
    Scalar& _write(int row, int col=0)
    {
      EI_UNUSED(col);
      EI_CHECK_ROW_RANGE(*this, row);
      return m_matrix.write(m_row, row);
    }
    
    Scalar _read(int row, int col=0) const
    {
      EI_UNUSED(col);
      EI_CHECK_ROW_RANGE(*this, row);
      return m_matrix.read(m_row, row);
    }
    
  protected:
    MatRef m_matrix;
    const int m_row;
};

template<typename MatrixType> class EiColumn
  : public EiObject<typename MatrixType::Scalar, EiColumn<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class EiObject<Scalar, EiColumn<MatrixType> >;
    typedef EiColumn Ref;
    
    static const int RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     ColsAtCompileTime = 1;
    
    EiColumn(const MatRef& matrix, int col)
      : m_matrix(matrix), m_col(col)
    {
      EI_CHECK_COL_RANGE(matrix, col);
    }
    
    EiColumn(const EiColumn& other)
      : m_matrix(other.m_matrix), m_col(other.m_col) {}
    
    EI_INHERIT_ASSIGNMENT_OPERATORS(EiColumn)
    
  private:
    const Ref& _ref() const { return *this; }
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return 1; }
    
    Scalar& _write(int row, int col=0)
    {
      EI_UNUSED(col);
      EI_CHECK_ROW_RANGE(*this, row);
      return m_matrix.write(row, m_col);
    }
    
    Scalar _read(int row, int col=0) const
    {
      EI_UNUSED(col);
      EI_CHECK_ROW_RANGE(*this, row);
      return m_matrix.read(row, m_col);
    }
    
  protected:
    MatRef m_matrix;
    const int m_col;
};

template<typename Scalar, typename Derived>
EiRow<Derived>
EiObject<Scalar, Derived>::row(int i)
{
  return EiRow<Derived>(static_cast<Derived*>(this)->ref(), i);
}

template<typename Scalar, typename Derived>
EiColumn<Derived>
EiObject<Scalar, Derived>::col(int i)
{
  return EiColumn<Derived>(static_cast<Derived*>(this)->ref(), i);
}

#endif // EI_ROWANDCOL_H
