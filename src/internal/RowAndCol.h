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

#ifndef EIGEN_ROWANDCOL_H
#define EIGEN_ROWANDCOL_H

template<typename MatrixType> class MatrixRow
  : public EigenBase<typename MatrixType::Scalar, MatrixRow<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class EigenBase<Scalar, MatrixRow<MatrixType> >;
    typedef MatrixRow Ref;

    static const int RowsAtCompileTime = MatrixType::ColsAtCompileTime,
                     ColsAtCompileTime = 1;

    MatrixRow(const MatRef& matrix, int row)
      : m_matrix(matrix), m_row(row)
    {
      EIGEN_CHECK_ROW_RANGE(matrix, row);
    }
    
    MatrixRow(const MatrixRow& other)
      : m_matrix(other.m_matrix), m_row(other.m_row) {}
    
    template<typename OtherDerived>
    MatrixRow& operator=(const EigenBase<Scalar, OtherDerived>& other)
    {
      return EigenBase<Scalar, MatrixRow<MatrixType> >::operator=(other);
    }
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(MatrixRow)
    
  private:
    const Ref& _ref() const { return *this; }
    
    int _rows() const { return m_matrix.cols(); }
    int _cols() const { return 1; }
    
    Scalar& _write(int row, int col=0)
    {
      EIGEN_UNUSED(col);
      EIGEN_CHECK_ROW_RANGE(*this, row);
      return m_matrix.write(m_row, row);
    }
    
    Scalar _read(int row, int col=0) const
    {
      EIGEN_UNUSED(col);
      EIGEN_CHECK_ROW_RANGE(*this, row);
      return m_matrix.read(m_row, row);
    }
    
  protected:
    MatRef m_matrix;
    const int m_row;
};

template<typename MatrixType> class MatrixCol
  : public EigenBase<typename MatrixType::Scalar, MatrixCol<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class EigenBase<Scalar, MatrixCol<MatrixType> >;
    typedef MatrixCol Ref;
    
    static const int RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     ColsAtCompileTime = 1;
    
    MatrixCol(const MatRef& matrix, int col)
      : m_matrix(matrix), m_col(col)
    {
      EIGEN_CHECK_COL_RANGE(matrix, col);
    }
    
    MatrixCol(const MatrixCol& other)
      : m_matrix(other.m_matrix), m_col(other.m_col) {}
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(MatrixCol)
    
  private:
    const Ref& _ref() const { return *this; }
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return 1; }
    
    Scalar& _write(int row, int col=0)
    {
      EIGEN_UNUSED(col);
      EIGEN_CHECK_ROW_RANGE(*this, row);
      return m_matrix.write(row, m_col);
    }
    
    Scalar _read(int row, int col=0) const
    {
      EIGEN_UNUSED(col);
      EIGEN_CHECK_ROW_RANGE(*this, row);
      return m_matrix.read(row, m_col);
    }
    
  protected:
    MatRef m_matrix;
    const int m_col;
};

template<typename Scalar, typename Derived>
MatrixRow<EigenBase<Scalar, Derived> >
EigenBase<Scalar, Derived>::row(int i)
{
  return MatrixRow<EigenBase>(ref(), i);
}

template<typename Scalar, typename Derived>
MatrixCol<EigenBase<Scalar, Derived> >
EigenBase<Scalar, Derived>::col(int i)
{
  return MatrixCol<EigenBase>(ref(), i);
}

#endif // EIGEN_ROWANDCOL_H
