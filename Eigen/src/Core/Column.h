// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_COLUMN_H
#define EIGEN_COLUMN_H

/** \class Column
  *
  * \brief Expression of a column
  *
  * \param MatrixType the type of the object in which we are taking a column
  *
  * This class represents an expression of a column. It is the return
  * type of MatrixBase::col() and most of the time this is the only way it
  * is used.
  *
  * However, if you want to directly maniputate column expressions,
  * for instance if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating this:
  * \include class_Column.cpp
  * Output: \verbinclude class_Column.out
  *
  * \sa MatrixBase::col()
  */
template<typename MatrixType> class Column
  : public MatrixBase<typename MatrixType::Scalar, Column<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, Column>;
    friend class MatrixBase<Scalar, Column>::Traits;
    typedef MatrixBase<Scalar, Column> Base;

    Column(const MatRef& matrix, int col)
      : m_matrix(matrix), m_col(col)
    {
      assert(col >= 0 && col < matrix.cols());
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Column)

  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = 1,
      MaxRowsAtCompileTime = MatrixType::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = 1
    };

    const Column& _ref() const { return *this; }
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return 1; }

    Scalar& _coeffRef(int row, int)
    {
      return m_matrix.coeffRef(row, m_col);
    }

    Scalar _coeff(int row, int) const
    {
      return m_matrix.coeff(row, m_col);
    }

  protected:
    MatRef m_matrix;
    const int m_col;
};

/** \returns an expression of the \a i-th column of *this. Note that the numbering starts at 0.
  *
  * Example: \include MatrixBase_col.cpp
  * Output: \verbinclude MatrixBase_col.out
  *
  * \sa row(), class Column */
template<typename Scalar, typename Derived>
Column<Derived>
MatrixBase<Scalar, Derived>::col(int i)
{
  return Column<Derived>(ref(), i);
}

/** This is the const version of col(). */
template<typename Scalar, typename Derived>
const Column<Derived>
MatrixBase<Scalar, Derived>::col(int i) const
{
  return Column<Derived>(ref(), i);
}

#endif // EIGEN_COLUMN_H
