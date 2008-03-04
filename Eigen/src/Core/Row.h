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

#ifndef EIGEN_ROW_H
#define EIGEN_ROW_H

/** \class Row
  *
  * \brief Expression of a row
  *
  * \param MatrixType the type of the object in which we are taking a row
  *
  * This class represents an expression of a row. It is the return
  * type of MatrixBase::row() and most of the time this is the only way it
  * is used.
  *
  * However, if you want to directly maniputate row expressions,
  * for instance if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating this:
  * \include class_Row.cpp
  * Output: \verbinclude class_Row.out
  *
  * \sa MatrixBase::row()
  */
template<typename MatrixType> class Row
  : public MatrixBase<typename MatrixType::Scalar, Row<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, Row>;
    friend class MatrixBase<Scalar, Row>::Traits;
    typedef MatrixBase<Scalar, Row> Base;

    Row(const MatRef& matrix, int row)
      : m_matrix(matrix), m_row(row)
    {
      assert(row >= 0 && row < matrix.rows());
    }

    template<typename OtherDerived>
    Row& operator=(const MatrixBase<Scalar, OtherDerived>& other)
    {
      return MatrixBase<Scalar, Row<MatrixType> >::operator=(other);
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Row)

  private:
    enum {
      RowsAtCompileTime = 1,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = 1,
      MaxColsAtCompileTime = MatrixType::Traits::MaxColsAtCompileTime
    };

    const Row& _ref() const { return *this; }

    int _rows() const { return 1; }
    int _cols() const { return m_matrix.cols(); }

    Scalar& _coeffRef(int, int col)
    {
      return m_matrix.coeffRef(m_row, col);
    }

    Scalar _coeff(int, int col) const
    {
      return m_matrix.coeff(m_row, col);
    }

  protected:
    MatRef m_matrix;
    const int m_row;
};

/** \returns an expression of the \a i-th row of *this. Note that the numbering starts at 0.
  *
  * Example: \include MatrixBase_row.cpp
  * Output: \verbinclude MatrixBase_row.out
  *
  * \sa col(), class Row */
template<typename Scalar, typename Derived>
Row<Derived>
MatrixBase<Scalar, Derived>::row(int i)
{
  return Row<Derived>(ref(), i);
}

/** This is the const version of row(). */
template<typename Scalar, typename Derived>
const Row<Derived>
MatrixBase<Scalar, Derived>::row(int i) const
{
  return Row<Derived>(ref(), i);
}

#endif // EIGEN_ROW_H
