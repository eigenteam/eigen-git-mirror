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

#ifndef EIGEN_CAST_H
#define EIGEN_CAST_H

/** \class Cast
  *
  * \brief Expression with casted scalar type
  *
  * \param NewScalar the new scalar type
  * \param MatrixType the type of the object in which we are casting the scalar type
  *
  * This class represents an expression where we are casting the scalar type to a new
  * type. It is the return type of MatrixBase::cast() and most of the time this is the
  * only way it is used.
  *
  * However, if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating this:
  * \include class_Cast.cpp
  * Output: \verbinclude class_Cast.out
  *
  * \sa MatrixBase::cast()
  */
template<typename NewScalar, typename MatrixType> class Cast : NoOperatorEquals,
  public MatrixBase<NewScalar, Cast<NewScalar, MatrixType> >
{
  public:
    typedef NewScalar Scalar;
    typedef typename MatrixType::Ref MatRef;
    friend class MatrixBase<Scalar, Cast>;
    friend class MatrixBase<Scalar, Cast>::Traits;
    typedef MatrixBase<Scalar, Cast> Base;

    Cast(const MatRef& matrix) : m_matrix(matrix) {}

  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::Traits::MaxColsAtCompileTime
    };
    const Cast& _ref() const { return *this; }
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }

    NewScalar _coeff(int row, int col) const
    {
      return static_cast<NewScalar>(m_matrix.coeff(row, col));
    }

  protected:
    const MatRef m_matrix;
};

/** \returns an expression of *this with the \a Scalar type casted to
  * \a NewScalar.
  *
  * The template parameter \a NewScalar is the type we are casting the scalars to.
  *
  * Example: \include MatrixBase_cast.cpp
  * Output: \verbinclude MatrixBase_cast.out
  *
  * \sa class Cast
  */
template<typename Scalar, typename Derived>
template<typename NewScalar>
const Cast<NewScalar, Derived>
MatrixBase<Scalar, Derived>::cast() const
{
  return Cast<NewScalar, Derived>(ref());
}

#endif // EIGEN_CAST_H
