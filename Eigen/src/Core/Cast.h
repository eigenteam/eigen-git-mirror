// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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
    friend class MatrixBase<Scalar, Cast<Scalar, MatrixType> >;
    
    Cast(const MatRef& matrix) : m_matrix(matrix) {}
    
  private:
    static const int RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
                     ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime;
    const Cast& _ref() const { return *this; }
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }
    
    Scalar _coeff(int row, int col) const
    {
      return static_cast<Scalar>(m_matrix.coeff(row, col));
    }
    
  protected:
    MatRef m_matrix;
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
