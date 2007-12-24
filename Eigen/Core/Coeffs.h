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

#ifndef EIGEN_COEFFS_H
#define EIGEN_COEFFS_H

/** Short version: don't use this function, use
  * \link operator()(int,int) const \endlink instead.
  *
  * Long version: this function is similar to
  * \link operator()(int,int) const \endlink, but without the assertion.
  * Use this for limiting the performance cost of debugging code when doing
  * repeated coefficient access. Only use this when it is guaranteed that the
  * parameters \a row and \a col are in range.
  *
  * If EIGEN_INTERNAL_DEBUGGING is defined, an assertion will be made, making this
  * function equivalent to \link operator()(int,int) const \endlink.
  *
  * \sa operator()(int,int) const, coeffRef(int,int), coeff(int) const
  */
template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::coeff(int row, int col) const
{
  eigen_internal_assert(row >= 0 && row < rows()
                     && col >= 0 && col < cols());
  return static_cast<const Derived *>(this)->_coeff(row, col);
}

/** \returns the coefficient at given the given row and column.
  *
  * \sa operator()(int,int), operator[](int) const
  */
template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::operator()(int row, int col) const
{
  assert(row >= 0 && row < rows()
      && col >= 0 && col < cols());
  return static_cast<const Derived *>(this)->_coeff(row, col);
}

/** Short version: don't use this function, use
  * \link operator()(int,int) \endlink instead.
  *
  * Long version: this function is similar to
  * \link operator()(int,int) \endlink, but without the assertion.
  * Use this for limiting the performance cost of debugging code when doing
  * repeated coefficient access. Only use this when it is guaranteed that the
  * parameters \a row and \a col are in range.
  *
  * If EIGEN_INTERNAL_DEBUGGING is defined, an assertion will be made, making this
  * function equivalent to \link operator()(int,int) \endlink.
  *
  * \sa operator()(int,int), coeff(int, int) const, coeffRef(int)
  */
template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::coeffRef(int row, int col)
{
  eigen_internal_assert(row >= 0 && row < rows()
                     && col >= 0 && col < cols());
  return static_cast<Derived *>(this)->_coeffRef(row, col);
}

/** \returns a reference to the coefficient at given the given row and column.
  *
  * \sa operator()(int,int) const, operator[](int)
  */
template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::operator()(int row, int col)
{
  assert(row >= 0 && row < rows()
      && col >= 0 && col < cols());
  return static_cast<Derived *>(this)->_coeffRef(row, col);
}

/** Short version: don't use this function, use
  * \link operator[](int) const \endlink instead.
  *
  * Long version: this function is similar to
  * \link operator[](int) const \endlink, but without the assertion.
  * Use this for limiting the performance cost of debugging code when doing
  * repeated coefficient access. Only use this when it is guaranteed that the
  * parameters \a row and \a col are in range.
  *
  * If EIGEN_INTERNAL_DEBUGGING is defined, an assertion will be made, making this
  * function equivalent to \link operator[](int) const \endlink.
  *
  * \sa operator[](int) const, coeffRef(int), coeff(int,int) const
  */
template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::coeff(int index) const
{
  eigen_internal_assert(IsVector);
  if(RowsAtCompileTime == 1)
  {
    eigen_internal_assert(index >= 0 && index < cols());
    return coeff(0, index);
  }
  else
  {
    eigen_internal_assert(index >= 0 && index < rows());
    return coeff(index, 0);
  }
}

/** \returns the coefficient at given index.
  *
  * \only_for_vectors
  *
  * \sa operator[](int), operator()(int,int) const, x() const, y() const,
  * z() const, w() const
  */
template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::operator[](int index) const
{
  assert(IsVector);
  if(RowsAtCompileTime == 1)
  {
    assert(index >= 0 && index < cols());
    return coeff(0, index);
  }
  else
  {
    assert(index >= 0 && index < rows());
    return coeff(index, 0);
  }
}

/** Short version: don't use this function, use
  * \link operator[](int) \endlink instead.
  *
  * Long version: this function is similar to
  * \link operator[](int) \endlink, but without the assertion.
  * Use this for limiting the performance cost of debugging code when doing
  * repeated coefficient access. Only use this when it is guaranteed that the
  * parameters \a row and \a col are in range.
  *
  * If EIGEN_INTERNAL_DEBUGGING is defined, an assertion will be made, making this
  * function equivalent to \link operator[](int) \endlink.
  *
  * \sa operator[](int), coeff(int) const, coeffRef(int,int)
  */
template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::coeffRef(int index)
{
  eigen_internal_assert(IsVector);
  if(RowsAtCompileTime == 1)
  {
    eigen_internal_assert(index >= 0 && index < cols());
    return coeffRef(0, index);
  }
  else
  {
    eigen_internal_assert(index >= 0 && index < rows());
    return coeffRef(index, 0);
  }
}

/** \returns a reference to the coefficient at given index.
  *
  * \only_for_vectors
  *
  * \sa operator[](int) const, operator()(int,int), x(), y(), z(), w()
  */
template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::operator[](int index)
{
  assert(IsVector);
  if(RowsAtCompileTime == 1)
  {
    assert(index >= 0 && index < cols());
    return coeffRef(0, index);
  }
  else
  {
    assert(index >= 0 && index < rows());
    return coeffRef(index, 0);
  }
}

/** equivalent to operator[](0). \only_for_vectors */
template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::x() const { return (*this)[0]; }

/** equivalent to operator[](1). \only_for_vectors */
template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::y() const { return (*this)[1]; }

/** equivalent to operator[](2). \only_for_vectors */
template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::z() const { return (*this)[2]; }

/** equivalent to operator[](3). \only_for_vectors */
template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::w() const { return (*this)[3]; }

/** equivalent to operator[](0). \only_for_vectors */
template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::x() { return (*this)[0]; }

/** equivalent to operator[](1). \only_for_vectors */
template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::y() { return (*this)[1]; }

/** equivalent to operator[](2). \only_for_vectors */
template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::z() { return (*this)[2]; }

/** equivalent to operator[](3). \only_for_vectors */
template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::w() { return (*this)[3]; }

#endif // EIGEN_COEFFS_H
