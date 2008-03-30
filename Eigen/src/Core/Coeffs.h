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
template<typename Derived>
const typename ei_traits<Derived>::Scalar MatrixBase<Derived>
  ::coeff(int row, int col) const
{
  ei_internal_assert(row >= 0 && row < rows()
                     && col >= 0 && col < cols());
  return derived()._coeff(row, col);
}

/** \returns the coefficient at given the given row and column.
  *
  * \sa operator()(int,int), operator[](int) const
  */
template<typename Derived>
const typename ei_traits<Derived>::Scalar MatrixBase<Derived>
  ::operator()(int row, int col) const
{
  ei_assert(row >= 0 && row < rows()
      && col >= 0 && col < cols());
  return derived()._coeff(row, col);
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
template<typename Derived>
typename ei_traits<Derived>::Scalar& MatrixBase<Derived>
  ::coeffRef(int row, int col)
{
  ei_internal_assert(row >= 0 && row < rows()
                     && col >= 0 && col < cols());
  return derived()._coeffRef(row, col);
}

/** \returns a reference to the coefficient at given the given row and column.
  *
  * \sa operator()(int,int) const, operator[](int)
  */
template<typename Derived>
typename ei_traits<Derived>::Scalar& MatrixBase<Derived>
  ::operator()(int row, int col)
{
  ei_assert(row >= 0 && row < rows()
      && col >= 0 && col < cols());
  return derived()._coeffRef(row, col);
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
template<typename Derived>
const typename ei_traits<Derived>::Scalar MatrixBase<Derived>
  ::coeff(int index) const
{
  ei_internal_assert(IsVectorAtCompileTime);
  if(RowsAtCompileTime == 1)
  {
    ei_internal_assert(index >= 0 && index < cols());
    return coeff(0, index);
  }
  else
  {
    ei_internal_assert(index >= 0 && index < rows());
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
template<typename Derived>
const typename ei_traits<Derived>::Scalar MatrixBase<Derived>
  ::operator[](int index) const
{
  ei_assert(IsVectorAtCompileTime);
  if(RowsAtCompileTime == 1)
  {
    ei_assert(index >= 0 && index < cols());
    return coeff(0, index);
  }
  else
  {
    ei_assert(index >= 0 && index < rows());
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
template<typename Derived>
typename ei_traits<Derived>::Scalar& MatrixBase<Derived>
  ::coeffRef(int index)
{
  ei_internal_assert(IsVectorAtCompileTime);
  if(RowsAtCompileTime == 1)
  {
    ei_internal_assert(index >= 0 && index < cols());
    return coeffRef(0, index);
  }
  else
  {
    ei_internal_assert(index >= 0 && index < rows());
    return coeffRef(index, 0);
  }
}

/** \returns a reference to the coefficient at given index.
  *
  * \only_for_vectors
  *
  * \sa operator[](int) const, operator()(int,int), x(), y(), z(), w()
  */
template<typename Derived>
typename ei_traits<Derived>::Scalar& MatrixBase<Derived>
  ::operator[](int index)
{
  ei_assert(IsVectorAtCompileTime);
  if(RowsAtCompileTime == 1)
  {
    ei_assert(index >= 0 && index < cols());
    return coeffRef(0, index);
  }
  else
  {
    ei_assert(index >= 0 && index < rows());
    return coeffRef(index, 0);
  }
}

/** equivalent to operator[](0). \only_for_vectors */
template<typename Derived>
const typename ei_traits<Derived>::Scalar MatrixBase<Derived>
  ::x() const { return (*this)[0]; }

/** equivalent to operator[](1). \only_for_vectors */
template<typename Derived>
const typename ei_traits<Derived>::Scalar MatrixBase<Derived>
  ::y() const { return (*this)[1]; }

/** equivalent to operator[](2). \only_for_vectors */
template<typename Derived>
const typename ei_traits<Derived>::Scalar MatrixBase<Derived>
  ::z() const { return (*this)[2]; }

/** equivalent to operator[](3). \only_for_vectors */
template<typename Derived>
const typename ei_traits<Derived>::Scalar MatrixBase<Derived>
  ::w() const { return (*this)[3]; }

/** equivalent to operator[](0). \only_for_vectors */
template<typename Derived>
typename ei_traits<Derived>::Scalar& MatrixBase<Derived>
  ::x() { return (*this)[0]; }

/** equivalent to operator[](1). \only_for_vectors */
template<typename Derived>
typename ei_traits<Derived>::Scalar& MatrixBase<Derived>
  ::y() { return (*this)[1]; }

/** equivalent to operator[](2). \only_for_vectors */
template<typename Derived>
typename ei_traits<Derived>::Scalar& MatrixBase<Derived>
  ::z() { return (*this)[2]; }

/** equivalent to operator[](3). \only_for_vectors */
template<typename Derived>
typename ei_traits<Derived>::Scalar& MatrixBase<Derived>
  ::w() { return (*this)[3]; }

#endif // EIGEN_COEFFS_H
