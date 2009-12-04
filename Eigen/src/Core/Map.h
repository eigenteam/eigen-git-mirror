// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_MAP_H
#define EIGEN_MAP_H

/** \class Map
  *
  * \brief A matrix or vector expression mapping an existing array of data.
  *
  * \param MatrixType the equivalent matrix type of the mapped data
  * \param Options specifies whether the pointer is \c Aligned, or \c Unaligned.
  *                The default is \c Unaligned.
  *
  * This class represents a matrix or vector expression mapping an existing array of data.
  * It can be used to let Eigen interface without any overhead with non-Eigen data structures,
  * such as plain C arrays or structures from other libraries.
  *
  * \b Tip: to change the array of data mapped by a Map object, you can use the C++
  * placement new syntax:
  *
  * Example: \include Map_placement_new.cpp
  * Output: \verbinclude Map_placement_new.out
  *
  * This class is the return type of Matrix::Map() but can also be used directly.
  *
  * \sa Matrix::Map()
  */
template<typename MatrixType, int Options>
struct ei_traits<Map<MatrixType, Options> > : public ei_traits<MatrixType>
{
  enum {
    Flags = (Options&Aligned)==Aligned ? ei_traits<MatrixType>::Flags |  AlignedBit
                                       : ei_traits<MatrixType>::Flags & ~AlignedBit
  };
};

template<typename MatrixType, int Options> class Map
  : public MapBase<Map<MatrixType, Options>,
                   typename MatrixType::template MakeBase< Map<MatrixType, Options> >::Type>
{
  public:

    typedef MapBase<Map,typename MatrixType::template MakeBase<Map>::Type> Base;
    _EIGEN_GENERIC_PUBLIC_INTERFACE(Map)

    inline int stride() const { return this->innerSize(); }

    inline Map(const Scalar* data) : Base(data) {}

    inline Map(const Scalar* data, int size) : Base(data, size) {}

    inline Map(const Scalar* data, int rows, int cols) : Base(data, rows, cols) {}

    inline void resize(int rows, int cols)
    {
      EIGEN_ONLY_USED_FOR_DEBUG(rows);
      EIGEN_ONLY_USED_FOR_DEBUG(cols);
      ei_assert(rows == this->rows());
      ei_assert(cols == this->cols());
    }

    inline void resize(int size)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(MatrixType)
      EIGEN_ONLY_USED_FOR_DEBUG(size);
      ei_assert(size == this->size());
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Map)
};

/** Constructor copying an existing array of data.
  * Only for fixed-size matrices and vectors.
  * \param data The array of data to copy
  *
  * \sa Matrix::Map(const Scalar *)
  */
template<typename _Scalar, int _Rows, int _Cols, int _StorageOrder, int _MaxRows, int _MaxCols>
inline Matrix<_Scalar, _Rows, _Cols, _StorageOrder, _MaxRows, _MaxCols>
  ::Matrix(const Scalar *data)
{
  _set_noalias(Eigen::Map<Matrix>(data));
}

#endif // EIGEN_MAP_H
