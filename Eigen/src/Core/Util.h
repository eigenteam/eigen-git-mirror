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

#ifndef EIGEN_UTIL_H
#define EIGEN_UTIL_H

#ifdef EIGEN_DONT_USE_UNROLLED_LOOPS
#define EIGEN_UNROLLED_LOOPS (false)
#else
#define EIGEN_UNROLLED_LOOPS (true)
#endif

#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER RowMajor
#else
#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER ColumnMajor
#endif

#undef minor

#define USING_PART_OF_NAMESPACE_EIGEN \
EIGEN_USING_MATRIX_TYPEDEFS \
using Eigen::Matrix; \
using Eigen::MatrixBase;

#ifdef EIGEN_INTERNAL_DEBUGGING
#define eigen_internal_assert(x) assert(x);
#else
#define eigen_internal_assert(x)
#endif

#ifdef NDEBUG
#define EIGEN_ONLY_USED_FOR_DEBUG(x) (void)x
#else
#define EIGEN_ONLY_USED_FOR_DEBUG(x)
#endif

#ifdef __GNUC__
#define EIGEN_ALWAYS_INLINE __attribute__((always_inline))
#else
#define EIGEN_ALWAYS_INLINE
#endif

#define EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Derived, Op) \
template<typename OtherScalar, typename OtherDerived> \
Derived& operator Op(const MatrixBase<OtherScalar, OtherDerived>& other) \
{ \
  return MatrixBase<Scalar, Derived>::operator Op(other); \
} \
Derived& operator Op(const Derived& other) \
{ \
  return MatrixBase<Scalar, Derived>::operator Op(other); \
}

#define EIGEN_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, Op) \
template<typename Other> \
Derived& operator Op(const Other& scalar) \
{ \
  return MatrixBase<Scalar, Derived>::operator Op(scalar); \
}

#define EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Derived) \
EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Derived, =) \
EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Derived, +=) \
EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Derived, -=) \
EIGEN_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, *=) \
EIGEN_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, /=)

#define EIGEN_ENUM_MIN(a,b) (((int)a <= (int)b) ? (int)a : (int)b)

const int Dynamic = -10;
const int ColumnMajor = 0;
const int RowMajor = 1;

//classes inheriting NoOperatorEquals don't generate a default operator=.
class NoOperatorEquals
{
  private:
    NoOperatorEquals& operator=(const NoOperatorEquals&);
};

template<int Value> class IntAtRunTimeIfDynamic
{
  public:
    IntAtRunTimeIfDynamic() {}
    explicit IntAtRunTimeIfDynamic(int) {}
    static int value() { return Value; }
    void setValue(int) {}
};

template<> class IntAtRunTimeIfDynamic<Dynamic>
{
    int m_value;
    IntAtRunTimeIfDynamic() {}
  public:
    explicit IntAtRunTimeIfDynamic(int value) : m_value(value) {}
    int value() const { return m_value; }
    void setValue(int value) { m_value = value; }
};

#endif // EIGEN_UTIL_H
