// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_MACROS_H
#define EIGEN_MACROS_H

#undef minor

#ifdef EIGEN_DONT_USE_UNROLLED_LOOPS
#define EIGEN_UNROLLING_LIMIT 0
#endif

/** Defines the maximal loop size to enable meta unrolling of loops */
#ifndef EIGEN_UNROLLING_LIMIT
#define EIGEN_UNROLLING_LIMIT 400
#endif

#ifndef EIGEN_PARALLELIZATION_TRESHOLD
#define EIGEN_PARALLELIZATION_TRESHOLD 2000
#endif

#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER RowMajorBit
#else
#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER 0
#endif

#define EIGEN_DEFAULT_MATRIX_FLAGS EIGEN_DEFAULT_MATRIX_STORAGE_ORDER

#define USING_PART_OF_NAMESPACE_EIGEN \
EIGEN_USING_MATRIX_TYPEDEFS \
using Eigen::Matrix; \
using Eigen::MatrixBase;

#ifdef NDEBUG
#define EIGEN_NO_DEBUG
#endif

#ifndef ei_assert
#ifdef EIGEN_NO_DEBUG
#define ei_assert(x)
#else
#define ei_assert(x) assert(x)
#endif
#endif

#ifdef EIGEN_INTERNAL_DEBUGGING
#define ei_internal_assert(x) ei_assert(x);
#else
#define ei_internal_assert(x)
#endif

#ifdef EIGEN_NO_DEBUG
#define EIGEN_ONLY_USED_FOR_DEBUG(x) (void)x
#else
#define EIGEN_ONLY_USED_FOR_DEBUG(x)
#endif

#ifdef EIGEN_USE_OPENMP
# ifdef __INTEL_COMPILER
#   define EIGEN_PRAGMA_OMP_PARALLEL _Pragma("omp parallel default(none) shared(other)")
# else
#   define EIGEN_PRAGMA_OMP_PARALLEL _Pragma("omp parallel default(none)")
# endif
# define EIGEN_RUN_PARALLELIZABLE_LOOP(condition) \
  if(condition) \
  { \
    EIGEN_PRAGMA_OMP_PARALLEL \
    { \
      _Pragma("omp for") \
      EIGEN_THE_PARALLELIZABLE_LOOP \
    } \
  } \
  else \
  { \
    EIGEN_THE_PARALLELIZABLE_LOOP \
  }
#else // EIGEN_USE_OPENMP
# define EIGEN_RUN_PARALLELIZABLE_LOOP(condition) EIGEN_THE_PARALLELIZABLE_LOOP
#endif


// FIXME with the always_inline attribute,
// gcc 3.4.x reports the following compilation error:
//   Eval.h:91: sorry, unimplemented: inlining failed in call to 'const Eigen::Eval<Derived> Eigen::MatrixBase<Scalar, Derived>::eval() const'
//    : function body not available
#if (defined __GNUC__) && (__GNUC__!=3)
#define EIGEN_ALWAYS_INLINE __attribute__((always_inline))
#else
#define EIGEN_ALWAYS_INLINE
#endif

#if (defined __GNUC__)
#define EIGEN_ALIGN_128 __attribute__ ((aligned(16)))
#else
#define EIGEN_ALIGN_128
#endif

#define EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Derived, Op) \
template<typename OtherDerived> \
Derived& operator Op(const MatrixBase<OtherDerived>& other) \
{ \
  return Eigen::MatrixBase<Derived>::operator Op(other); \
} \
Derived& operator Op(const Derived& other) \
{ \
  return Eigen::MatrixBase<Derived>::operator Op(other); \
}

#define EIGEN_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, Op) \
template<typename Other> \
Derived& operator Op(const Other& scalar) \
{ \
  return Eigen::MatrixBase<Derived>::operator Op(scalar); \
}

#define EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Derived) \
EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Derived, =) \
EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Derived, +=) \
EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Derived, -=) \
EIGEN_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, *=) \
EIGEN_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, /=)

#define _EIGEN_GENERIC_PUBLIC_INTERFACE(Derived, BaseClass) \
typedef BaseClass Base; \
typedef typename Eigen::ei_traits<Derived>::Scalar Scalar; \
typedef typename Eigen::NumTraits<Scalar>::Real RealScalar; \
typedef typename Base::PacketScalar PacketScalar; \
typedef typename Eigen::ei_nested<Derived>::type Nested; \
typedef typename Eigen::ei_eval<Derived>::type Eval; \
enum { RowsAtCompileTime = Base::RowsAtCompileTime, \
       ColsAtCompileTime = Base::ColsAtCompileTime, \
       MaxRowsAtCompileTime = Base::MaxRowsAtCompileTime, \
       MaxColsAtCompileTime = Base::MaxColsAtCompileTime, \
       SizeAtCompileTime = Base::SizeAtCompileTime, \
       MaxSizeAtCompileTime = Base::MaxSizeAtCompileTime, \
       IsVectorAtCompileTime = Base::IsVectorAtCompileTime, \
       Flags = Base::Flags, \
       CoeffReadCost = Base::CoeffReadCost };

#define EIGEN_GENERIC_PUBLIC_INTERFACE(Derived) \
_EIGEN_GENERIC_PUBLIC_INTERFACE(Derived, Eigen::MatrixBase<Derived>) \
friend class Eigen::MatrixBase<Derived>;

#define EIGEN_ENUM_MIN(a,b) (((int)a <= (int)b) ? (int)a : (int)b)

#endif // EIGEN_MACROS_H
