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

#ifndef EIGEN_UTIL_H
#define EIGEN_UTIL_H

#ifdef EIGEN_DONT_USE_UNROLLED_LOOPS
#define EIGEN_UNROLLING_LIMIT 0
#endif

/** Defines the maximal loop size to enable meta unrolling of loops */
#ifndef EIGEN_UNROLLING_LIMIT
#define EIGEN_UNROLLING_LIMIT 400
#endif

#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER RowMajorBit
#else
#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER 0
#endif

#undef minor

#define USING_PART_OF_NAMESPACE_EIGEN \
EIGEN_USING_MATRIX_TYPEDEFS \
using Eigen::Matrix; \
using Eigen::MatrixBase;

#ifdef NDEBUG
#define EIGEN_NO_DEBUG
#endif

#ifdef EIGEN_NO_DEBUG
#define ei_assert(x)
#else
#define ei_assert(x) assert(x)
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

// FIXME with the always_inline attribute,
// gcc 3.4.x reports the following compilation error:
//   Eval.h:91: sorry, unimplemented: inlining failed in call to 'const Eigen::Eval<Derived> Eigen::MatrixBase<Scalar, Derived>::eval() const'
//    : function body not available
#if (defined __GNUC__) && (__GNUC__!=3)
#define EIGEN_ALWAYS_INLINE __attribute__((always_inline))
#else
#define EIGEN_ALWAYS_INLINE
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
typedef typename Eigen::ei_xpr_copy<Derived>::type XprCopy; \
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

const int Dynamic = 10000;

// matrix/expression flags
const unsigned int RowMajorBit = 0x1;
const unsigned int EvalBeforeNestingBit = 0x2;
const unsigned int EvalBeforeAssigningBit = 0x4;
const unsigned int LargeBit = 0x8;
const unsigned int VectorizableBit = 0x10;


enum { ConditionalJumpCost = 5 };

enum CornerType { TopLeft, TopRight, BottomLeft, BottomRight };

enum DirectionType { Vertical, Horizontal };

enum ProductEvaluationMode { UnrolledDotProduct, CacheOptimal };

// just a workaround because GCC seems to not really like empty structs
#ifdef __GNUG__
  struct ei_empty_struct{char _ei_dummy_;};
  #define EIGEN_EMPTY_STRUCT : Eigen::ei_empty_struct
#else
  #define EIGEN_EMPTY_STRUCT
#endif

//classes inheriting ei_no_assignment_operator don't generate a default operator=.
class ei_no_assignment_operator
{
  private:
    ei_no_assignment_operator& operator=(const ei_no_assignment_operator&);
};

template<int Value> class ei_int_if_dynamic EIGEN_EMPTY_STRUCT
{
  public:
    ei_int_if_dynamic() {}
    explicit ei_int_if_dynamic(int) {}
    static int value() { return Value; }
    void setValue(int) {}
};

template<> class ei_int_if_dynamic<Dynamic>
{
    int m_value;
    ei_int_if_dynamic() {}
  public:
    explicit ei_int_if_dynamic(int value) : m_value(value) {}
    int value() const { return m_value; }
    void setValue(int value) { m_value = value; }
};


template <bool Condition, class Then, class Else>
struct ei_meta_if { typedef Then ret; };

template <class Then, class Else>
struct ei_meta_if <false, Then, Else> { typedef Else ret; };

template<typename T, typename U> struct ei_is_same_type { enum { ret = 0 }; };
template<typename T> struct ei_is_same_type<T,T> { enum { ret = 1 }; };


/** \internal
  * Convenient struct to get the result type of a unary or binary functor.
  *
  * It supports both the current STL mechanism (using the result_type member) as well as
  * upcoming next STL generation (using a templated result member).
  * If none of these members is provided, then the type of the first argument is returned.
  */
template<typename T> struct ei_result_of {};

struct ei_has_none {int a[1];};
struct ei_has_std_result_type {int a[2];};
struct ei_has_tr1_result {int a[3];};

template<typename Func, typename ArgType, int SizeOf=sizeof(ei_has_none)>
struct ei_unary_result_of_select {typedef ArgType type;};

template<typename Func, typename ArgType>
struct ei_unary_result_of_select<Func, ArgType, sizeof(ei_has_std_result_type)> {typedef typename Func::result_type type;};

template<typename Func, typename ArgType>
struct ei_unary_result_of_select<Func, ArgType, sizeof(ei_has_tr1_result)> {typedef typename Func::template result<Func(ArgType)>::type type;};

template<typename Func, typename ArgType>
struct ei_result_of<Func(ArgType)> {
    template<typename T>
    static ei_has_std_result_type testFunctor(T const *, typename T::result_type const * = 0);
    template<typename T>
    static ei_has_tr1_result      testFunctor(T const *, typename T::template result<T(ArgType)>::type const * = 0);
    static ei_has_none            testFunctor(...);

    // note that the following indirection is needed for gcc-3.3
    enum {FunctorType = sizeof(testFunctor(static_cast<Func*>(0)))};
    typedef typename ei_unary_result_of_select<Func, ArgType, FunctorType>::type type;
};

template<typename Func, typename ArgType0, typename ArgType1, int SizeOf=sizeof(ei_has_none)>
struct ei_binary_result_of_select {typedef ArgType0 type;};

template<typename Func, typename ArgType0, typename ArgType1>
struct ei_binary_result_of_select<Func, ArgType0, ArgType1, sizeof(ei_has_std_result_type)>
{typedef typename Func::result_type type;};

template<typename Func, typename ArgType0, typename ArgType1>
struct ei_binary_result_of_select<Func, ArgType0, ArgType1, sizeof(ei_has_tr1_result)>
{typedef typename Func::template result<Func(ArgType0,ArgType1)>::type type;};

template<typename Func, typename ArgType0, typename ArgType1>
struct ei_result_of<Func(ArgType0,ArgType1)> {
    template<typename T>
    static ei_has_std_result_type testFunctor(T const *, typename T::result_type const * = 0);
    template<typename T>
    static ei_has_tr1_result      testFunctor(T const *, typename T::template result<T(ArgType0,ArgType1)>::type const * = 0);
    static ei_has_none            testFunctor(...);

    // note that the following indirection is needed for gcc-3.3
    enum {FunctorType = sizeof(testFunctor(static_cast<Func*>(0)))};
    typedef typename ei_binary_result_of_select<Func, ArgType0, ArgType1, FunctorType>::type type;
};

#endif // EIGEN_UTIL_H
