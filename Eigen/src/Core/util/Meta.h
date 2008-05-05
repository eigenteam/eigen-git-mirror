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

#ifndef EIGEN_META_H
#define EIGEN_META_H

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

struct ei_meta_true {};
struct ei_meta_false {};


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

template<typename T> struct ei_functor_traits
{
  enum
  {
    Cost = 10,
    IsVectorizable = false
  };
};

template<typename T> struct ei_packet_traits
{
  typedef T type;
  enum {size=1};
};

template<typename Scalar, int Size, unsigned int SuggestedFlags>
class ei_corrected_matrix_flags
{
    enum { is_vectorizable
            = ei_packet_traits<Scalar>::size > 1
              && (Size%ei_packet_traits<Scalar>::size==0),
          _flags1 = (SuggestedFlags & ~(EvalBeforeNestingBit | EvalBeforeAssigningBit)) | Like1DArrayBit
    };

  public:
    enum { ret = is_vectorizable
                  ? _flags1 | VectorizableBit
                  : _flags1 & ~VectorizableBit
    };
};

template<int _Rows, int _Cols> struct ei_size_at_compile_time
{
  enum { ret = (_Rows==Dynamic || _Cols==Dynamic) ? Dynamic : _Rows * _Cols };
};

template<typename T> class ei_eval
{
    typedef typename ei_traits<T>::Scalar _Scalar;
    enum {_MaxRows = ei_traits<T>::MaxRowsAtCompileTime,
          _MaxCols = ei_traits<T>::MaxColsAtCompileTime,
          _Flags = ei_traits<T>::Flags
    };

  public:
    typedef Matrix<_Scalar,
                  ei_traits<T>::RowsAtCompileTime,
                  ei_traits<T>::ColsAtCompileTime,
                  ei_corrected_matrix_flags<_Scalar, ei_size_at_compile_time<_MaxRows,_MaxCols>::ret, _Flags>::ret,
                  ei_traits<T>::MaxRowsAtCompileTime,
                  ei_traits<T>::MaxColsAtCompileTime> type;
};

template<typename T> struct ei_unref { typedef T type; };
template<typename T> struct ei_unref<T&> { typedef T type; };

template<typename T> struct ei_unconst { typedef T type; };
template<typename T> struct ei_unconst<const T> { typedef T type; };

template<typename T> struct ei_is_temporary
{
  enum { ret = 0 };
};

template<typename T> struct ei_is_temporary<Temporary<T> >
{
  enum { ret = 1 };
};

template<typename T, int n=1> struct ei_nested
{
  typedef typename ei_meta_if<
    ei_is_temporary<T>::ret,
    T,
    typename ei_meta_if<
      ei_traits<T>::Flags & EvalBeforeNestingBit
      || (n+1) * NumTraits<typename ei_traits<T>::Scalar>::ReadCost < (n-1) * T::CoeffReadCost,
      typename ei_eval<T>::type,
      const T&
    >::ret
  >::ret type;
};

#endif // EIGEN_META_H
