// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*****************************************************************
 * TensroSyclTuple.h
 *
 * \brief:
 *  Minimal implementation of std::tuple that can be used inside a SYCL kernel.
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSORSYCL_TUPLE_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSORSYCL_TUPLE_HPP
namespace utility {
namespace tuple {
/// \struct EnableIf
/// \brief The EnableIf struct is used to statically define type based on the
/// condition.
template <bool, typename T = void>
struct EnableIf {};
/// \brief specialisation of the \ref EnableIf when the condition is true
template <typename T>
struct EnableIf<true, T> {
  typedef T type;
};

/// \struct Tuple
/// \brief is a fixed-size collection of heterogeneous values
/// \ztparam Ts... - the types of the elements that the tuple stores.
/// Empty list is supported.
template <class... Ts>
struct Tuple {};

/// \brief specialisation of the \ref Tuple class when the tuple has at least
/// one element.
/// \tparam T : the type of the first element in the tuple.
/// \tparam Ts... the rest of the elements in the tuple. Ts... can be empty.
template <class T, class... Ts>
struct Tuple<T, Ts...> {
  Tuple(T t, Ts... ts) : head(t), tail(ts...) {}

  T head;
  Tuple<Ts...> tail;
};

/// \struct ElemTypeHolder
/// \brief ElemTypeHolder class is used to specify the types of the
/// elements inside the tuple
/// \tparam size_t the number of elements inside the tuple
/// \tparam class the tuple class
template <size_t, class>
struct ElemTypeHolder;

/// \brief specialisation of the \ref ElemTypeHolder class when the number
/// elements inside the tuple is 1
template <class T, class... Ts>
struct ElemTypeHolder<0, Tuple<T, Ts...>> {
  typedef T type;
};

/// \brief specialisation of the \ref ElemTypeHolder class when the number of
/// elements inside the tuple is bigger than 1. It recursively call itself to
/// detect the type of each element in the tuple
/// \tparam T : the type of the first element in the tuple.
/// \tparam Ts... the rest of the elements in the tuple. Ts... can be empty.
/// \tparam K is the Kth element in the tuple
template <size_t k, class T, class... Ts>
struct ElemTypeHolder<k, Tuple<T, Ts...>> {
  typedef typename ElemTypeHolder<k - 1, Tuple<Ts...>>::type type;
};

/// get
/// \brief Extracts the first element from the tuple.
/// K=0 represents the first element of the tuple. The tuple cannot be empty.
/// \tparam Ts... are the elements type in the tuple.
/// \param t is the tuple whose contents to extract
/// \return  typename ElemTypeHolder<0, Tuple<Ts...>>::type &>::type
template <size_t k, class... Ts>
typename EnableIf<k == 0,
                  typename ElemTypeHolder<0, Tuple<Ts...>>::type &>::type
get(Tuple<Ts...> &t) {
  return t.head;
}
/// get
/// \brief Extracts the Kth element from the tuple.
/// \tparam K is an integer value in [0,sizeof...(Types)).
/// \tparam T is the (sizeof...(Types) -(K+1)) element in the tuple
/// \tparam Ts... are the elements type in the tuple.
/// \param t is the tuple whose contents to extract
/// \return  typename ElemTypeHolder<K, Tuple<Ts...>>::type &>::type
template <size_t k, class T, class... Ts>
typename EnableIf<k != 0,
                  typename ElemTypeHolder<k, Tuple<T, Ts...>>::type &>::type
get(Tuple<T, Ts...> &t) {
  return get<k - 1>(t.tail);
}

/// get
/// \brief Extracts the first element from the tuple when the tuple and all the
/// elements inside are const.
/// K=0 represents the first element of the tuple. The tuple cannot be empty.
/// \tparam Ts... are the elements type in the tuple.
/// \param t is the const tuple whose contents to extract
/// \return  const typename ElemTypeHolder<0, Tuple<Ts...>>::type &>::type
template <size_t k, class... Ts>
typename EnableIf<k == 0,
                  const typename ElemTypeHolder<0, Tuple<Ts...>>::type &>::type
get(const Tuple<Ts...> &t) {
  return t.head;
}

/// get
/// \brief Extracts the Kth element from the tuple when the tuple and all the
/// elements inside are const.
/// \tparam K is an integer value in [0,sizeof...(Types)).
/// \tparam T is the (sizeof...(Types) -(K+1)) element in the tuple
/// \tparam Ts... are the elements type in the tuple.
/// \param t is the const tuple whose contents to extract
/// \return  const typename ElemTypeHolder<K, Tuple<Ts...>>::type &>::type
template <size_t k, class T, class... Ts>
typename EnableIf<
    k != 0, const typename ElemTypeHolder<k, Tuple<T, Ts...>>::type &>::type
get(const Tuple<T, Ts...> &t) {
  return get<k - 1>(t.tail);
}
/// make_tuple
/// \brief Creates a tuple object, deducing the target type from the types of
/// arguments.
/// \tparam Args the type of the arguments to construct the tuple from
/// \param args zero or more arguments to construct the tuple from
/// \return Tuple<Args...>
template <typename... Args>
Tuple<Args...> make_tuple(Args... args) {
  return Tuple<Args...>(args...);
}

/// size
/// \brief Provides access to the number of elements in a tuple as a
/// compile-time constant expression.
/// \tparam Args the type of the arguments to construct the tuple from
/// \return size_t
template <typename... Args>
static constexpr size_t size(Tuple<Args...> &) {
  return sizeof...(Args);
}

/// \struct Index_list
/// \brief Creates a list of index from the elements in the tuple
/// \tparam Is... a list of index from [0 to sizeof...(tuple elements))
template <size_t... Is>
struct Index_list {};

/// \struct RangeBuilder
/// \brief Collects internal details for generating index ranges [MIN, MAX)
/// Declare primary template for index range builder
/// \tparam MIN is the starting index in the tuple
/// \tparam N represents sizeof..(elements)- sizeof...(Is)
/// \tparam Is... are the list of generated index so far
template <size_t MIN, size_t N, size_t... Is>
struct RangeBuilder;

/// \brief base Step: Specialisation of the \ref RangeBuilder when the
/// MIN==MAX. In this case the Is... is [0 to sizeof...(tuple elements))
/// \tparam MIN is the starting index of the tuple
/// \tparam Is is [0 to sizeof...(tuple elements))
template <size_t MIN, size_t... Is>
struct RangeBuilder<MIN, MIN, Is...> {
  typedef Index_list<Is...> type;
};

/// Induction step: Specialisation of the RangeBuilder class when N!=MIN
/// in this case we are recursively subtracting the N by one and adding one
/// index to Is... list until MIN==N
/// \tparam MIN is the starting index in the tuple
/// \tparam N represents sizeof..(elements)- sizeof...(Is)
/// \tparam Is... are the list of generated index so far
template <size_t MIN, size_t N, size_t... Is>
struct RangeBuilder : public RangeBuilder<MIN, N - 1, N - 1, Is...> {};

/// \brief IndexRange that returns a [MIN, MAX) index range
/// \tparam MIN is the starting index in the tuple
/// \tparam MAX is the size of the tuple
template <size_t MIN, size_t MAX>
using Index_range = typename RangeBuilder<MIN, MAX>::type;

/// append_impl
/// \brief unpacking the elements of the input tuple t and creating a new tuple
/// by adding element a at the end of it.
/// \tparam Args... the type of the elements inside the tuple t
/// \tparam T the type of the new element going to be added at the end of tuple
/// \tparam I... is the list of index from [0 to sizeof...(t))
/// \param t the tuple on which we want to append a.
/// \param a the new elements going to be added to the tuple
/// \return Tuple<Args..., T>
template <typename... Args, typename T, size_t... I>
Tuple<Args..., T> append_impl(utility::tuple::Tuple<Args...> t, T a,
                              utility::tuple::Index_list<I...>) {
  return utility::tuple::make_tuple(get<I>(t)..., a);
}

/// append
/// \brief the deduction function for \ref append_impl that automatically
/// generate the \ref Index_range
/// \tparam Args... the type of the elements inside the tuple t
/// \tparam T the type of the new element going to be added at the end of tuple
/// \param t the tuple on which we want to append a.
/// \param a the new elements going to be added to the tuple
/// \return Tuple<Args..., T>
template <typename... Args, typename T>
Tuple<Args..., T> append(Tuple<Args...> t, T a) {
  return utility::tuple::append_impl(
      t, a, utility::tuple::Index_range<0, sizeof...(Args)>());
}

/// append_impl
/// \brief This is an specialised of \ref append_impl when we want to
/// concatenate
/// tuple t2 at the end of the tuple t1. Here we unpack both tuples, generate
/// the
/// Index_range for each of them and create an output tuple T that contains both
/// elements of t1 and t2.
/// \tparam Args1... the type of the elements inside the tuple t1
/// \tparam Args2... the type of the elements inside the tuple t2
/// \tparam I1... is the list of index from [0 to sizeof...(t1))
/// \tparam I2... is the list of index from [0 to sizeof...(t2))
/// \param t1 is the tuple on which we want to append t2.
/// \param t2 is the tuple that is going to be added on t1.
/// \return Tuple<Args1..., Args2...>
template <typename... Args1, typename... Args2, size_t... I1, size_t... I2>
Tuple<Args1..., Args2...> append_impl(utility::tuple::Tuple<Args1...> t1,
                                      utility::tuple::Tuple<Args2...> t2,
                                      utility::tuple::Index_list<I1...>,
                                      utility::tuple::Index_list<I2...>) {
  return utility::tuple::make_tuple(utility::tuple::get<I1>(t1)...,
                                    utility::tuple::get<I2>(t2)...);
}
/// append
/// \brief deduction function for \ref append_impl when we are appending tuple
/// t1 by tuple t2. In this case the \ref Index_range for both tuple are
/// automatically generated.
/// \tparam Args1... the type of the elements inside the tuple t1
/// \tparam Args2... the type of the elements inside the tuple t2
/// \param t1 is the tuple on which we want to append t2.
/// \param t2 is the tuple that is going to be added on t1.
/// \return Tuple<Args1..., Args2...>
template <typename... Args1, typename... Args2>
Tuple<Args1..., Args2...> append(utility::tuple::Tuple<Args1...> t1,
                                 utility::tuple::Tuple<Args2...> t2) {
  return utility::tuple::append_impl(
      t1, t2, utility::tuple::Index_range<0, sizeof...(Args1)>(),
      utility::tuple::Index_range<0, sizeof...(Args2)>());
}
}  // tuple
}  // utility
#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSORSYCL_TUPLE_HPP
