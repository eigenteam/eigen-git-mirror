/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * lesser General Public License for more details.
 *
 * You should have received a copy of the GNU lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: Vector.h,v 1.44 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_VECTOR_H
#define TVMET_VECTOR_H

#include <iterator>					// reverse_iterator

#include <tvmet/tvmet.h>
#include <tvmet/TypePromotion.h>
#include <tvmet/CommaInitializer.h>
#include <tvmet/RunTimeError.h>

#include <tvmet/xpr/Vector.h>

namespace tvmet {


/* forwards */
template<class T, std::size_t Sz> class Vector;


/**
 * \class VectorConstReference Vector.h "tvmet/Vector.h"
 * \brief Const value iterator for ET
 */
template<class T, std::size_t Sz>
class VectorConstReference
  : public TvmetBase< VectorConstReference<T, Sz> >
{
public: // types
  typedef T 						value_type;
  typedef T*						pointer;
  typedef const T*					const_pointer;

public:
  /** Dimensions. */
  enum {
    Size = Sz			/**< The size of the vector. */
  };

public:
  /** Complexity counter. */
  enum {
    ops        = Size
  };

private:
  VectorConstReference();
  VectorConstReference& operator=(const VectorConstReference&);

public:
  /** Constructor. */
  explicit VectorConstReference(const Vector<T, Size>& rhs)
    : m_data(rhs.data())
  { }

  /** Constructor by a given memory pointer. */
  explicit VectorConstReference(const_pointer data)
    : m_data(data)
  { }

public: // access operators
  /** access by index. */
  value_type operator()(std::size_t i) const {
    TVMET_RT_CONDITION(i < Size, "VectorConstReference Bounce Violation")
    return m_data[i];
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l)
       << "VectorConstReference[O=" << ops << "]<"
       << "T=" << typeid(T).name() << ">,"
       << std::endl;
  }

private:
  const_pointer _tvmet_restrict 			m_data;
};


/**
 * \class Vector Vector.h "tvmet/Vector.h"
 * \brief Compile time fixed length vector with evaluation on compile time.
 */
template<class T, std::size_t Sz>
class Vector
{
public:
  /** Data type of the tvmet::Vector. */
  typedef T     					value_type;

  /** Reference type of the tvmet::Vector data elements. */
  typedef T&     					reference;

  /** const reference type of the tvmet::Vector data elements. */
  typedef const T&     					const_reference;

  /** STL iterator interface. */
  typedef T*     					iterator;

  /** STL const_iterator interface. */
  typedef const T*     					const_iterator;

  /** STL reverse iterator interface. */
  typedef std::reverse_iterator<iterator> 		reverse_iterator;

  /** STL const reverse iterator interface. */
  typedef std::reverse_iterator<const_iterator> 	const_reverse_iterator;

public:
  /** Dimensions. */
  enum {
    Size = Sz			/**< The size of the vector. */
  };

public:
  /** Complexity counter. */
  enum {
    ops_assign = Size,
    ops        = ops_assign,
    use_meta   = ops < TVMET_COMPLEXITY_V_ASSIGN_TRIGGER ? true : false
  };

public: // STL  interface
  /** STL iterator interface. */
  iterator begin() { return m_data; }

  /** STL iterator interface. */
  iterator end() { return m_data + Size; }

  /** STL const_iterator interface. */
  const_iterator begin() const { return m_data; }

  /** STL const_iterator interface. */
  const_iterator end() const { return m_data + Size; }

  /** STL reverse iterator interface reverse begin. */
  reverse_iterator rbegin() { return reverse_iterator( end() ); }

  /** STL const reverse iterator interface reverse begin. */
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator( end() );
  }

  /** STL reverse iterator interface reverse end. */
  reverse_iterator rend() { return reverse_iterator( begin() ); }

  /** STL const reverse iterator interface reverse end. */
  const_reverse_iterator rend() const {
    return const_reverse_iterator( begin() );
  }

  /** STL vector front element. */
  value_type front() { return m_data[0]; }

  /** STL vector const front element. */
  const_reference front() const { return m_data[0]; }

  /** STL vector back element. */
  value_type back() { return m_data[Size-1]; }

  /** STL vector const back element. */
  const_reference back() const { return m_data[Size-1]; }

  /** STL vector empty() - returns allways false. */
  static bool empty() { return false; }

  /** The size of the vector. */
  static std::size_t size() { return Size; }

  /** STL vector max_size() - returns allways Size. */
  static std::size_t max_size() { return Size; }

public:
  /** Default Destructor */
  ~Vector() {
#if defined(TVMET_DYNAMIC_MEMORY)
    delete [] m_data;
#endif
  }

  /** Default Constructor. The allocated memory region isn't cleared. If you want
   a clean use the constructor argument zero. */
  explicit Vector()
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  { }

  /** Copy Constructor, not explicit! */
  Vector(const Vector& rhs)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    *this = XprVector<ConstReference, Size>(rhs.const_ref());
  }

  /**
   * Constructor with STL iterator interface. The data will be copied into the
   * vector self, there isn't any stored reference to the array pointer.
   */
  template<class InputIterator>
  explicit Vector(InputIterator first, InputIterator last)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    TVMET_RT_CONDITION( static_cast<std::size_t>(std::distance(first, last)) <= Size,
			"InputIterator doesn't fits in size" )
    std::copy(first, last, m_data);
  }

  /**
   * Constructor with STL iterator interface. The data will be copied into the
   * vector self, there isn't any stored reference to the array pointer.
   */
  template<class InputIterator>
  explicit Vector(InputIterator first, std::size_t sz)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    TVMET_RT_CONDITION( sz <= Size, "InputIterator doesn't fits in size" )
    std::copy(first, first + sz, m_data);
  }

  /** Constructor with initializer for all elements.  */
  explicit Vector(value_type rhs)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    typedef XprLiteral<value_type> expr_type;
    *this = XprVector<expr_type, Size>(expr_type(rhs));
  }

  /** Default Constructor with initializer list. */
  explicit Vector(value_type x0, value_type x1)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    TVMET_CT_CONDITION(2 <= Size, ArgumentList_is_too_long)
    m_data[0] = x0; m_data[1] = x1;
  }

  /** Default Constructor with initializer list. */
  explicit Vector(value_type x0, value_type x1, value_type x2)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    TVMET_CT_CONDITION(3 <= Size, ArgumentList_is_too_long)
    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2;
  }

  /** Default Constructor with initializer list. */
  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    TVMET_CT_CONDITION(4 <= Size, ArgumentList_is_too_long)
    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3;
  }

  /** Default Constructor with initializer list. */
  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3,
		  value_type x4)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    TVMET_CT_CONDITION(5 <= Size, ArgumentList_is_too_long)
    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3; m_data[4] = x4;
  }

  /** Default Constructor with initializer list. */
  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3,
		  value_type x4, value_type x5)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    TVMET_CT_CONDITION(6 <= Size, ArgumentList_is_too_long)
    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3; m_data[4] = x4;
    m_data[5] = x5;
  }

  /** Default Constructor with initializer list. */
  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3,
		  value_type x4, value_type x5, value_type x6)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    TVMET_CT_CONDITION(7 <= Size, ArgumentList_is_too_long)
    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3; m_data[4] = x4;
    m_data[5] = x5; m_data[6] = x6;
  }

  /** Default Constructor with initializer list. */
  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3,
		  value_type x4, value_type x5, value_type x6, value_type x7)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    TVMET_CT_CONDITION(8 <= Size, ArgumentList_is_too_long)
    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3; m_data[4] = x4;
    m_data[5] = x5; m_data[6] = x6; m_data[7] = x7;
  }

  /** Default Constructor with initializer list. */
  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3,
		  value_type x4, value_type x5, value_type x6, value_type x7,
		  value_type x8)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    TVMET_CT_CONDITION(9 <= Size, ArgumentList_is_too_long)
    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3; m_data[4] = x4;
    m_data[5] = x5; m_data[6] = x6; m_data[7] = x7; m_data[8] = x8;
  }

  /** Default Constructor with initializer list. */
  explicit Vector(value_type x0, value_type x1, value_type x2, value_type x3,
		  value_type x4, value_type x5, value_type x6, value_type x7,
		  value_type x8, value_type x9)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    TVMET_CT_CONDITION(10 <= Size, ArgumentList_is_too_long)
    m_data[0] = x0; m_data[1] = x1; m_data[2] = x2; m_data[3] = x3; m_data[4] = x4;
    m_data[5] = x5; m_data[6] = x6; m_data[7] = x7; m_data[8] = x8; m_data[9] = x9;
  }

  /** Construct a vector by expression. */
  template <class E>
  explicit Vector(const XprVector<E, Size>& e)
#if defined(TVMET_DYNAMIC_MEMORY)
    : m_data( new value_type[Size] )
#endif
  {
    *this = e;
  }

  /** Assign a value_type on array, this can be used for a single value
      or a comma separeted list of values. */
  CommaInitializer<Vector, Size> operator=(value_type rhs) {
    return CommaInitializer<Vector, Size>(*this, rhs);
  }

public: // access operators
  value_type* _tvmet_restrict data() { return m_data; }
  const value_type* _tvmet_restrict data() const { return m_data; }

public: // index access operators
  value_type& _tvmet_restrict operator()(std::size_t i) {
    // Note: g++-2.95.3 does have problems on typedef reference
    TVMET_RT_CONDITION(i < Size, "Vector Bounce Violation")
    return m_data[i];
  }

  value_type operator()(std::size_t i) const {
    TVMET_RT_CONDITION(i < Size, "Vector Bounce Violation")
    return m_data[i];
  }

  value_type& _tvmet_restrict operator[](std::size_t i) {
    // Note: g++-2.95.3 does have problems on typedef reference
    return this->operator()(i);
  }

  value_type operator[](std::size_t i) const {
    return this->operator()(i);
  }

public: // ET interface
  typedef VectorConstReference<T, Size>    		ConstReference;

  /** Return a const Reference of the internal data */
  ConstReference const_ref() const { return ConstReference(*this); }

  /** Return the vector as const expression. */
  XprVector<ConstReference, Size> as_expr() const {
    return XprVector<ConstReference, Size>(this->const_ref());
  }

private:
  /** Wrapper for meta assign. */
  template<class Dest, class Src, class Assign>
  static inline
  void do_assign(dispatch<true>, Dest& dest, const Src& src, const Assign& assign_fn) {
    meta::Vector<Size, 0>::assign(dest, src, assign_fn);
  }

  /** Wrapper for loop assign. */
  template<class Dest, class Src, class Assign>
  static inline
  void do_assign(dispatch<false>, Dest& dest, const Src& src, const Assign& assign_fn) {
    loop::Vector<Size>::assign(dest, src, assign_fn);
  }

public:
  /** assign this to a vector expression using the functional assign_fn. */
  template<class T2, class Assign>
  void assign_to(Vector<T2, Size>& dest, const Assign& assign_fn) const {
    do_assign(dispatch<use_meta>(), dest, *this, assign_fn);
  }

public:   // assign operations
  /** assign a given Vector element wise to this vector.
      The operator=(const Vector&) is compiler generated. */
  template<class T2>
  Vector& operator=(const Vector<T2, Size>& rhs) {
    rhs.assign_to(*this, Fcnl_assign<value_type, T2>());
    return *this;
  }

  /** assign a given XprVector element wise to this vector. */
  template<class E>
  Vector& operator=(const XprVector<E, Size>& rhs) {
    rhs.assign_to(*this, Fcnl_assign<value_type, typename E::value_type>());
    return *this;
  }

private:
  template<class Obj, std::size_t LEN> friend class CommaInitializer;

  /** This is a helper for assigning a comma separated initializer
      list. It's equal to Vector& operator=(value_type) which does
      replace it. */
  Vector& assign_value(value_type rhs) {
    typedef XprLiteral<value_type> 			expr_type;
    *this = XprVector<expr_type, Size>(expr_type(rhs));
    return *this;
  }

public: // math operators with scalars
  // NOTE: this meaning is clear - element wise ops even if not in ns element_wise
  Vector& operator+=(value_type) TVMET_CXX_ALWAYS_INLINE;
  Vector& operator-=(value_type) TVMET_CXX_ALWAYS_INLINE;
  Vector& operator*=(value_type) TVMET_CXX_ALWAYS_INLINE;
  Vector& operator/=(value_type) TVMET_CXX_ALWAYS_INLINE;

  Vector& operator%=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
  Vector& operator^=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
  Vector& operator&=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
  Vector& operator|=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
  Vector& operator<<=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
  Vector& operator>>=(std::size_t) TVMET_CXX_ALWAYS_INLINE;

public: // math assign operators with vectors
  // NOTE: access using the operators in ns element_wise, since that's what is does
  template <class T2> Vector& M_add_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& M_sub_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& M_mul_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& M_div_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& M_mod_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& M_xor_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& M_and_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& M_or_eq (const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& M_shl_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& M_shr_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;

public: // math operators with expressions
  // NOTE: access using the operators in ns element_wise, since that's what is does
  template <class E> Vector& M_add_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& M_sub_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& M_mul_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& M_div_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& M_mod_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& M_xor_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& M_and_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& M_or_eq (const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& M_shl_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& M_shr_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;

public: // aliased math operators with expressions, used with proxy
  template <class T2> Vector& alias_assign(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& alias_add_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& alias_sub_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& alias_mul_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Vector& alias_div_eq(const Vector<T2, Size>&) TVMET_CXX_ALWAYS_INLINE;

  template <class E> Vector& alias_assign(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& alias_add_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& alias_sub_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& alias_mul_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Vector& alias_div_eq(const XprVector<E, Size>&) TVMET_CXX_ALWAYS_INLINE;

public: // io
  /** Structure for info printing as Vector<T, Size>. */
  struct Info : public TvmetBase<Info> {
    std::ostream& print_xpr(std::ostream& os) const {
      os << "Vector<T=" << typeid(value_type).name()
	 << ", Sz=" << Size << ">";
      return os;
    }
  };

  /** Get an info object of this vector. */
  static Info info() { return Info(); }

  /** Member function for expression level printing. */
  std::ostream& print_xpr(std::ostream& os, std::size_t l=0) const;

  /** Member function for printing internal data. */
  std::ostream& print_on(std::ostream& os) const;

private:
  /** The data of vector self. */

#if defined(TVMET_DYNAMIC_MEMORY)
  value_type*						m_data;
#else
  value_type 						m_data[Size];
#endif
};


} // namespace tvmet

#include <tvmet/VectorImpl.h>
#include <tvmet/VectorFunctions.h>
#include <tvmet/VectorBinaryFunctions.h>
#include <tvmet/VectorUnaryFunctions.h>
#include <tvmet/VectorOperators.h>
#include <tvmet/VectorEval.h>
#include <tvmet/AliasProxy.h>

#endif // TVMET_VECTOR_H

// Local Variables:
// mode:C++
// End:
