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

#include <iterator> // reverse_iterator
#include <cassert>

#include <tvmet/tvmet.h>
#include <tvmet/TypePromotion.h>
#include <tvmet/CommaInitializer.h>

#include <tvmet/xpr/Vector.h>

namespace tvmet {


/* forwards */
template<class T, int Sz> class Vector;


/**
 * \class VectorConstRef Vector.h "tvmet/Vector.h"
 * \brief Const value iterator for ET
 */
template<class T, int Sz>
class VectorConstRef
  : public TvmetBase< VectorConstRef<T, Sz> >
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
  VectorConstRef();
  VectorConstRef& operator=(const VectorConstRef&);

public:
  /** Constructor. */
  explicit VectorConstRef(const Vector<T, Size>& rhs)
    : m_array(rhs.array())
  { }

  /** Constructor by a given memory pointer. */
  explicit VectorConstRef(const_pointer data)
    : m_array(data)
  { }

public: // access operators
  /** access by index. */
  value_type operator()(int i) const {
    assert(i < Size);
    return m_array[i];
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, int l=0) const {
    os << IndentLevel(l)
       << "VectorConstRef[O=" << ops << "]<"
       << "T=" << typeid(T).name() << ">,"
       << std::endl;
  }

private:
  const_pointer _tvmet_restrict 			m_array;
};


/**
 * \class Vector Vector.h "tvmet/Vector.h"
 * \brief Compile time fixed length vector with evaluation on compile time.
 */
template<class T, int Size>
class Vector
{
public:
  /** Data type of the tvmet::Vector. */
  typedef T     					value_type;

public:
  /** Complexity counter. */
  enum {
    ops_assign = Size,
    ops        = ops_assign,
    use_meta   = ops < TVMET_COMPLEXITY_V_ASSIGN_TRIGGER ? true : false
  };

public:
  /** Default Destructor */
  ~Vector() {}

  /** Default Constructor. Does nothing. */
  explicit Vector() {}

  /** Copy Constructor, not explicit! */
  Vector(const Vector& rhs)
  {
    *this = XprVector<ConstRef, Size>(rhs.constRef());
  }
  
  explicit Vector(const value_type* array)
  {
    for(int i = 0; i < Size; i++) m_array[i] = array[i];
  }

  /** Construct a vector by expression. */
  template <class E>
  explicit Vector(const XprVector<E, Size>& e)
  {
    *this = e;
  }

  /** Assign a value_type on array, this can be used for a single value
      or a comma separeted list of values. */
  CommaInitializer<Vector, Size> operator=(value_type rhs) {
    return CommaInitializer<Vector, Size>(*this, rhs);
  }

public: // access operators
  value_type* _tvmet_restrict array() { return m_array; }
  const value_type* _tvmet_restrict array() const { return m_array; }

public: // index access operators
  value_type& _tvmet_restrict operator()(int i) {
    // Note: g++-2.95.3 does have problems on typedef reference
    assert(i < Size);
    return m_array[i];
  }

  value_type operator()(int i) const {
    assert(i < Size);
    return m_array[i];
  }

  value_type& _tvmet_restrict operator[](int i) {
    // Note: g++-2.95.3 does have problems on typedef reference
    return this->operator()(i);
  }

  value_type operator[](int i) const {
    return this->operator()(i);
  }

public: // ET interface
  typedef VectorConstRef<T, Size>    		ConstRef;

  /** Return a const Reference of the internal data */
  ConstRef constRef() const { return ConstRef(*this); }

  /** Return the vector as const expression. */
  XprVector<ConstRef, Size> expr() const {
    return XprVector<ConstRef, Size>(this->constRef());
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
  template<class Obj, int LEN> friend class CommaInitializer;

  void commaWrite(int index, T rhs)
  {
    m_array[index] = rhs;
  }

public: // math operators with scalars
  // NOTE: this meaning is clear - element wise ops even if not in ns element_wise
  Vector& operator+=(value_type) _tvmet_always_inline;
  Vector& operator-=(value_type) _tvmet_always_inline;
  Vector& operator*=(value_type) _tvmet_always_inline;
  Vector& operator/=(value_type) _tvmet_always_inline;

public: // math assign operators with vectors
  // NOTE: access using the operators in ns element_wise, since that's what is does
  template <class T2> Vector& M_add_eq(const Vector<T2, Size>&) _tvmet_always_inline;
  template <class T2> Vector& M_sub_eq(const Vector<T2, Size>&) _tvmet_always_inline;
  template <class T2> Vector& M_mul_eq(const Vector<T2, Size>&) _tvmet_always_inline;
  template <class T2> Vector& M_div_eq(const Vector<T2, Size>&) _tvmet_always_inline;

public: // math operators with expressions
  // NOTE: access using the operators in ns element_wise, since that's what is does
  template <class E> Vector& M_add_eq(const XprVector<E, Size>&) _tvmet_always_inline;
  template <class E> Vector& M_sub_eq(const XprVector<E, Size>&) _tvmet_always_inline;
  template <class E> Vector& M_mul_eq(const XprVector<E, Size>&) _tvmet_always_inline;
  template <class E> Vector& M_div_eq(const XprVector<E, Size>&) _tvmet_always_inline;

public: // aliased math operators with expressions, used with proxy
  template <class T2> Vector& alias_assign(const Vector<T2, Size>&) _tvmet_always_inline;
  template <class T2> Vector& alias_add_eq(const Vector<T2, Size>&) _tvmet_always_inline;
  template <class T2> Vector& alias_sub_eq(const Vector<T2, Size>&) _tvmet_always_inline;
  template <class T2> Vector& alias_mul_eq(const Vector<T2, Size>&) _tvmet_always_inline;
  template <class T2> Vector& alias_div_eq(const Vector<T2, Size>&) _tvmet_always_inline;

  template <class E> Vector& alias_assign(const XprVector<E, Size>&) _tvmet_always_inline;
  template <class E> Vector& alias_add_eq(const XprVector<E, Size>&) _tvmet_always_inline;
  template <class E> Vector& alias_sub_eq(const XprVector<E, Size>&) _tvmet_always_inline;
  template <class E> Vector& alias_mul_eq(const XprVector<E, Size>&) _tvmet_always_inline;
  template <class E> Vector& alias_div_eq(const XprVector<E, Size>&) _tvmet_always_inline;

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
  std::ostream& print_xpr(std::ostream& os, int l=0) const;

  /** Member function for printing internal data. */
  std::ostream& print_on(std::ostream& os) const;

private:
  /** The data of vector self. */

  value_type m_array[Size];
};

typedef Vector<int, 2> Vector2i;
typedef Vector<int, 3> Vector3i;
typedef Vector<int, 4> Vector4i;
typedef Vector<float, 2> Vector2f;
typedef Vector<float, 3> Vector3f;
typedef Vector<float, 4> Vector4f;
typedef Vector<double, 2> Vector2d;
typedef Vector<double, 3> Vector3d;
typedef Vector<double, 4> Vector4d;

} // namespace tvmet

#include <tvmet/VectorImpl.h>
#include <tvmet/VectorFunctions.h>
#include <tvmet/VectorUnaryFunctions.h>
#include <tvmet/VectorOperators.h>
#include <tvmet/VectorEval.h>
#include <tvmet/AliasProxy.h>

#endif // TVMET_VECTOR_H

// Local Variables:
// mode:C++
// End:
