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
 * $Id: Vector.h,v 1.24 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_XPR_VECTOR_H
#define TVMET_XPR_VECTOR_H

#include <tvmet/meta/Vector.h>
#include <tvmet/loop/Vector.h>

namespace tvmet {


/* forwards */
template <class T, int Sz> class Vector;

/**
 * \class XprVector Vector.h "tvmet/xpr/Vector.h"
 * \brief Represents the expression for vectors at any node in the parse tree.
 *
 * Specifically, XprVector is the class that wraps the expression, and the
 * expression itself is represented by the template parameter E. The
 * class XprVector is known as an anonymizing expression wrapper because
 * it can hold any subexpression of arbitrary complexity, allowing
 * clients to work with any expression by holding on to it via the
 * wrapper, without having to know the name of the type object that
 * actually implements the expression.
 * \note leave the Ctors non-explicit to allow implicit type conversation.
 */
template<class E, int Sz>
class XprVector : public TvmetBase< XprVector<E, Sz> >
{
  XprVector();
  XprVector& operator=(const XprVector&);

public:
  typedef typename E::value_type			value_type;

public:
  /** Dimensions. */
  enum {
    Size = Sz			/**< The size of the vector. */
  };

public:
  /** Complexity counter */
  enum {
    ops_assign = Size,
    ops        = E::ops,
    use_meta   = ops_assign < TVMET_COMPLEXITY_V_ASSIGN_TRIGGER ? true : false
  };

public:
  /** Constructor. */
  explicit XprVector(const E& e)
    : m_expr(e)
  { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprVector(const XprVector& e)
    : m_expr(e.m_expr)
  { }
#endif

 /** const index operator for vectors. */
  value_type operator()(int i) const {
    assert(i < Size);
    return m_expr(i);
  }

  /** const index operator for vectors. */
  value_type operator[](int i) const {
    return this->operator()(i);
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
  /** assign this expression to Vector dest. */
  template<class Dest, class Assign>
  void assign_to(Dest& dest, const Assign& assign_fn) const {
    /* here is a way for caching, since each complex 'Node'
       is of type XprVector. */
    do_assign(dispatch<use_meta>(), dest, *this, assign_fn);
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, int l=0) const {
    os << IndentLevel(l++)
       << "XprVector["
       << (use_meta ? "M" :  "L") << ", O=" << ops << "]<"
       << std::endl;
    m_expr.print_xpr(os, l);
    os << IndentLevel(l)
       << "Sz=" << Size << std::endl;
    os << IndentLevel(--l) << ">"
       << ((l != 0) ? "," : "") << std::endl;
  }

private:
  const E						m_expr;
};


} // namespace tvmet

#include <tvmet/Functional.h>

#include <tvmet/xpr/BinOperator.h>
#include <tvmet/xpr/UnOperator.h>
#include <tvmet/xpr/Literal.h>

#include <tvmet/xpr/VectorFunctions.h>
#include <tvmet/xpr/VectorBinaryFunctions.h>
#include <tvmet/xpr/VectorOperators.h>
#include <tvmet/xpr/Eval.h>

#endif // TVMET_XPR_VECTOR_H

// Local Variables:
// mode:C++
// End:
