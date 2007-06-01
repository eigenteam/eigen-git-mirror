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
 * $Id: Matrix.h,v 1.22 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_XPR_MATRIX_H
#define TVMET_XPR_MATRIX_H

#include <tvmet/meta/Matrix.h>
#include <tvmet/loop/Matrix.h>

namespace tvmet {


/* forwards */
template <class T, int Rows, int Cols> class Matrix;

/**
 * \class XprMatrix Matrix.h "tvmet/xpr/Matrix.h"
 * \brief Represents the expression for vectors at any node in the parse tree.
 *
 * Specifically, XprMatrix is the class that wraps the expression, and the
 * expression itself is represented by the template parameter E. The
 * class XprMatrix is known as an anonymizing expression wrapper because
 * it can hold any subexpression of arbitrary complexity, allowing
 * clients to work with any expression by holding on to it via the
 * wrapper, without having to know the name of the type object that
 * actually implements the expression.
 * \note leave the CCtors non-explicit to allow implicit type conversation.
 */
template<class E, int NRows, int NCols>
class XprMatrix
  : public TvmetBase< XprMatrix<E, NRows, NCols> >
{
  XprMatrix();
  XprMatrix& operator=(const XprMatrix&);

public:
  /** Dimensions. */
  enum {
    Rows = NRows,			/**< Number of rows. */
    Cols = NCols,			/**< Number of cols. */
    Size = Rows * Cols			/**< Complete Size of Matrix. */
  };

public:
  /** Complexity counter. */
  enum {
    ops_assign = Rows * Cols,
    ops        = E::ops,
    use_meta   = ops_assign < TVMET_COMPLEXITY_M_ASSIGN_TRIGGER ? true : false
  };

public:
  typedef typename E::value_type			value_type;

public:
  /** Constructor. */
  explicit XprMatrix(const E& e)
    : m_expr(e)
  { }

 /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprMatrix(const XprMatrix& rhs)
    : m_expr(rhs.m_expr)
  { }
#endif

  /** access by index. */
  value_type operator()(int i, int j) const {
    TVMET_RT_CONDITION((i < Rows) && (j < Cols), "XprMatrix Bounce Violation")
    return m_expr(i, j);
  }

private:
  /** Wrapper for meta assign. */
  template<class Dest, class Src, class Assign>
  static inline
  void do_assign(dispatch<true>, Dest& dest, const Src& src, const Assign& assign_fn) {
    meta::Matrix<Rows, Cols, 0, 0>::assign(dest, src, assign_fn);
  }

  /** Wrapper for loop assign. */
  template<class Dest, class Src, class Assign>
  static inline
  void do_assign(dispatch<false>, Dest& dest, const Src& src, const Assign& assign_fn) {
    loop::Matrix<Rows, Cols>::assign(dest, src, assign_fn);
  }

public:
  /** assign this expression to Matrix dest. */
  template<class Dest, class Assign>
  void assign_to(Dest& dest, const Assign& assign_fn) const {
    /* here is a way for caching, since each complex 'Node'
       is of type XprMatrix. */
    do_assign(dispatch<use_meta>(), dest, *this, assign_fn);
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, int l=0) const {
    os << IndentLevel(l++)
       << "XprMatrix["
       << (use_meta ? "M" :  "L") << ", O=" << ops << "]<"
       << std::endl;
    m_expr.print_xpr(os, l);
    os << IndentLevel(l)
       << "R=" << Rows << ", C=" << Cols << std::endl;
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

#include <tvmet/xpr/Identity.h>

#include <tvmet/xpr/MMProduct.h>
#include <tvmet/xpr/MMProductTransposed.h>
#include <tvmet/xpr/MMtProduct.h>
#include <tvmet/xpr/MtMProduct.h>
#include <tvmet/xpr/MVProduct.h>
#include <tvmet/xpr/MtVProduct.h>
#include <tvmet/xpr/MatrixTranspose.h>

#include <tvmet/xpr/MatrixFunctions.h>
#include <tvmet/xpr/MatrixBinaryFunctions.h>
#include <tvmet/xpr/MatrixUnaryFunctions.h>
#include <tvmet/xpr/MatrixOperators.h>
#include <tvmet/xpr/Eval.h>

#endif // TVMET_XPR_MATRIX_H

// Local Variables:
// mode:C++
// End:
