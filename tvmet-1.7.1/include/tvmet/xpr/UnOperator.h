/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: UnOperator.h,v 1.13 2003/11/30 18:35:17 opetzold Exp $
 */

#ifndef TVMET_XPR_UNOPERATOR_H
#define TVMET_XPR_UNOPERATOR_H

namespace tvmet {


/**
 * \class XprUnOp UnOperator.h "tvmet/xpr/UnOperator.h"
 * \brief Unary operator working on one subexpression.
 *
 * Using the access operator() the unary operation will be evaluated.
 */
template<class UnOp, class E>
class XprUnOp
  : public TvmetBase< XprUnOp<UnOp, E> >
{
  XprUnOp();
  XprUnOp& operator=(const XprUnOp&);

public:
  typedef typename UnOp::value_type				value_type;

public:
  /** Complexity counter. */
  enum {
    ops_expr  = E::ops,
    ops       = 1 * ops_expr
  };

public:
  /** Constructor for an expressions. */
  explicit XprUnOp(const E& e)
    : m_expr(e)
  { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprUnOp(const XprUnOp& e)
    : m_expr(e.m_expr)
  { }
#endif

  /** Index operator, evaluates the expression inside. */
  value_type operator()(int i) const {
    return UnOp::apply_on(m_expr(i));
  }

  /** index operator for arrays/matrices. */
  value_type operator()(int i, int j) const {
    return UnOp::apply_on(m_expr(i, j));
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, int l=0) const {
    os << IndentLevel(l++)
       << "XprUnOp[O="<< ops << ", (O=" << ops_expr << ")]<"
       << std::endl;
    UnOp::print_xpr(os, l);
    m_expr.print_xpr(os, l);
    os << IndentLevel(--l)
       << ">," << std::endl;
  }

private:
  const E							m_expr;
};


} // namespace tvmet

#endif // TVMET_XPR_UNOPERATOR_H

// Local Variables:
// mode:C++
// End:
