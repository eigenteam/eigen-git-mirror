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
 * $Id: Eval.h,v 1.9 2003/11/30 18:35:17 opetzold Exp $
 */

#ifndef TVMET_XPR_EVAL_H
#define TVMET_XPR_EVAL_H

namespace tvmet {


/**
 * \class XprEval Eval.h "tvmet/xpr/Eval.h"
 * \brief evaluate the expression
 *
 * Since we can't overwrite the ? operator we have to write a wrapper
 * for expression like return v1>v2 ? true : false
 */
template<class E1, class E2, class E3>
class XprEval
  : public TvmetBase< XprEval<E1, E2, E3> >
{
public:
  typedef E1 						expr1_type;
  typedef E2 						expr2_type;
  typedef E3 						expr3_type;

  typedef typename expr2_type::value_type 		value2_type;
  typedef typename expr3_type::value_type 		value3_type;

  typedef typename
  PromoteTraits<value2_type, value3_type>::value_type 	value_type;

public:
  /** Complexity Counter */
  enum {
    ops_expr1 = E1::ops,
    ops_expr2 = E2::ops,
    ops_expr3 = E3::ops,
    ops = ops_expr1	// only (e1 op e2) are evaluated
  };

private:
  XprEval();
  XprEval& operator=(const XprEval<expr1_type, expr2_type, expr3_type>&);

public:
  /** Constructor */
  explicit XprEval(const expr1_type& e1, const expr2_type& e2, const expr3_type& e3)
    : m_expr1(e1), m_expr2(e2), m_expr3(e3)
  { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprEval(const XprEval& rhs)
    : m_expr1(rhs.m_expr1), m_expr2(rhs.m_expr2), m_expr3(rhs.m_expr3)
  { }
#endif

public: //access
  /** index operator for vectors. */
  value_type operator()(std::size_t i) const {
    return m_expr1(i) ? m_expr2(i) : m_expr3(i);
  }

  /** index operator for matrizes. */
  value_type operator()(std::size_t i, std::size_t j) const {
    return m_expr1(i, j) ? m_expr2(i, j) : m_expr3(i, j);
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprEval[" << ops << ", ("
       << ops_expr1 << ", " << ops_expr2 << ", " << ops_expr3 << ")]<"
       << std::endl;
    m_expr1.print_xpr(os, l);
    m_expr2.print_xpr(os, l);
    m_expr3.print_xpr(os, l);
    os << IndentLevel(--l)
       << ">," << std::endl;
  }

private:
  const expr1_type					m_expr1;
  const expr2_type					m_expr2;
  const expr3_type					m_expr3;
};


} // namespace tvmet

#endif // TVMET_XPR_EVAL_H

// Local Variables:
// mode:C++
// End:
