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
 * $Id: BinOperator.h,v 1.15 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_XPR_BINOPERATOR_H
#define TVMET_XPR_BINOPERATOR_H

#include <tvmet/TypePromotion.h>

namespace tvmet {


/**
 * \class XprBinOp BinOperator.h "tvmet/xpr/BinOperator.h"
 * \brief Binary operators working on two sub expressions.
 *
 * On acessing using the index operator() the binary operation will be
 * evaluated at compile time.
 */
template<class BinOp, class E1, class E2>
class XprBinOp
  : public TvmetBase< XprBinOp<BinOp, E1, E2> >
{
  XprBinOp();
  XprBinOp& operator=(const XprBinOp&);

public:
  typedef typename BinOp::value_type			value_type;

public:
  /** Complexity counter. */
  enum {
    ops_lhs   = E1::ops,
    ops_rhs   = E2::ops,
    ops       = 2 * (ops_lhs + ops_rhs) // lhs op rhs
  };

public:
  /** Constructor for two expressions. */
  explicit XprBinOp(const E1& lhs, const E2& rhs)
    : m_lhs(lhs), m_rhs(rhs)
  { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprBinOp(const XprBinOp& e)
    : m_lhs(e.m_lhs), m_rhs(e.m_rhs)
  { }
#endif

  /** Index operator, evaluates the expression inside. */
  value_type operator()(std::size_t i) const {
    return BinOp::apply_on(m_lhs(i), m_rhs(i));
  }

  /** Index operator for arrays/matrices */
  value_type operator()(std::size_t i, std::size_t j) const {
    return BinOp::apply_on(m_lhs(i, j), m_rhs(i, j));
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprBinOp[O="<< ops << ", (O1=" << ops_lhs << ", O2=" << ops_rhs << ")]<"
       << std::endl;
    BinOp::print_xpr(os, l);
    m_lhs.print_xpr(os, l);
    m_rhs.print_xpr(os, l);
    os << IndentLevel(--l)
       << ">," << std::endl;
  }

private:
  const E1						m_lhs;
  const E2						m_rhs;
};


} // namespace tvmet

#endif // TVMET_XPR_BINOPERATOR_H

// Local Variables:
// mode:C++
// End:
