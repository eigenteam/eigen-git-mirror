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
 * $Id: MatrixDiag.h,v 1.13 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_XPR_MATRIX_DIAG_H
#define TVMET_XPR_MATRIX_DIAG_H

namespace tvmet {


/**
 * \class XprMatrixDiag MatrixDiag.h "tvmet/xpr/MatrixDiag.h"
 * \brief Expression on matrix used for access on the diagonal vector.
 */
template<class E, std::size_t Sz>
class XprMatrixDiag
  : public TvmetBase< XprMatrixDiag<E, Sz> >
{
  XprMatrixDiag();
  XprMatrixDiag& operator=(const XprMatrixDiag<E, Sz>&);

public:
  typedef typename E::value_type			value_type;

public:
  /** Complexity counter. */
  enum {
    ops_expr  = E::ops,
    ops       = ops_expr/Sz
  };

public:
  /** Constructor. */
  explicit XprMatrixDiag(const E& e)
    : m_expr(e)
  { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprMatrixDiag(const XprMatrixDiag& e)
    : m_expr(e.m_expr)
  { }
#endif

  /** index operator for arrays/matrizes */
  value_type operator()(std::size_t i) const {
    TVMET_RT_CONDITION(i < Sz, "XprMatrixDiag Bounce Violation")
    return m_expr(i, i);
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprMatrixDiag[O=" << ops << ", (O=" << ops_expr << ")]<"
       << std::endl;
    m_expr.print_xpr(os, l);
    os << IndentLevel(l)
       << "Sz=" << Sz << std::endl
       << IndentLevel(--l) << ">"
       << ((l != 0) ? "," : "") << std::endl;
  }

private:
  const E						m_expr;
};


} // namespace tvmet

#endif // TVMET_XPR_MATRIX_DIAG_H

// Local Variables:
// mode:C++
// End:
