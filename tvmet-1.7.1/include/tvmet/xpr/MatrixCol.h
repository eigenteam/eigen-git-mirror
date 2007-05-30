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
 * $Id: MatrixCol.h,v 1.15 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_XPR_MATRIX_COL_H
#define TVMET_XPR_MATRIX_COL_H

namespace tvmet {


/**
 * \class XprMatrixCol MatrixCol.h "tvmet/xpr/MatrixCol.h"
 * \brief Expression on matrix used for access on the column vector.
 */
template<class E, std::size_t Rows, std::size_t Cols>
class XprMatrixCol
  : public TvmetBase< XprMatrixCol<E, Rows, Cols> >
{
  XprMatrixCol();
  XprMatrixCol& operator=(const XprMatrixCol&);

public:
  typedef typename E::value_type			value_type;

public:
  /** Complexity counter. */
  enum {
    ops_expr  = E::ops,
    ops       = ops_expr/Cols	// equal Row accesses
  };

public:
  /** Constructor. */
  explicit XprMatrixCol(const E& e, std::size_t no)
    : m_expr(e), m_col(no)
  {
    TVMET_RT_CONDITION(no < Cols, "XprMatrixCol Bounce Violation")
  }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprMatrixCol(const XprMatrixCol& e)
    : m_expr(e.m_expr), m_col(e.m_col)
  { }
#endif

  value_type operator()(std::size_t i) const {
    TVMET_RT_CONDITION(i < Rows, "XprMatrixCol Bounce Violation")
    return m_expr(i, m_col);
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprMatrixCol[O=" << ops << ", (O=" << ops_expr << ")]<"
       << std::endl;
    m_expr.print_xpr(os, l);
    os << IndentLevel(l)
       << "R=" << Rows << ", C=" << Cols << std::endl
       << IndentLevel(--l) << ">"
       << ((l != 0) ? "," : "") << std::endl;
  }

private:
  const E						m_expr;
  const std::size_t					m_col;
};


}

#endif // TVMET_XPR_MATRIX_COL_H

// Local Variables:
// mode:C++
// End:
