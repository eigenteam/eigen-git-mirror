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
 * $Id: MatrixRow.h,v 1.14 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_XPR_MATRIX_ROW_H
#define TVMET_XPR_MATRIX_ROW_H

namespace tvmet {


/**
 * \class XprMatrixRow MatrixRow.h "tvmet/xpr/MatrixRow.h"
 * \brief Expression on matrix used for access on the row vector.
 */
template<class E, std::size_t Rows, std::size_t Cols>
class XprMatrixRow
  : public TvmetBase< XprMatrixRow<E, Rows, Cols> >
{
  XprMatrixRow();
  XprMatrixRow& operator=(const XprMatrixRow&);

public:
  typedef typename E::value_type			value_type;

public:
  /** Complexity counter. */
  enum {
    ops_expr  = E::ops,
    ops       = ops_expr/Rows	// equal Col accesses
  };

public:
  /** Constructor. */
  explicit XprMatrixRow(const E& e, std::size_t no)
    : m_expr(e), m_row(no)
  {
    TVMET_RT_CONDITION(no < Rows, "XprMatrixRow Bounce Violation")
  }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprMatrixRow(const XprMatrixRow& rhs)
    : m_expr(rhs.m_expr), m_row(rhs.m_row)
  { }
#endif

  value_type operator()(std::size_t j) const {
    TVMET_RT_CONDITION(j < Cols, "XprMatrixRow Bounce Violation")
    return m_expr(m_row, j);
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprMatrixRow[O=" << ops << ", (O=" << ops_expr << ")]<"
       << std::endl;
    m_expr.print_xpr(os, l);
    os << IndentLevel(l)
       << "R=" << Rows << ", C=" << Cols << std::endl
       << IndentLevel(--l) << ">"
       << ((l != 0) ? "," : "") << std::endl;
  }

private:
  const E		 				m_expr;
  const std::size_t					m_row;
};


}

#endif // TVMET_XPR_MATRIX_ROW_H

// Local Variables:
// mode:C++
// End:
