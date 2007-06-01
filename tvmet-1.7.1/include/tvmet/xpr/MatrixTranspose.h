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
 * $Id: MatrixTranspose.h,v 1.11 2004/06/10 16:36:55 opetzold Exp $
 */

#ifndef TVMET_XPR_MATRIX_TRANSPOSE_H
#define TVMET_XPR_MATRIX_TRANSPOSE_H

namespace tvmet {


/**
 * \class XprMatrixTranspose MatrixTranspose.h "tvmet/xpr/MatrixTranspose.h"
 * \brief Expression for transpose matrix
 */
template<class E>
class XprMatrixTranspose
  : public TvmetBase< XprMatrixTranspose<E> >
{
  XprMatrixTranspose();
  XprMatrixTranspose& operator=(const XprMatrixTranspose&);

public:
  typedef typename E::value_type			value_type;

  /** Complexity counter. */
  enum {
    ops_expr  = E::ops,
    ops       = 1 * ops_expr
  };

public:
  /** Constructor. */
  explicit XprMatrixTranspose(const E& e)
    : m_expr(e)
  { }

 /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprMatrixTranspose(const XprMatrixTranspose& e)
    : m_expr(e.m_expr)
  { }
#endif

  /** index operator for arrays/matrices. This simple swap the index
      access for transpose. */
  value_type operator()(int i, int j) const { return m_expr(j, i); }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, int l=0) const {
    os << IndentLevel(l++)
       << "XprMatrixTranspose[O=" << ops << ", (O=" << ops_expr << ")]<"
       << std::endl;
    m_expr.print_xpr(os, l);
    os << IndentLevel(--l)
       << ">," << std::endl;
  }

private:
  const E						m_expr;
};


} // namespace tvmet

#endif // TVMET_XPR_MATRIX_TRANSPOSE_H

// Local Variables:
// mode:C++
// End:
