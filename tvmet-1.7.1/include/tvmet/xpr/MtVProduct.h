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
 * $Id: MtVProduct.h,v 1.10 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_XPR_MTVPRODUCT_H
#define TVMET_XPR_MTVPRODUCT_H

#include <cassert>

#include <tvmet/meta/Gemtv.h>
#include <tvmet/loop/Gemtv.h>

namespace tvmet {


/**
 * \class XprMtVProduct MtVProduct.h "tvmet/xpr/MtVProduct.h"
 * \brief Expression for matrix-transposed vector product
 *        using formula
 *        \f[
 *        M^T\,v
 *        \f]
 */
template<class E1, int Rows, int Cols,
	 class E2>
class XprMtVProduct
  : public TvmetBase< XprMtVProduct<E1, Rows, Cols, E2> >
{
  XprMtVProduct();
  XprMtVProduct& operator=(const XprMtVProduct&);

public:
  typedef typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
  >::value_type							value_type;

public:
  /** Complexity counter. */
  enum {
    ops_lhs   = E1::ops,
    ops_rhs   = E2::ops,
    M         = Cols * Rows,
    N         = Cols * (Rows - 1),
    ops_plus  = M * Traits<value_type>::ops_plus,
    ops_muls  = N * Traits<value_type>::ops_muls,
    ops       = ops_plus + ops_muls,
    use_meta  = Rows*Cols < TVMET_COMPLEXITY_MV_TRIGGER ? true : false
  };

public:
  /** Constructor. */
  explicit XprMtVProduct(const E1& lhs, const E2& rhs)
    : m_lhs(lhs), m_rhs(rhs)
  { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprMtVProduct(const XprMtVProduct& e)
    : m_lhs(e.m_lhs), m_rhs(e.m_rhs)
  { }
#endif

private:
  /** Wrapper for meta gemm. */
  static inline
  value_type do_gemtv(dispatch<true>, const E1& lhs, const E2& rhs, int i) {
    return meta::gemtv<Rows, Cols, 0>::prod(lhs, rhs, i);
  }

  /** Wrapper for loop gemm. */
  static inline
  value_type do_gemtv(dispatch<false>, const E1& lhs, const E2& rhs, int i) {
    return loop::gemtv<Rows, Cols>::prod(lhs, rhs, i);
  }

public:
  /** index operator, returns the expression by index. This is the vector
      style since a matrix*vector gives a vector. */
  value_type operator()(int j) const {
    assert(j < Cols);
    return do_gemtv(dispatch<use_meta>(), m_lhs, m_rhs, j);
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, int l=0) const {
    os << IndentLevel(l++)
       << "XprMtVProduct[O=" << ops << ", (O1=" << ops_lhs << ", O2=" << ops_rhs << ")]<"
       << std::endl;
    m_lhs.print_xpr(os, l);
    os << IndentLevel(l)
       << "R=" << Rows << ", C=" << Cols << ",\n";
    m_rhs.print_xpr(os, l);
    os << IndentLevel(--l)
       << ">," << std::endl;
  }

private:
  const E1							m_lhs;
  const E2							m_rhs;
};


} // namespace tvmet

#endif // TVMET_XPR_MTVPRODUCT_H

// Local Variables:
// mode:C++
// End:
