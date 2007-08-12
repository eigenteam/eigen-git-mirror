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
 * $Id: MVProduct.h,v 1.17 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_XPR_MVPRODUCT_H
#define TVMET_XPR_MVPRODUCT_H

#include <cassert>

#include <tvmet/meta/Gemv.h>
#include <tvmet/loop/Gemv.h>

namespace tvmet {


/**
 * \class XprMVProduct MVProduct.h "tvmet/xpr/MVProduct.h"
 * \brief Expression for matrix-vector product
 *        using formula
 *        \f[
 *        M\,v
 *        \f]
 */
template<class E1, int Rows, int Cols,
	 class E2>
class XprMVProduct
  : public TvmetBase< XprMVProduct<E1, Rows, Cols, E2> >
{
  XprMVProduct();
  XprMVProduct& operator=(const XprMVProduct&);

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
    M         = Rows * Cols,
    N         = Rows * (Cols - 1),
    ops_plus  = M * Traits<value_type>::ops_plus,
    ops_muls  = N * Traits<value_type>::ops_muls,
    ops       = ops_plus + ops_muls,
    use_meta  = Rows*Cols < TVMET_COMPLEXITY_MV_TRIGGER ? true : false
  };

public:
  /** Constructor. */
  explicit XprMVProduct(const E1& lhs, const E2& rhs)
    : m_lhs(lhs), m_rhs(rhs)
  { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprMVProduct(const XprMVProduct& e)
    : m_lhs(e.m_lhs), m_rhs(e.m_rhs)
  { }
#endif

private:
  /** Wrapper for meta gemm. */
  static inline
  value_type do_gemv(dispatch<true>, const E1& lhs, const E2& rhs, int j) {
    return meta::gemv<Rows, Cols,
                      0>::prod(lhs, rhs, j);
  }

  /** Wrapper for loop gemm. */
  static inline
  value_type do_gemv(dispatch<false>, const E1& lhs, const E2& rhs, int j) {
    return loop::gemv<Rows, Cols>::prod(lhs, rhs, j);
  }

public:
  /** index operator, returns the expression by index. This is the vector
      style since a matrix*vector gives a vector. */
  value_type operator()(int j) const {
    assert(j < Rows);
    return do_gemv(dispatch<use_meta>(), m_lhs, m_rhs, j);
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, int l=0) const {
    os << IndentLevel(l++)
       << "XprMVProduct["
       << (use_meta ? "M" :  "L") << ", O=" << ops
       << ", (O1=" << ops_lhs << ", O2=" << ops_rhs << ")]<"
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

#endif // TVMET_XPR_MVPRODUCT_H

// Local Variables:
// mode:C++
// End:
