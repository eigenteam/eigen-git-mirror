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
 * $Id: MMtProduct.h,v 1.16 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_XPR_MMTPRODUCT_H
#define TVMET_XPR_MMTPRODUCT_H

#include <tvmet/meta/Gemmt.h>
#include <tvmet/loop/Gemmt.h>

namespace tvmet {


/**
 * \class XprMMtProduct MMtProduct.h "tvmet/xpr/MMtProduct.h"
 * \brief Expression for matrix-matrix product.
 *        Using formula:
 *        \f[
 *        M_1\,M_2^T
 *        \f]
 * \note The number of cols of rhs matrix have to be equal to cols of rhs matrix.
 *       The result is a (Rows1 x Rows2) matrix.
 */
template<class E1, int Rows1, int Cols1,
	 class E2, int Cols2>
class XprMMtProduct
  : public TvmetBase< XprMMtProduct<E1, Rows1, Cols1, E2, Cols2> >
{
private:
  XprMMtProduct();
  XprMMtProduct& operator=(const XprMMtProduct&);

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
    Rows2 = Cols1,
    M = Rows1 * Cols1 * Rows1,
    N = Rows1 * (Cols1 - 1) * Rows2,
    ops_plus  = M * NumericTraits<value_type>::ops_plus,
    ops_muls  = N * NumericTraits<value_type>::ops_muls,
    ops       = ops_plus + ops_muls,
    use_meta  = Rows1*Rows2 < TVMET_COMPLEXITY_MM_TRIGGER ? true : false
  };

public:
  /** Constructor. */
  explicit XprMMtProduct(const E1& lhs, const E2& rhs)
    : m_lhs(lhs), m_rhs(rhs)
  { }

  /** Copy Constructor. Not explicit! */
#if defined(TVMET_OPTIMIZE_XPR_MANUAL_CCTOR)
  XprMMtProduct(const XprMMtProduct& e)
    : m_lhs(e.m_lhs), m_rhs(e.m_rhs)
  { }
#endif

private:
  /** Wrapper for meta gemm. */
  static inline
  value_type do_gemmt(dispatch<true>, const E1& lhs, const E2& rhs, int i, int j) {
    return meta::gemmt<Rows1, Cols1,
                       Cols2,
                       0>::prod(lhs, rhs, i, j);
  }

  /** Wrapper for loop gemm. */
  static inline
  value_type do_gemmt(dispatch<false>, const E1& lhs, const E2& rhs, int i, int j) {
    return loop::gemmt<Rows1, Cols1, Cols1>::prod(lhs, rhs, i, j);
  }

public:
  /** index operator for arrays/matrices */
  value_type operator()(int i, int j) const {
    TVMET_RT_CONDITION((i < Rows1) && (j < Rows2), "XprMMtProduct Bounce Violation")
    return do_gemmt(dispatch<use_meta>(), m_lhs, m_rhs, i, j);
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, int l=0) const {
    os << IndentLevel(l++)
       << "XprMMtProduct["
       << (use_meta ? "M" :  "L") << ", O=" << ops
       << ", (O1=" << ops_lhs << ", O2=" << ops_rhs << ")]<"
       << std::endl;
    m_lhs.print_xpr(os, l);
    os << IndentLevel(l)
       << "R1=" << Rows1 << ", C1=" << Cols1 << ",\n";
    m_rhs.print_xpr(os, l);
    os << IndentLevel(l)
       << "C2=" << Cols2 << ",\n"
       << "\n"
       << IndentLevel(--l)
       << ">," << std::endl;
  }

private:
  const E1		 				m_lhs;
  const E2 						m_rhs;
};


} // namespace tvmet

#endif // TVMET_XPR_MMTPRODUCT_H

// Local Variables:
// mode:C++
// End:
