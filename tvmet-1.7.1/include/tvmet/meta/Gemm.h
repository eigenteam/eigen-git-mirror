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
 * $Id: Gemm.h,v 1.11 2004/06/17 15:53:12 opetzold Exp $
 */

#ifndef TVMET_META_GEMM_H
#define TVMET_META_GEMM_H

#include <tvmet/xpr/Null.h>

namespace tvmet {

namespace meta {


/**
 * \class gemm Gemm.h "tvmet/meta/Gemm.h"
 * \brief Meta class for matrix-matrix operations, like product
 *        using formula
 *        \f[
 *        M_1\,M_2
 *        \f]
 * \note The rows of matrix 2 have to be equal to cols of matrix 1.
 */
template<std::size_t Rows1, std::size_t Cols1,
	 std::size_t Cols2,
	 std::size_t K>
class gemm
{
  gemm();
  gemm(const gemm&);
  gemm& operator=(const gemm&);

private:
  enum {
    doIt = (K != Cols1 - 1) 		/**< recursive counter */
  };

public:
  template<class E1, class E2>
  static inline
  typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
  >::value_type
  prod(const E1& lhs, const E2& rhs, std::size_t i, std::size_t j) {
    return lhs(i, K) * rhs(K, j)
      + gemm<Rows1 * doIt, Cols1 * doIt,
             Cols2 * doIt,
             (K+1) * doIt>::prod(lhs, rhs, i, j);
  }
};


/**
 * \class gemm<0,0,0,0> Gemm.h "tvmet/meta/Gemm.h"
 * \brief gemm Specialized for recursion.
 */
template<>
class gemm<0,0,0,0>
{
  gemm();
  gemm(const gemm&);
  gemm& operator=(const gemm&);

public:
  template<class E1, class E2>
  static inline
  XprNull prod(const E1&, const E2&, std::size_t, std::size_t) {
    return XprNull();
  }
};


} // namespace meta

} // namespace tvmet

#endif /* TVMET_META_GEMM_H */

// Local Variables:
// mode:C++
// End:
