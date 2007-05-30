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
 * $Id: Gemmt.h,v 1.9 2004/06/17 15:53:12 opetzold Exp $
 */

#ifndef TVMET_META_GEMMT_H
#define TVMET_META_GEMMT_H

#include <tvmet/xpr/Null.h>

namespace tvmet {

namespace meta {


/**
 * \class gemmt Gemmt.h "tvmet/meta/Gemmt.h"
 * \brief Meta class for product matrix-transpose(matrix) operations.
 *        using formula
 *        \f[
 *        M_1\,M_2^{T}
 *        \f]
 * \note The rows of matrix 2 have to be equal to cols of matrix 1. The result
 *       is a rows1 * cols2 matrix.
 */
template<std::size_t Rows1, std::size_t Cols1,
	 std::size_t Cols2,
	 std::size_t K>
class gemmt
{
  gemmt();
  gemmt(const gemmt&);
  gemmt& operator=(const gemmt&);

private:
  enum {
    doIt = (K != Cols2 - 1) 		/**< recursive counter */
  };

public:
  template<class E1, class E2>
  static inline
  typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
  >::value_type
  prod(const E1& lhs, const E2& rhs, std::size_t i, std::size_t j) {
    return lhs(i, K) * rhs(j, K)
      + gemmt<Rows1 * doIt, Cols1 * doIt,
              Cols2 * doIt,
              (K+1) * doIt>::prod(lhs, rhs, i, j);
  }
};


/**
 * \class gemmt<0,0,0,0> Gemmt.h "tvmet/meta/Gemmt.h"
 * \brief gemmt Specialized for recursion.
 */
template<>
class gemmt<0,0,0,0>
{
  gemmt();
  gemmt(const gemmt&);
  gemmt& operator=(const gemmt&);

public:
  template<class E1, class E2>
  static inline
  XprNull prod(const E1&, const E2&, std::size_t, std::size_t) {
    return XprNull();
  }
};


} // namespace meta

} // namespace tvmet

#endif /* TVMET_META_GEMMT_H */

// Local Variables:
// mode:C++
// End:
