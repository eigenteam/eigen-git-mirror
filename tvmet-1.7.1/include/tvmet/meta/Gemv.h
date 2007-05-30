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
 * $Id: Gemv.h,v 1.9 2004/06/17 15:53:12 opetzold Exp $
 */

#ifndef TVMET_META_GEMV_H
#define TVMET_META_GEMV_H

#include <tvmet/xpr/Null.h>

namespace tvmet {

namespace meta {


/**
 * \class gemv Gemv.h "tvmet/meta/Gemv.h"
 * \brief Meta class for matrix-vector operations.
 *        using formula
 *        \f[
 *        M\,v
 *        \f]
 */
template<std::size_t Rows, std::size_t Cols,
	 std::size_t J>
class gemv
{
  gemv();
  gemv(const gemv&);
  gemv& operator=(const gemv&);

private:
  enum {
    doIt = J < (Cols-1)  		/**< recursive counter */
  };

public:
  /** Meta template for %Matrix lhs %Vector rhs product. */
  template<class E1, class E2>
  static inline
  typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
  >::value_type
  prod(const E1& lhs, const E2& rhs, std::size_t i) {
    return lhs(i, J) * rhs(J)
      + gemv<Rows * doIt, Cols * doIt,
             (J+1)* doIt>::prod(lhs, rhs, i);
  }
};


/**
 * \class gemv<0,0,0> Gemv.h "tvmet/meta/Gemv.h"
 * \brief gemv Specialized for recursion
 */
template<>
class gemv<0,0,0>
{
  gemv();
  gemv(const gemv&);
  gemv& operator=(const gemv&);

public:
  template<class E1, class E2>
  static inline
  XprNull prod(const E1&, const E2&, std::size_t) {
    return XprNull();
  }
};


} // namespace meta

} // namespace tvmet

#endif /* TVMET_META_GEMV_H */

// Local Variables:
// mode:C++
// End:
