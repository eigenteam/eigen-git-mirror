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
 * $Id: Gemtv.h,v 1.4 2004/06/17 15:53:12 opetzold Exp $
 */

#ifndef TVMET_META_GEMTV_H
#define TVMET_META_GEMTV_H

#include <tvmet/xpr/Null.h>

namespace tvmet {

namespace meta {


/**
 * \class gemtv Gemtv.h "tvmet/meta/Gemtv.h"
 * \brief Meta class for matrix-transpose-vector operations.
 *        using formula
 *        \f[
 *        M^T\,v
 *        \f]
 */
template<int Rows, int Cols,
	 int I>
class gemtv
{
  gemtv();
  gemtv(const gemtv&);
  gemtv& operator=(const gemtv&);

private:
  enum {
    doIt = I < (Rows-1)  		/**< recursive counter */
  };

public:
  /** Meta template for %Matrix lhs %Vector rhs product. */
  template<class E1, class E2>
  static inline
  typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
  >::value_type
  prod(const E1& lhs, const E2& rhs, int j) {
    return lhs(I, j) * rhs(I)
      + gemtv<Rows * doIt, Cols * doIt,
              (I+1)* doIt>::prod(lhs, rhs, j);
  }
};


/**
 * \class gemtv<0,0,0> Gemtv.h "tvmet/meta/Gemtv.h"
 * \brief gemtv Specialized for recursion
 */
template<>
class gemtv<0,0,0>
{
  gemtv();
  gemtv(const gemtv&);
  gemtv& operator=(const gemtv&);

public:
  template<class E1, class E2>
  static inline
  XprNull prod(const E1&, const E2&, int) {
    return XprNull();
  }
};


} // namespace meta

} // namespace tvmet

#endif /* TVMET_META_GEMTV_H */

// Local Variables:
// mode:C++
// End:
