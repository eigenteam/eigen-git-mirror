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
 * $Id: Gemv.h,v 1.3 2004/06/16 09:30:07 opetzold Exp $
 */

#ifndef TVMET_LOOP_GEMV_H
#define TVMET_LOOP_GEMV_H

namespace tvmet {

namespace loop {


/**
 * \class gemv Gemv.h "tvmet/loop/Gemv.h"
 * \brief class for matrix-vector product using loop unrolling.
 *        using formula
 *        \f[
 *        M\,v
 *        \f]
 * \par Example:
 * \code
 * template<class T, int Rows, int Cols>
 * inline
 * void
 * prod(const Matrix<T, Rows, Cols>& lhs, const Vector<T, Cols>& rhs,
 * 	Vector<T, Rows>& dest)
 * {
 *   for (int i = 0; i != Rows; ++i) {
 *     dest(i) = tvmet::loop::gemv<Rows, Cols>().prod(lhs, rhs, i);
 *   }
 * }
 * \endcode
 */
template<int Rows, int Cols>
class gemv
{
  gemv(const gemv&);
  gemv& operator=(const gemv&);

private:
  enum {
    count 	= Cols,
    N 		= (count+7)/8
  };

public:
  gemv() { }

public:
  template<class E1, class E2>
  static inline
  typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
    >::value_type
  prod(const E1& lhs, const E2& rhs, int i) {
    typename PromoteTraits<
      typename E1::value_type,
      typename E2::value_type
    >::value_type  				sum(0);
    int 				j(0);
    int 				n(N);

    // Duff's device
    switch(count % 8) {
    case 0: do { sum += lhs(i, j) * rhs(j); ++j;
    case 7:      sum += lhs(i, j) * rhs(j); ++j;
    case 6:      sum += lhs(i, j) * rhs(j); ++j;
    case 5:      sum += lhs(i, j) * rhs(j); ++j;
    case 4:      sum += lhs(i, j) * rhs(j); ++j;
    case 3:      sum += lhs(i, j) * rhs(j); ++j;
    case 2:      sum += lhs(i, j) * rhs(j); ++j;
    case 1:      sum += lhs(i, j) * rhs(j); ++j;
            } while(--n != 0);
    }

    return sum;
  }
};


} // namespace loop

} // namespace tvmet

#endif /* TVMET_LOOP_GEMV_H */

// Local Variables:
// mode:C++
// End:
