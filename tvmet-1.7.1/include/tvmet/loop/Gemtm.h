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
 * $Id: Gemtm.h,v 1.5 2004/06/16 09:30:07 opetzold Exp $
 */

#ifndef TVMET_LOOP_GEMTM_H
#define TVMET_LOOP_GEMTM_H

namespace tvmet {

namespace loop {


/**
 * \class gemtm Gemtm.h "tvmet/loop/Gemtm.h"
 * \brief class for matrix-matrix product using loop unrolling.
 *        using formula
 *        \f[
 *        M_1^{T}\,M_2
 *        \f]
 * \par Example:
 * \code
 * template<class T, int Rows1, int Cols1, int Cols2>
 * inline
 * void
 * prod(const Matrix<T, Rows1, Cols1>& lhs, const Matrix<T, Rows1, Cols2>& rhs,
 * 	Matrix<T, Cols2, Cols1>& dest)
 * {
 *   for (int i = 0; i != Cols1; ++i) {
 *     for (int j = 0; j != Cols2; ++j) {
 *       dest(i, j) = tvmet::loop::gemtm<Rows1, Cols1, Cols2>::prod(lhs, rhs, i, j);
 *     }
 *   }
 * }
 * \endcode
 * \note The number of rows of rhs matrix have to be equal rows of rhs matrix,
 *       since lhs matrix 1 is transposed.
 *       The result is a (Cols1 x Cols2) matrix.
 */
template<int Rows1, int Cols1,
	 int Cols2>
class gemtm
{
  gemtm(const gemtm&);
  gemtm& operator=(const gemtm&);

private:
  enum {
    count 	= Cols1,
    N 		= (count+7)/8
  };

public:
  gemtm() { }

public:
  template<class E1, class E2>
  static inline
  typename PromoteTraits<
    typename E1::value_type,
    typename E2::value_type
    >::value_type
  prod(const E1& lhs, const E2& rhs, int i, int j) {
    typename PromoteTraits<
      typename E1::value_type,
      typename E2::value_type
    >::value_type  				sum(0);
    int 				k(0);
    int 				n(N);

    // Duff's device
    switch(count % 8) {
    case 0: do { sum += lhs(k, i) * rhs(k, j); ++k;
    case 7:      sum += lhs(k, i) * rhs(k, j); ++k;
    case 6:      sum += lhs(k, i) * rhs(k, j); ++k;
    case 5:      sum += lhs(k, i) * rhs(k, j); ++k;
    case 4:      sum += lhs(k, i) * rhs(k, j); ++k;
    case 3:      sum += lhs(k, i) * rhs(k, j); ++k;
    case 2:      sum += lhs(k, i) * rhs(k, j); ++k;
    case 1:      sum += lhs(k, i) * rhs(k, j); ++k;
            } while(--n != 0);
    }

    return sum;
  }
};


} // namespace loop

} // namespace tvmet

#endif /* TVMET_LOOP_GEMTM_H */

// Local Variables:
// mode:C++
// End:
