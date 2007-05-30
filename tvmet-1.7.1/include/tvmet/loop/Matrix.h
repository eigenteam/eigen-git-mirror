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
 * $Id: Matrix.h,v 1.7 2004/06/27 20:32:55 opetzold Exp $
 */

#ifndef TVMET_LOOP_MATRIX_H
#define TVMET_LOOP_MATRIX_H

namespace tvmet {

namespace loop {


/**
 * \class Matrix Matrix.h "tvmet/loop/Matrix.h"
 * \brief Loop %Matrix class using expression and loop templates.
 */
template<std::size_t Rows, std::size_t Cols>
class Matrix
{
  Matrix(const Matrix&);
  Matrix& operator=(const Matrix&);

public:
  Matrix() { }

public:
  /** assign an expression on columns on given row using the functional fn. */
  template<class E1, class E2, class Assign>
  static inline
  void assign(E1& lhs, const E2& rhs, const Assign& assign_fn) {
    for(std::size_t i = 0; i != Rows; ++i)
      for(std::size_t j = 0; j != Cols; ++j)
	assign_fn.apply_on(lhs(i, j), rhs(i, j));
  }
};


} // namespace loop

} // namespace tvmet

#endif /* TVMET_LOOP_MATRIX_H */

// Local Variables:
// mode:C++
// End:
