// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#include "main.h"

namespace Eigen {

template<typename Scalar> void smallVectors()
{
  typedef Matrix<Scalar, 1, 2> V2;
  typedef Matrix<Scalar, 3, 1> V3;
  typedef Matrix<Scalar, 1, 4> V4;
  Scalar x1 = ei_random<Scalar>(),
         x2 = ei_random<Scalar>(),
         x3 = ei_random<Scalar>(),
         x4 = ei_random<Scalar>();
  V2 v2(x1, x2);
  V3 v3(x1, x2, x3);
  V4 v4(x1, x2, x3, x4);
  VERIFY_IS_APPROX(x1, v2.x());
  VERIFY_IS_APPROX(x1, v3.x());
  VERIFY_IS_APPROX(x1, v4.x());
  VERIFY_IS_APPROX(x2, v2.y());
  VERIFY_IS_APPROX(x2, v3.y());
  VERIFY_IS_APPROX(x2, v4.y());
  VERIFY_IS_APPROX(x3, v3.z());
  VERIFY_IS_APPROX(x3, v4.z());
  VERIFY_IS_APPROX(x4, v4.w());
}

void EigenTest::testSmallVectors()
{
  for(int i = 0; i < m_repeat; i++) {
    smallVectors<int>();
    smallVectors<float>();
    smallVectors<double>();
  }
}

} // namespace Eigen
