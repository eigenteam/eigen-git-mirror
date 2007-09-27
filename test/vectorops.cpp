// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
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

template<typename VectorType> void vectorOps(const VectorType& v)
{
  typedef typename VectorType::Scalar Scalar;
  int size = v.size();
  
  VectorType a(size), b(size), c(b);
  Scalar s;
  a * s;
  s * a;
  a + b;
  a - b;
  (a + b) * s;
  s * (a + b);
  a + b + c;
  a = b;
  a = b + c;
  a = s * (b - c);
  a = eval(s * (b - c));
  
  a += b;
  a -= b + b;
  a *= s;
  b /= s;
  a += eval(a + a);
}

void EigenTest::testVectorOps()
{
  vectorOps(Vector2i());
  vectorOps(Vector3d());
  vectorOps(Vector4cf());
  vectorOps(VectorXf(1));
  vectorOps(VectorXi(2));
  vectorOps(VectorXd(3));
  vectorOps(VectorXcf(4));
}
