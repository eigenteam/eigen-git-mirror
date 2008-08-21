// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#include "main.h"

void test_ioformat()
{
  std::string sep = "\n\n----------------------------------------\n\n";
  Matrix4f m1;
  m1 << 0, 1.111111, 2, 3.33333, 4, 5, 6, 7, 8.888888, 9, 10, 11, 12, 13, 14, 15;

  IoFormat CommaInitFmt(4, Raw, ", ", ", ", "", "", " << ", ";");
  IoFormat CleanFmt(4, AlignCols, ", ", "\n", "[", "]");
  IoFormat OctaveFmt(4, AlignCols, ", ", ";\n", "", "", "[", "]");
  IoFormat HeavyFmt(4, AlignCols, ", ", ";\n", "[", "]", "[", "]");
  
  
  std::cout << m1 << sep;
  std::cout << m1.format(CommaInitFmt) << sep;
  std::cout << m1.format(CleanFmt) << sep;
  std::cout << m1.format(OctaveFmt) << sep;
  std::cout << m1.format(HeavyFmt) << sep;
}
