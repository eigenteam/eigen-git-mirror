// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Daniel Gomez Ferro <dgomezferro@gmail.com>
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
#include <Eigen/Sparse>

template<typename Scalar> void sparse()
{
  int rows = 8, cols = 8;
  double density = std::max(8./(rows*cols), 0.01);
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  Scalar eps = 1e-6;

  SparseMatrix<Scalar> m(rows, cols);
  DenseMatrix refMat = DenseMatrix::Zero(rows, cols);

  std::vector<Vector2i> zeroCoords;
  std::vector<Vector2i> nonzeroCoords;
  m.startFill(rows*cols*density);
  for(int j=0; j<cols; j++)
  {
    for(int i=0; i<rows; i++)
    {
      Scalar v = (ei_random<Scalar>(0,1) < density) ? ei_random<Scalar>() : 0;
      if (v!=0)
      {
        m.fill(i,j) = v;
        nonzeroCoords.push_back(Vector2i(i,j));
      }
      else
      {
        zeroCoords.push_back(Vector2i(i,j));
      }
      refMat(i,j) = v;
    }
  }
  m.endFill();

  VERIFY(zeroCoords.size()>0 && "re-run the test");
  VERIFY(nonzeroCoords.size()>0 && "re-run the test");

  // test coeff and coeffRef
  for (int i=0; i<zeroCoords.size(); ++i)
  {
    VERIFY_IS_MUCH_SMALLER_THAN( m.coeff(zeroCoords[i].x(),zeroCoords[i].y()), eps );
    VERIFY_RAISES_ASSERT( m.coeffRef(zeroCoords[0].x(),zeroCoords[0].y()) = 5 );
  }
  VERIFY_IS_APPROX(m, refMat);

  m.coeffRef(nonzeroCoords[0].x(), nonzeroCoords[0].y()) = Scalar(5);
  refMat.coeffRef(nonzeroCoords[0].x(), nonzeroCoords[0].y()) = Scalar(5);

  VERIFY_IS_APPROX(m, refMat);

  // test SparseSetters
  // coherent setter
  // TODO extend the MatrixSetter
//   {
//     m.setZero();
//     VERIFY_IS_NOT_APPROX(m, refMat);
//     SparseSetter<SparseMatrix<Scalar>, FullyCoherentAccessPattern> w(m);
//     for (int i=0; i<nonzeroCoords.size(); ++i)
//     {
//       w->coeffRef(nonzeroCoords[i].x(),nonzeroCoords[i].y()) = refMat.coeff(nonzeroCoords[i].x(),nonzeroCoords[i].y());
//     }
//   }
//   VERIFY_IS_APPROX(m, refMat);
  
  // random setter
  {
    m.setZero();
    VERIFY_IS_NOT_APPROX(m, refMat);
    SparseSetter<SparseMatrix<Scalar>, RandomAccessPattern> w(m);
    std::vector<Vector2i> remaining = nonzeroCoords;
    while(!remaining.empty())
    {
      int i = ei_random<int>(0,remaining.size()-1);
      w->coeffRef(remaining[i].x(),remaining[i].y()) = refMat.coeff(remaining[i].x(),remaining[i].y());
      remaining[i] = remaining.back();
      remaining.pop_back();
    }
  }
  VERIFY_IS_APPROX(m, refMat);
}

void test_sparse()
{
  sparse<double>();
}
