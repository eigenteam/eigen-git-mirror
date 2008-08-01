// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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
#include <Eigen/QR>

template<typename MatrixType> void eigensolver(const MatrixType& m)
{
  /* this test covers the following files:
     EigenSolver.h, SelfAdjointEigenSolver.h (and indirectly: Tridiagonalization.h)
  */
  int rows = m.rows();
  int cols = m.cols();

  typedef typename std::complex<typename NumTraits<typename MatrixType::Scalar>::Real> Complex;

  MatrixType a = MatrixType::Random(rows,cols);
  MatrixType symmA =  a.adjoint() * a;

  SelfAdjointEigenSolver<MatrixType> eiSymm(symmA);
  VERIFY_IS_APPROX(symmA * eiSymm.eigenvectors(), (eiSymm.eigenvectors() * eiSymm.eigenvalues().asDiagonal().eval()));

  // generalized eigen problem Ax = lBx
  MatrixType b = MatrixType::Random(rows,cols);
  MatrixType symmB =  b.adjoint() * b;
  eiSymm.compute(symmA,symmB);
  VERIFY_IS_APPROX(symmA * eiSymm.eigenvectors(), symmB * (eiSymm.eigenvectors() * eiSymm.eigenvalues().asDiagonal().eval()));

//   EigenSolver<MatrixType> eiNotSymmButSymm(covMat);
//   VERIFY_IS_APPROX((covMat.template cast<Complex>()) * (eiNotSymmButSymm.eigenvectors().template cast<Complex>()),
//     (eiNotSymmButSymm.eigenvectors().template cast<Complex>()) * (eiNotSymmButSymm.eigenvalues().asDiagonal()));

//   EigenSolver<MatrixType> eiNotSymm(a);
//   VERIFY_IS_APPROX(a.template cast<Complex>() * eiNotSymm.eigenvectors().template cast<Complex>(),
//     eiNotSymm.eigenvectors().template cast<Complex>() * eiNotSymm.eigenvalues().asDiagonal());

}

void test_eigensolver()
{
  for(int i = 0; i < 1; i++) {
    // very important to test a 3x3 matrix since we provide a special path for it
    CALL_SUBTEST( eigensolver(Matrix3f()) );
    CALL_SUBTEST( eigensolver(Matrix4d()) );
    CALL_SUBTEST( eigensolver(MatrixXf(7,7)) );
    CALL_SUBTEST( eigensolver(MatrixXcd(6,6)) );
    CALL_SUBTEST( eigensolver(MatrixXcf(3,3)) );
  }
}
