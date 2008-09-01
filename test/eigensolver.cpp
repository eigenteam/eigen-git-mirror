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

#ifdef HAS_GSL
#include "gsl_helper.h"
#endif

template<typename MatrixType> void eigensolver(const MatrixType& m)
{
  /* this test covers the following files:
     EigenSolver.h, SelfAdjointEigenSolver.h (and indirectly: Tridiagonalization.h)
  */
  int rows = m.rows();
  int cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;
  typedef Matrix<RealScalar, MatrixType::RowsAtCompileTime, 1> RealVectorType;
  typedef typename std::complex<typename NumTraits<typename MatrixType::Scalar>::Real> Complex;

  RealScalar largerEps = 10*test_precision<RealScalar>();

  MatrixType a = MatrixType::Random(rows,cols);
  MatrixType a1 = MatrixType::Random(rows,cols);
  MatrixType symmA =  a.adjoint() * a + a1.adjoint() * a1;

  MatrixType b = MatrixType::Random(rows,cols);
  MatrixType b1 = MatrixType::Random(rows,cols);
  MatrixType symmB = b.adjoint() * b + b1.adjoint() * b1;

  SelfAdjointEigenSolver<MatrixType> eiSymm(symmA);
  // generalized eigen pb
  SelfAdjointEigenSolver<MatrixType> eiSymmGen(symmA, symmB);

  #ifdef HAS_GSL
  if (ei_is_same_type<RealScalar,double>::ret)
  {
    typedef GslTraits<Scalar> Gsl;
    typename Gsl::Matrix gEvec=0, gSymmA=0, gSymmB=0;
    typename GslTraits<RealScalar>::Vector gEval=0;
    RealVectorType _eval;
    MatrixType _evec;
    convert<MatrixType>(symmA, gSymmA);
    convert<MatrixType>(symmB, gSymmB);
    convert<MatrixType>(symmA, gEvec);
    gEval = GslTraits<RealScalar>::createVector(rows);
    
    Gsl::eigen_symm(gSymmA, gEval, gEvec);
    convert(gEval, _eval);
    convert(gEvec, _evec);
    
    // test gsl itself !
    VERIFY((symmA * _evec).isApprox(_evec * _eval.asDiagonal().eval(), largerEps));

    // compare with eigen
    VERIFY_IS_APPROX(_eval, eiSymm.eigenvalues());
    VERIFY_IS_APPROX(_evec.cwise().abs(), eiSymm.eigenvectors().cwise().abs());

    // generalized pb
    Gsl::eigen_symm_gen(gSymmA, gSymmB, gEval, gEvec);
    convert(gEval, _eval);
    convert(gEvec, _evec);
    // test GSL itself:
    VERIFY((symmA * _evec).isApprox(symmB * (_evec * _eval.asDiagonal().eval()), largerEps));

    // compare with eigen
//     std::cerr << _eval.transpose() << "\n" << eiSymmGen.eigenvalues().transpose() << "\n\n";
//     std::cerr << _evec.format(6) << "\n\n" << eiSymmGen.eigenvectors().format(6) << "\n\n\n";
    VERIFY_IS_APPROX(_eval, eiSymmGen.eigenvalues());
    VERIFY_IS_APPROX(_evec.cwise().abs(), eiSymmGen.eigenvectors().cwise().abs());

    Gsl::free(gSymmA);
    Gsl::free(gSymmB);
    GslTraits<RealScalar>::free(gEval);
    Gsl::free(gEvec);
  }
  #endif

  VERIFY((symmA * eiSymm.eigenvectors()).isApprox(
          eiSymm.eigenvectors() * eiSymm.eigenvalues().asDiagonal().eval(), largerEps));

  // generalized eigen problem Ax = lBx
  VERIFY((symmA * eiSymmGen.eigenvectors()).isApprox(
          symmB * (eiSymmGen.eigenvectors() * eiSymmGen.eigenvalues().asDiagonal().eval()), largerEps));

//   EigenSolver<MatrixType> eiNotSymmButSymm(covMat);
//   VERIFY_IS_APPROX((covMat.template cast<Complex>()) * (eiNotSymmButSymm.eigenvectors().template cast<Complex>()),
//     (eiNotSymmButSymm.eigenvectors().template cast<Complex>()) * (eiNotSymmButSymm.eigenvalues().asDiagonal()));

//   EigenSolver<MatrixType> eiNotSymm(a);
//   VERIFY_IS_APPROX(a.template cast<Complex>() * eiNotSymm.eigenvectors().template cast<Complex>(),
//     eiNotSymm.eigenvectors().template cast<Complex>() * eiNotSymm.eigenvalues().asDiagonal());

}

void test_eigensolver()
{
  for(int i = 0; i < g_repeat; i++) {
    // very important to test a 3x3 matrix since we provide a special path for it
    CALL_SUBTEST( eigensolver(Matrix3f()) );
    CALL_SUBTEST( eigensolver(Matrix4d()) );
    CALL_SUBTEST( eigensolver(MatrixXf(7,7)) );
    CALL_SUBTEST( eigensolver(MatrixXcd(5,5)) );
    CALL_SUBTEST( eigensolver(MatrixXd(19,19)) );
  }
}
