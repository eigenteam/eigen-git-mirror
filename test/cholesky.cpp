// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
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

#ifndef EIGEN_NO_ASSERTION_CHECKING
#define EIGEN_NO_ASSERTION_CHECKING
#endif

static int nb_temporaries;

#define EIGEN_DEBUG_MATRIX_CTOR { if(size!=0) nb_temporaries++; }

#include "main.h"
#include <Eigen/Cholesky>
#include <Eigen/QR>

#define VERIFY_EVALUATION_COUNT(XPR,N) {\
    nb_temporaries = 0; \
    XPR; \
    if(nb_temporaries!=N) std::cerr << "nb_temporaries == " << nb_temporaries << "\n"; \
    VERIFY( (#XPR) && nb_temporaries==N ); \
  }

#ifdef HAS_GSL
#include "gsl_helper.h"
#endif

template<typename MatrixType> void cholesky(const MatrixType& m)
{
  /* this test covers the following files:
     LLT.h LDLT.h
  */
  int rows = m.rows();
  int cols = m.cols();

  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  MatrixType a0 = MatrixType::Random(rows,cols);
  VectorType vecB = VectorType::Random(rows), vecX(rows);
  MatrixType matB = MatrixType::Random(rows,cols), matX(rows,cols);
  SquareMatrixType symm =  a0 * a0.adjoint();
  // let's make sure the matrix is not singular or near singular
  for (int k=0; k<3; ++k)
  {
    MatrixType a1 = MatrixType::Random(rows,cols);
    symm += a1 * a1.adjoint();
  }

  SquareMatrixType symmUp = symm.template triangularView<Upper>();
  SquareMatrixType symmLo = symm.template triangularView<Lower>();

  // to test if really Cholesky only uses the upper triangular part, uncomment the following
  // FIXME: currently that fails !!
  //symm.template part<StrictlyLower>().setZero();

  #ifdef HAS_GSL
//   if (ei_is_same_type<RealScalar,double>::ret)
//   {
//     typedef GslTraits<Scalar> Gsl;
//     typename Gsl::Matrix gMatA=0, gSymm=0;
//     typename Gsl::Vector gVecB=0, gVecX=0;
//     convert<MatrixType>(symm, gSymm);
//     convert<MatrixType>(symm, gMatA);
//     convert<VectorType>(vecB, gVecB);
//     convert<VectorType>(vecB, gVecX);
//     Gsl::cholesky(gMatA);
//     Gsl::cholesky_solve(gMatA, gVecB, gVecX);
//     VectorType vecX(rows), _vecX, _vecB;
//     convert(gVecX, _vecX);
//     symm.llt().solve(vecB, &vecX);
//     Gsl::prod(gSymm, gVecX, gVecB);
//     convert(gVecB, _vecB);
//     // test gsl itself !
//     VERIFY_IS_APPROX(vecB, _vecB);
//     VERIFY_IS_APPROX(vecX, _vecX);
//
//     Gsl::free(gMatA);
//     Gsl::free(gSymm);
//     Gsl::free(gVecB);
//     Gsl::free(gVecX);
//   }
  #endif

  {
    LLT<SquareMatrixType,Lower> chollo(symmLo);
    VERIFY_IS_APPROX(symm, chollo.reconstructedMatrix());
    vecX = chollo.solve(vecB);
    VERIFY_IS_APPROX(symm * vecX, vecB);
    matX = chollo.solve(matB);
    VERIFY_IS_APPROX(symm * matX, matB);

    // test the upper mode
    LLT<SquareMatrixType,Upper> cholup(symmUp);
    VERIFY_IS_APPROX(symm, cholup.reconstructedMatrix());
    vecX = cholup.solve(vecB);
    VERIFY_IS_APPROX(symm * vecX, vecB);
    matX = cholup.solve(matB);
    VERIFY_IS_APPROX(symm * matX, matB);
  }

  int sign = ei_random<int>()%2 ? 1 : -1;

  if(sign == -1)
  {
    symm = -symm; // test a negative matrix
  }

  {
    LDLT<SquareMatrixType,Lower> ldltlo(symmLo);
    VERIFY_IS_APPROX(symm, ldltlo.reconstructedMatrix());
    vecX = ldltlo.solve(vecB);
    VERIFY_IS_APPROX(symm * vecX, vecB);
    matX = ldltlo.solve(matB);
    VERIFY_IS_APPROX(symm * matX, matB);

    LDLT<SquareMatrixType,Upper> ldltup(symmUp);
    VERIFY_IS_APPROX(symm, ldltup.reconstructedMatrix());
    vecX = ldltup.solve(vecB);
    VERIFY_IS_APPROX(symm * vecX, vecB);
    matX = ldltup.solve(matB);
    VERIFY_IS_APPROX(symm * matX, matB);

//     if(MatrixType::RowsAtCompileTime==Dynamic)
//     {
//       // note : each inplace permutation requires a small temporary vector (mask)
//
//       // check inplace solve
//       matX = matB;
//       VERIFY_EVALUATION_COUNT(matX = ldltlo.solve(matX), 0);
//       VERIFY_IS_APPROX(matX, ldltlo.solve(matB).eval());
//
//
//       matX = matB;
//       VERIFY_EVALUATION_COUNT(matX = ldltup.solve(matX), 0);
//       VERIFY_IS_APPROX(matX, ldltup.solve(matB).eval());
//     }
  }

}

template<typename MatrixType> void cholesky_verify_assert()
{
  MatrixType tmp;

  LLT<MatrixType> llt;
  VERIFY_RAISES_ASSERT(llt.matrixL())
  VERIFY_RAISES_ASSERT(llt.matrixU())
  VERIFY_RAISES_ASSERT(llt.solve(tmp))
  VERIFY_RAISES_ASSERT(llt.solveInPlace(&tmp))

  LDLT<MatrixType> ldlt;
  VERIFY_RAISES_ASSERT(ldlt.matrixL())
  VERIFY_RAISES_ASSERT(ldlt.permutationP())
  VERIFY_RAISES_ASSERT(ldlt.vectorD())
  VERIFY_RAISES_ASSERT(ldlt.isPositive())
  VERIFY_RAISES_ASSERT(ldlt.isNegative())
  VERIFY_RAISES_ASSERT(ldlt.solve(tmp))
  VERIFY_RAISES_ASSERT(ldlt.solveInPlace(&tmp))
}

void test_cholesky()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( cholesky(Matrix<double,1,1>()) );
    CALL_SUBTEST_2( cholesky(MatrixXd(1,1)) );
    CALL_SUBTEST_3( cholesky(Matrix2d()) );
    CALL_SUBTEST_4( cholesky(Matrix3f()) );
    CALL_SUBTEST_5( cholesky(Matrix4d()) );
    CALL_SUBTEST_2( cholesky(MatrixXd(200,200)) );
    CALL_SUBTEST_6( cholesky(MatrixXcd(100,100)) );
  }

  CALL_SUBTEST_4( cholesky_verify_assert<Matrix3f>() );
  CALL_SUBTEST_7( cholesky_verify_assert<Matrix3d>() );
  CALL_SUBTEST_8( cholesky_verify_assert<MatrixXf>() );
  CALL_SUBTEST_2( cholesky_verify_assert<MatrixXd>() );

  // Test problem size constructors
  CALL_SUBTEST_9( LLT<MatrixXf>(10) );
  CALL_SUBTEST_9( LDLT<MatrixXf>(10) );
}
