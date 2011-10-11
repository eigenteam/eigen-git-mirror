// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
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

#include "sparse.h"
#include <Eigen/SparseExtra>

template<typename Solver, typename Rhs, typename DenseMat, typename DenseRhs>
void check_sparse_solving(Solver& solver, const typename Solver::MatrixType& A, const Rhs& b, const DenseMat& dA, const DenseRhs& db)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;

  DenseRhs refX = dA.lu().solve(db);

  Rhs x(b.rows(), b.cols());
  Rhs oldb = b;

  solver.compute(A);
  if (solver.info() != Success)
  {
    std::cerr << "sparse SPD: factorization failed (check_sparse_solving)\n";
    exit(0);
    return;
  }
  x = solver.solve(b);
  if (solver.info() != Success)
  {
    std::cerr << "sparse SPD: solving failed\n";
    return;
  }
  VERIFY(oldb.isApprox(b) && "sparse SPD: the rhs should not be modified!");

  VERIFY(x.isApprox(refX,test_precision<Scalar>()));
}

template<typename Solver, typename DenseMat>
void check_sparse_determinant(Solver& solver, const typename Solver::MatrixType& A, const DenseMat& dA)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef typename Mat::RealScalar RealScalar;
  
  solver.compute(A);
  if (solver.info() != Success)
  {
    std::cerr << "sparse SPD: factorization failed (check_sparse_determinant)\n";
    return;
  }

  Scalar refDet = dA.determinant();
  VERIFY_IS_APPROX(refDet,solver.determinant());
}


template<typename Solver, typename DenseMat>
int generate_sparse_spd_problem(Solver& , typename Solver::MatrixType& A, typename Solver::MatrixType& halfA, DenseMat& dA, int maxSize = 300)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;

  int size = internal::random<int>(1,maxSize);
  double density = (std::max)(8./(size*size), 0.01);

  Mat M(size, size);
  DenseMatrix dM(size, size);

  initSparse<Scalar>(density, dM, M, ForceNonZeroDiag);

  A = M * M.adjoint();
  dA = dM * dM.adjoint();
  
  halfA.resize(size,size);
  halfA.template selfadjointView<Solver::UpLo>().rankUpdate(M);
  
  return size;
}

template<typename Solver> void check_sparse_spd_solving(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;

  // generate the problem
  Mat A, halfA;
  DenseMatrix dA;
  int size = generate_sparse_spd_problem(solver, A, halfA, dA);

  // generate the right hand sides
  int rhsCols = internal::random<int>(1,16);
  double density = (std::max)(8./(size*rhsCols), 0.1);
  Mat B(size,rhsCols);
  DenseVector b = DenseVector::Random(size);
  DenseMatrix dB(size,rhsCols);
  initSparse<Scalar>(density, dB, B);

  check_sparse_solving(solver, A,     b,  dA, b);
  check_sparse_solving(solver, halfA, b,  dA, b);
  check_sparse_solving(solver, A,     dB, dA, dB);
  check_sparse_solving(solver, halfA, dB, dA, dB);
  check_sparse_solving(solver, A,     B,  dA, dB);
  check_sparse_solving(solver, halfA, B,  dA, dB);
}

template<typename Solver> void check_sparse_spd_determinant(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;

  // generate the problem
  Mat A, halfA;
  DenseMatrix dA;
  generate_sparse_spd_problem(solver, A, halfA, dA, 30);

  check_sparse_determinant(solver, A,     dA);
  check_sparse_determinant(solver, halfA, dA );
}

template<typename Solver, typename DenseMat>
int generate_sparse_square_problem(Solver&, typename Solver::MatrixType& A, DenseMat& dA, int maxSize = 300)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;

  int size = internal::random<int>(1,maxSize);
  double density = (std::max)(8./(size*size), 0.01);
  
  A.resize(size,size);
  dA.resize(size,size);

  initSparse<Scalar>(density, dA, A, ForceNonZeroDiag);
  
  return size;
}

template<typename Solver> void check_sparse_square_solving(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;

  int rhsCols = internal::random<int>(1,16);

  Mat A;
  DenseMatrix dA;
  int size = generate_sparse_square_problem(solver, A, dA);

  DenseVector b = DenseVector::Random(size);
  DenseMatrix dB = DenseMatrix::Random(size,rhsCols);

  check_sparse_solving(solver, A, b,  dA, b);
  check_sparse_solving(solver, A, dB, dA, dB);
}

template<typename Solver> void check_sparse_square_determinant(Solver& solver)
{
  typedef typename Solver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;

  // generate the problem
  Mat A;
  DenseMatrix dA;
  generate_sparse_square_problem(solver, A, dA, 30);

  check_sparse_determinant(solver, A, dA);
}
