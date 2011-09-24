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

template<typename LUSolver, typename Rhs, typename DenseMat, typename DenseRhs>
void check_slu(LUSolver& lu, const typename LUSolver::MatrixType& A, const Rhs& b, const DenseMat& dA, const DenseRhs& db)
{
  typedef typename LUSolver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;

  DenseRhs refX = dA.lu().solve(db);
  Scalar refDet = dA.determinant();

  Rhs x(b.rows(), b.cols());
  Rhs oldb = b;

  lu.compute(A);
  if (lu.info() != Success)
  {
    std::cerr << "sparse LU: factorization failed\n";
    return;
  }
  x = lu.solve(b);
  if (lu.info() != Success)
  {
    std::cerr << "sparse LU: solving failed\n";
    return;
  }
  VERIFY(oldb.isApprox(b) && "sparse LU: the rhs should not be modified!");

  VERIFY(refX.isApprox(x,test_precision<Scalar>()));

  if(A.cols()<30)
  {
    //std::cout << refDet << " == " << lu.determinant() << "\n";
    VERIFY_IS_APPROX(refDet,lu.determinant());
  }
}

template<typename LUSolver> void sparse_lu(LUSolver& lu)
{
  typedef typename LUSolver::MatrixType Mat;
  typedef typename Mat::Scalar Scalar;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;

  std::vector<Vector2i> zeroCoords;
  std::vector<Vector2i> nonzeroCoords;

  int size = internal::random<int>(1,300);
  double density = (std::max)(8./(size*size), 0.01);
  //int rhsSize = internal::random<int>(1,10);

  Mat m2(size, size);
  DenseMatrix refMat2(size, size);

  DenseVector b = DenseVector::Random(size);
  DenseVector refX(size), x(size);

  initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag, &zeroCoords, &nonzeroCoords);

  check_slu(lu, m2, b, refMat2, b);
}
