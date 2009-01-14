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

#include "sparse.h"

template<typename SetterType,typename DenseType, typename SparseType>
bool test_random_setter(SparseType& sm, const DenseType& ref, const std::vector<Vector2i>& nonzeroCoords)
{
  {
    sm.setZero();
    SetterType w(sm);
    std::vector<Vector2i> remaining = nonzeroCoords;
    while(!remaining.empty())
    {
      int i = ei_random<int>(0,remaining.size()-1);
      w(remaining[i].x(),remaining[i].y()) = ref.coeff(remaining[i].x(),remaining[i].y());
      remaining[i] = remaining.back();
      remaining.pop_back();
    }
  }
  return sm.isApprox(ref);
}

template<typename Scalar> void sparse_basic(int rows, int cols)
{
  double density = std::max(8./(rows*cols), 0.01);
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;
  Scalar eps = 1e-6;

  SparseMatrix<Scalar> m(rows, cols);
  DenseMatrix refMat = DenseMatrix::Zero(rows, cols);
  DenseVector vec1 = DenseVector::Random(rows);

  std::vector<Vector2i> zeroCoords;
  std::vector<Vector2i> nonzeroCoords;
  initSparse<Scalar>(density, refMat, m, 0, &zeroCoords, &nonzeroCoords);

  if (zeroCoords.size()==0 || nonzeroCoords.size()==0)
    return;

  // test coeff and coeffRef
  for (int i=0; i<(int)zeroCoords.size(); ++i)
  {
    VERIFY_IS_MUCH_SMALLER_THAN( m.coeff(zeroCoords[i].x(),zeroCoords[i].y()), eps );
    VERIFY_RAISES_ASSERT( m.coeffRef(zeroCoords[0].x(),zeroCoords[0].y()) = 5 );
  }
  VERIFY_IS_APPROX(m, refMat);

  m.coeffRef(nonzeroCoords[0].x(), nonzeroCoords[0].y()) = Scalar(5);
  refMat.coeffRef(nonzeroCoords[0].x(), nonzeroCoords[0].y()) = Scalar(5);

  VERIFY_IS_APPROX(m, refMat);
  /*
  // test InnerIterators and Block expressions
  for (int t=0; t<10; ++t)
  {
    int j = ei_random<int>(0,cols-1);
    int i = ei_random<int>(0,rows-1);
    int w = ei_random<int>(1,cols-j-1);
    int h = ei_random<int>(1,rows-i-1);

//     VERIFY_IS_APPROX(m.block(i,j,h,w), refMat.block(i,j,h,w));
    for(int c=0; c<w; c++)
    {
      VERIFY_IS_APPROX(m.block(i,j,h,w).col(c), refMat.block(i,j,h,w).col(c));
      for(int r=0; r<h; r++)
      {
//         VERIFY_IS_APPROX(m.block(i,j,h,w).col(c).coeff(r), refMat.block(i,j,h,w).col(c).coeff(r));
      }
    }
//     for(int r=0; r<h; r++)
//     {
//       VERIFY_IS_APPROX(m.block(i,j,h,w).row(r), refMat.block(i,j,h,w).row(r));
//       for(int c=0; c<w; c++)
//       {
//         VERIFY_IS_APPROX(m.block(i,j,h,w).row(r).coeff(c), refMat.block(i,j,h,w).row(r).coeff(c));
//       }
//     }
  }

  for(int c=0; c<cols; c++)
  {
    VERIFY_IS_APPROX(m.col(c) + m.col(c), (m + m).col(c));
    VERIFY_IS_APPROX(m.col(c) + m.col(c), refMat.col(c) + refMat.col(c));
  }

  for(int r=0; r<rows; r++)
  {
    VERIFY_IS_APPROX(m.row(r) + m.row(r), (m + m).row(r));
    VERIFY_IS_APPROX(m.row(r) + m.row(r), refMat.row(r) + refMat.row(r));
  }
  */

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
//   {
//     m.setZero();
//     VERIFY_IS_NOT_APPROX(m, refMat);
//     SparseSetter<SparseMatrix<Scalar>, RandomAccessPattern> w(m);
//     std::vector<Vector2i> remaining = nonzeroCoords;
//     while(!remaining.empty())
//     {
//       int i = ei_random<int>(0,remaining.size()-1);
//       w->coeffRef(remaining[i].x(),remaining[i].y()) = refMat.coeff(remaining[i].x(),remaining[i].y());
//       remaining[i] = remaining.back();
//       remaining.pop_back();
//     }
//   }
//   VERIFY_IS_APPROX(m, refMat);

    VERIFY(( test_random_setter<RandomSetter<SparseMatrix<Scalar>, StdMapTraits> >(m,refMat,nonzeroCoords) ));
    #ifdef _HASH_MAP
    VERIFY(( test_random_setter<RandomSetter<SparseMatrix<Scalar>, GnuHashMapTraits> >(m,refMat,nonzeroCoords) ));
    #endif
    #ifdef _DENSE_HASH_MAP_H_
    VERIFY(( test_random_setter<RandomSetter<SparseMatrix<Scalar>, GoogleDenseHashMapTraits> >(m,refMat,nonzeroCoords) ));
    #endif
    #ifdef _SPARSE_HASH_MAP_H_
    VERIFY(( test_random_setter<RandomSetter<SparseMatrix<Scalar>, GoogleSparseHashMapTraits> >(m,refMat,nonzeroCoords) ));
    #endif

    // test fillrand
    {
      DenseMatrix m1(rows,cols);
      m1.setZero();
      SparseMatrix<Scalar> m2(rows,cols);
      m2.startFill();
      for (int j=0; j<cols; ++j)
      {
        for (int k=0; k<rows/2; ++k)
        {
          int i = ei_random<int>(0,rows-1);
          if (m1.coeff(i,j)==Scalar(0))
            m2.fillrand(i,j) = m1(i,j) = ei_random<Scalar>();
        }
      }
      m2.endFill();
      std::cerr << m1 << "\n\n" << m2 << "\n";
      VERIFY_IS_APPROX(m2,m1);
    }
//   {
//     m.setZero();
//     VERIFY_IS_NOT_APPROX(m, refMat);
// //     RandomSetter<SparseMatrix<Scalar> > w(m);
//     RandomSetter<SparseMatrix<Scalar>, GoogleDenseHashMapTraits > w(m);
//     // RandomSetter<SparseMatrix<Scalar>, GnuHashMapTraits > w(m);
//     std::vector<Vector2i> remaining = nonzeroCoords;
//     while(!remaining.empty())
//     {
//       int i = ei_random<int>(0,remaining.size()-1);
//       w(remaining[i].x(),remaining[i].y()) = refMat.coeff(remaining[i].x(),remaining[i].y());
//       remaining[i] = remaining.back();
//       remaining.pop_back();
//     }
//   }
//   std::cerr << m.transpose() << "\n\n"  << refMat.transpose() << "\n\n";
//   VERIFY_IS_APPROX(m, refMat);

  // test innerVector()
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, rows);
    SparseMatrix<Scalar> m2(rows, rows);
    initSparse<Scalar>(density, refMat2, m2);
    int j0 = ei_random(0,rows-1);
    int j1 = ei_random(0,rows-1);
//     VERIFY_IS_APPROX(m2.innerVector(j0), refMat2.col(j0));
//     VERIFY_IS_APPROX(m2.innerVector(j0)+m2.innerVector(j1), refMat2.col(j0)+refMat2.col(j1));
  }

  // test transpose
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, rows);
    SparseMatrix<Scalar> m2(rows, rows);
    initSparse<Scalar>(density, refMat2, m2);
    VERIFY_IS_APPROX(m2.transpose().eval(), refMat2.transpose().eval());
    VERIFY_IS_APPROX(m2.transpose(), refMat2.transpose());
  }

  // test matrix product
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, rows);
    DenseMatrix refMat3 = DenseMatrix::Zero(rows, rows);
    DenseMatrix refMat4 = DenseMatrix::Zero(rows, rows);
    SparseMatrix<Scalar> m2(rows, rows);
    SparseMatrix<Scalar> m3(rows, rows);
    SparseMatrix<Scalar> m4(rows, rows);
    initSparse<Scalar>(density, refMat2, m2);
    initSparse<Scalar>(density, refMat3, m3);
    initSparse<Scalar>(density, refMat4, m4);
    VERIFY_IS_APPROX(m4=m2*m3, refMat4=refMat2*refMat3);
    VERIFY_IS_APPROX(m4=m2.transpose()*m3, refMat4=refMat2.transpose()*refMat3);
    VERIFY_IS_APPROX(m4=m2.transpose()*m3.transpose(), refMat4=refMat2.transpose()*refMat3.transpose());
    VERIFY_IS_APPROX(m4=m2*m3.transpose(), refMat4=refMat2*refMat3.transpose());
  }
}

void test_sparse_basic()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( sparse_basic<double>(8, 8) );
    CALL_SUBTEST( sparse_basic<std::complex<double> >(16, 16) );
    CALL_SUBTEST( sparse_basic<double>(33, 33) );
  }
}
