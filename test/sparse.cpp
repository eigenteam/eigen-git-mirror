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

#ifdef __GNUC__
#include <ext/hash_map>
#endif

#ifdef EIGEN_GOOGLEHASH_SUPPORT
  #include <google/sparse_hash_map>
#endif

#include "main.h"
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/Sparse>

enum {
  ForceNonZeroDiag = 1,
  MakeLowerTriangular = 2,
  MakeUpperTriangular = 4
};

template<typename Scalar> void
initSparse(double density,
           Matrix<Scalar,Dynamic,Dynamic>& refMat,
           SparseMatrix<Scalar>& sparseMat,
           int flags = 0,
           std::vector<Vector2i>* zeroCoords = 0,
           std::vector<Vector2i>* nonzeroCoords = 0)
{
  sparseMat.startFill(refMat.rows()*refMat.cols()*density);
  for(int j=0; j<refMat.cols(); j++)
  {
    for(int i=0; i<refMat.rows(); i++)
    {
      Scalar v = (ei_random<double>(0,1) < density) ? ei_random<Scalar>() : Scalar(0);
      if ((flags&ForceNonZeroDiag) && (i==j))
      {
        v = ei_random<Scalar>()*Scalar(3.);
        v = v*v + Scalar(5.);
      }
      if ((flags & MakeLowerTriangular) && j>i)
        v = Scalar(0);
      else if ((flags & MakeUpperTriangular) && j<i)
        v = Scalar(0);
      if (v!=Scalar(0))
      {
        sparseMat.fill(i,j) = v;
        if (nonzeroCoords)
          nonzeroCoords->push_back(Vector2i(i,j));
      }
      else if (zeroCoords)
      {
        zeroCoords->push_back(Vector2i(i,j));
      }
      refMat(i,j) = v;
    }
  }
  sparseMat.endFill();
}

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

template<typename Scalar> void sparse(int rows, int cols)
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

  VERIFY(zeroCoords.size()>0 && "re-run the test");
  VERIFY(nonzeroCoords.size()>0 && "re-run the test");

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

  // test InnerIterators and Block expressions
  for (int t=0; t<10; ++t)
  {
    int j = ei_random<int>(0,cols-1);
    int i = ei_random<int>(0,rows-1);
    int w = ei_random<int>(1,cols-j-1);
    int h = ei_random<int>(1,rows-i-1);

    VERIFY_IS_APPROX(m.block(i,j,h,w), refMat.block(i,j,h,w));
    for(int c=0; c<w; c++)
    {
      VERIFY_IS_APPROX(m.block(i,j,h,w).col(c), refMat.block(i,j,h,w).col(c));
      for(int r=0; r<h; r++)
      {
        VERIFY_IS_APPROX(m.block(i,j,h,w).col(c).coeff(r), refMat.block(i,j,h,w).col(c).coeff(r));
      }
    }
    for(int r=0; r<h; r++)
    {
      VERIFY_IS_APPROX(m.block(i,j,h,w).row(r), refMat.block(i,j,h,w).row(r));
      for(int c=0; c<w; c++)
      {
        VERIFY_IS_APPROX(m.block(i,j,h,w).row(r).coeff(c), refMat.block(i,j,h,w).row(r).coeff(c));
      }
    }
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

  // test transpose
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, rows);
    SparseMatrix<Scalar> m2(rows, rows);
    initSparse<Scalar>(density, refMat2, m2);
    VERIFY_IS_APPROX(m2.transpose().eval(), refMat2.transpose().eval());
    VERIFY_IS_APPROX(m2.transpose(), refMat2.transpose());
  }
#if 0
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

  // test triangular solver
  {
    DenseVector vec2 = vec1, vec3 = vec1;
    SparseMatrix<Scalar> m2(rows, cols);
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, cols);

    // lower
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeLowerTriangular, &zeroCoords, &nonzeroCoords);
    VERIFY_IS_APPROX(refMat2.template marked<Lower>().solveTriangular(vec2),
                     m2.template marked<Lower>().solveTriangular(vec3));

    // upper
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeUpperTriangular, &zeroCoords, &nonzeroCoords);
    VERIFY_IS_APPROX(refMat2.template marked<Upper>().solveTriangular(vec2),
                     m2.template marked<Upper>().solveTriangular(vec3));

    // TODO test row major
  }

  // test LLT
  if (!NumTraits<Scalar>::IsComplex)
  {
    // TODO fix the issue with complex (see SparseLLT::solveInPlace)
    SparseMatrix<Scalar> m2(rows, cols);
    DenseMatrix refMat2(rows, cols);
    
    DenseVector b = DenseVector::Random(cols);
    DenseVector refX(cols), x(cols);
    
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag|MakeLowerTriangular, &zeroCoords, &nonzeroCoords);
    refMat2 += refMat2.adjoint();
    refMat2.diagonal() *= 0.5;

    refMat2.llt().solve(b, &refX);
    typedef SparseMatrix<Scalar,Lower|SelfAdjoint> SparseSelfAdjointMatrix;
    x = b;
    SparseLLT<SparseSelfAdjointMatrix> (m2).solveInPlace(x);
    //VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LLT: default");
    #ifdef EIGEN_CHOLMOD_SUPPORT
    x = b;
    SparseLLT<SparseSelfAdjointMatrix,Cholmod>(m2).solveInPlace(x);
    VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LLT: cholmod");
    #endif
    #ifdef EIGEN_TAUCS_SUPPORT
    x = b;
    SparseLLT<SparseSelfAdjointMatrix,Taucs>(m2,IncompleteFactorization).solveInPlace(x);
    VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LLT: taucs (IncompleteFactorization)");
    x = b;
    SparseLLT<SparseSelfAdjointMatrix,Taucs>(m2,SupernodalMultifrontal).solveInPlace(x);
    VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LLT: taucs (SupernodalMultifrontal)");
    x = b;
    SparseLLT<SparseSelfAdjointMatrix,Taucs>(m2,SupernodalLeftLooking).solveInPlace(x);
    VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LLT: taucs (SupernodalLeftLooking)");
    #endif
  }

  // test LU
  {
    static int count = 0;
    SparseMatrix<Scalar> m2(rows, cols);
    DenseMatrix refMat2(rows, cols);
    
    DenseVector b = DenseVector::Random(cols);
    DenseVector refX(cols), x(cols);
    
    initSparse<Scalar>(density, refMat2, m2, ForceNonZeroDiag, &zeroCoords, &nonzeroCoords);

    LU<DenseMatrix> refLu(refMat2);
    refLu.solve(b, &refX);
    Scalar refDet = refLu.determinant();
    x.setZero();
    // // SparseLU<SparseMatrix<Scalar> > (m2).solve(b,&x);
    // // VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LU: default");
    #ifdef EIGEN_SUPERLU_SUPPORT
    {
      x.setZero();
      SparseLU<SparseMatrix<Scalar>,SuperLU> slu(m2);
      if (slu.succeeded())
      {
        if (slu.solve(b,&x)) {
          VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LU: SuperLU");
        }
        // std::cerr << refDet << " == " << slu.determinant() << "\n";
        if (count==0) {
          VERIFY_IS_APPROX(refDet,slu.determinant()); // FIXME det is not very stable for complex
        }
      }
    }
    #endif
    #ifdef EIGEN_UMFPACK_SUPPORT
    {
      // check solve
      x.setZero();
      SparseLU<SparseMatrix<Scalar>,UmfPack> slu(m2);
      if (slu.succeeded()) {
        if (slu.solve(b,&x)) {
          if (count==0) {
            VERIFY(refX.isApprox(x,test_precision<Scalar>()) && "LU: umfpack");  // FIXME solve is not very stable for complex
          }
        }
        VERIFY_IS_APPROX(refDet,slu.determinant());
        // TODO check the extracted data
        //std::cerr << slu.matrixL() << "\n";
      }
    }
    #endif
    count++;
  }
#endif
}

void test_sparse()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( sparse<double>(8, 8) );
    CALL_SUBTEST( sparse<std::complex<double> >(16, 16) );
    CALL_SUBTEST( sparse<double>(33, 33) );
  }
}
