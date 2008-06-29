
#include <Eigen/Array>
#include <Eigen/Sparse>
#include <bench/BenchTimer.h>



using namespace std;
using namespace Eigen;
USING_PART_OF_NAMESPACE_EIGEN

#ifndef SIZE
#define SIZE 1024
#endif

#ifndef DENSITY
#define DENSITY 0.01
#endif

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;
typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
typedef SparseMatrix<Scalar> EigenSparseMatrix;

void fillMatrix(float density, int rows, int cols,  EigenSparseMatrix& dst)
{
  dst.startFill(rows*cols*density);
  for(int j = 0; j < cols; j++)
  {
    for(int i = 0; i < rows; i++)
    {
      Scalar v = (ei_random<float>(0,1) < density) ? ei_random<Scalar>() : 0;
      if (v!=0)
        dst.fill(i,j) = v;
    }
  }
  dst.endFill();
}

void eiToDense(const EigenSparseMatrix& src, DenseMatrix& dst)
{
  dst.setZero();
  for (int j=0; j<src.cols(); ++j)
    for (EigenSparseMatrix::InnerIterator it(src.derived(), j); it; ++it)
      dst(it.index(),j) = it.value();
}

#ifndef NOGMM
#include "gmm/gmm.h"
typedef gmm::csc_matrix<Scalar> GmmSparse;
typedef gmm::col_matrix< gmm::wsvector<Scalar> > GmmDynSparse;
void eiToGmm(const EigenSparseMatrix& src, GmmSparse& dst)
{
  GmmDynSparse tmp(src.rows(), src.cols());
  for (int j=0; j<src.cols(); ++j)
    for (EigenSparseMatrix::InnerIterator it(src.derived(), j); it; ++it)
      tmp(it.index(),j) = it.value();
  gmm::copy(tmp, dst);
}
#endif

#ifndef NOMTL
#include <boost/numeric/mtl/mtl.hpp>
typedef mtl::compressed2D<Scalar, mtl::matrix::parameters<mtl::tag::col_major> > MtlSparse;
typedef mtl::compressed2D<Scalar, mtl::matrix::parameters<mtl::tag::row_major> > MtlSparseRowMajor;
void eiToMtl(const EigenSparseMatrix& src, MtlSparse& dst)
{
  mtl::matrix::inserter<MtlSparse> ins(dst);
  for (int j=0; j<src.cols(); ++j)
    for (EigenSparseMatrix::InnerIterator it(src.derived(), j); it; ++it)
      ins[it.index()][j] = it.value();
}
#endif
