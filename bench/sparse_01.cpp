
// g++ -O3 -DNDEBUG sparse_01.cpp -I .. -o sparse_01 && ./sparse_01

#include <Eigen/Array>
#include <Eigen/Sparse>
#include <bench/BenchTimer.h>

#include "gmm/gmm.h"

using namespace std;
using namespace Eigen;
USING_PART_OF_NAMESPACE_EIGEN

#ifndef REPEAT
#define REPEAT 40000000
#endif

typedef MatrixXf DenseMatrix;
typedef SparseMatrix<float> EigenSparseMatrix;
typedef gmm::csc_matrix<float> GmmSparse;
typedef gmm::col_matrix< gmm::wsvector<float> > GmmDynSparse;

void fillMatrix(float density, int rows, int cols, MatrixXf* pDenseMatrix, EigenSparseMatrix* pSparseMatrix, GmmSparse* pGmmMatrix=0)
{
  GmmDynSparse gmmT(rows, cols);
  if (pSparseMatrix)
    pSparseMatrix->startFill(rows*cols*density);
  for(int j = 0; j < cols; j++)
  {
    for(int i = 0; i < rows; i++)
    {
      float v = (ei_random<float>(0,1) < density) ? ei_random<float>() : 0;
      if (pDenseMatrix)
        (*pDenseMatrix)(i,j) = v;
      if (v!=0)
      {
        if (pSparseMatrix)
          pSparseMatrix->fill(i,j) = v;
        if (pGmmMatrix)
          gmmT(i,j) = v;
      }
    }
  }
  if (pSparseMatrix)
    pSparseMatrix->endFill();
  if (pGmmMatrix)
    gmm::copy(gmmT, *pGmmMatrix);
}

int main(int argc, char *argv[])
{
  int rows = 4000;
  int cols = 4000;
  float density = 0.1;

  // dense matrices
  DenseMatrix m1(rows,cols), m2(rows,cols), m3(rows,cols), m4(rows,cols);

  // sparse matrices
  EigenSparseMatrix sm1(rows,cols), sm2(rows,cols), sm3(rows,cols), sm4(rows,cols);

  // GMM++ matrices

  GmmDynSparse  gmmT4(rows,cols);
  GmmSparse gmmM1(rows,cols), gmmM2(rows,cols), gmmM3(rows,cols), gmmM4(rows,cols);

  fillMatrix(density, rows, cols, &m1, &sm1, &gmmM1);
  fillMatrix(density, rows, cols, &m2, &sm2, &gmmM2);
  fillMatrix(density, rows, cols, &m3, &sm3, &gmmM3);

  BenchTimer timer;

  timer.start();
  for (int k=0; k<10; ++k)
    m4 = m1 + m2 + 2 * m3;
  timer.stop();
  std::cout << "Eigen dense = " << timer.value() << endl;

  timer.reset();
  timer.start();
  for (int k=0; k<10; ++k)
    sm4 = sm1 + sm2 + 2 * sm3;
  timer.stop();
  std::cout << "Eigen sparse = " << timer.value() << endl;

  timer.reset();
  timer.start();
  for (int k=0; k<10; ++k)
  {
    gmm::add(gmmM1, gmmM2, gmmT4);
    gmm::add(gmm::scaled(gmmM3,2), gmmT4);
  }
  timer.stop();
  std::cout << "GMM++ sparse = " << timer.value() << endl;

//   sm3 = sm1 + m2;
//   cout << m4.transpose() << "\n\n" << sm4 << endl;
  return 0;
}

