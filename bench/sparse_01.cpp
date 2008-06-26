
// g++ -O3 -DNDEBUG sparse_01.cpp -I .. -o sparse_01 && ./sparse_01

#include <Eigen/Array>
#include <Eigen/Sparse>
#include <bench/BenchTimer.h>

#include "gmm/gmm.h"

using namespace std;
using namespace Eigen;
USING_PART_OF_NAMESPACE_EIGEN

#ifndef REPEAT
#define REPEAT 10
#endif

#define REPEATPRODUCT 1

#define SIZE 10
#define DENSITY 0.2

// #define NODENSEMATRIX

typedef MatrixXf DenseMatrix;
// typedef Matrix<float,SIZE,SIZE> DenseMatrix;
typedef SparseMatrix<float> EigenSparseMatrix;
typedef gmm::csc_matrix<float> GmmSparse;
typedef gmm::col_matrix< gmm::wsvector<float> > GmmDynSparse;

void fillMatrix(float density, int rows, int cols, DenseMatrix* pDenseMatrix, EigenSparseMatrix* pSparseMatrix, GmmSparse* pGmmMatrix=0)
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
  int rows = SIZE;
  int cols = SIZE;
  float density = DENSITY;

  // dense matrices
  #ifndef NODENSEMATRIX
  DenseMatrix m1(rows,cols), m2(rows,cols), m3(rows,cols), m4(rows,cols);
  #endif

  // sparse matrices
  EigenSparseMatrix sm1(rows,cols), sm2(rows,cols), sm3(rows,cols), sm4(rows,cols);

  HashMatrix<float> hm4(rows,cols);

  // GMM++ matrices

  GmmDynSparse  gmmT4(rows,cols);
  GmmSparse gmmM1(rows,cols), gmmM2(rows,cols), gmmM3(rows,cols), gmmM4(rows,cols);

  #ifndef NODENSEMATRIX
  fillMatrix(density, rows, cols, &m1, &sm1, &gmmM1);
  fillMatrix(density, rows, cols, &m2, &sm2, &gmmM2);
  fillMatrix(density, rows, cols, &m3, &sm3, &gmmM3);
  #else
  fillMatrix(density, rows, cols, 0, &sm1, &gmmM1);
  fillMatrix(density, rows, cols, 0, &sm2, &gmmM2);
  fillMatrix(density, rows, cols, 0, &sm3, &gmmM3);
  #endif

  BenchTimer timer;

  //--------------------------------------------------------------------------------
  //  COEFF WISE OPERATORS
  //--------------------------------------------------------------------------------
#if 1
  std::cout << "\n\n\"m4 = m1 + m2 + 2 * m3\":\n\n";

  timer.reset();
  timer.start();
  asm("#begin");
  for (int k=0; k<REPEAT; ++k)
    m4 = m1 + m2 + 2 * m3;
  asm("#end");
  timer.stop();
  std::cout << "Eigen dense = " << timer.value() << endl;

  timer.reset();
  timer.start();
  for (int k=0; k<REPEAT; ++k)
    sm4 = sm1 + sm2 + 2 * sm3;
  timer.stop();
  std::cout << "Eigen sparse = " << timer.value() << endl;

  timer.reset();
  timer.start();
  for (int k=0; k<REPEAT; ++k)
    hm4 = sm1 + sm2 + 2 * sm3;
  timer.stop();
  std::cout << "Eigen hash = " << timer.value() << endl;

  LinkedVectorMatrix<float> lm4(rows, cols);
  timer.reset();
  timer.start();
  for (int k=0; k<REPEAT; ++k)
    lm4 = sm1 + sm2 + 2 * sm3;
  timer.stop();
  std::cout << "Eigen linked vector = " << timer.value() << endl;

  timer.reset();
  timer.start();
  for (int k=0; k<REPEAT; ++k)
  {
    gmm::add(gmmM1, gmmM2, gmmT4);
    gmm::add(gmm::scaled(gmmM3,2), gmmT4);
  }
  timer.stop();
  std::cout << "GMM++ sparse = " << timer.value() << endl;
#endif
  //--------------------------------------------------------------------------------
  //  PRODUCT
  //--------------------------------------------------------------------------------
#if 0
  std::cout << "\n\nProduct:\n\n";

  #ifndef NODENSEMATRIX
  timer.reset();
  timer.start();
  asm("#begin");
  for (int k=0; k<REPEATPRODUCT; ++k)
    m1 = m1 * m2;
  asm("#end");
  timer.stop();
  std::cout << "Eigen dense = " << timer.value() << endl;
  #endif

  timer.reset();
  timer.start();
  for (int k=0; k<REPEATPRODUCT; ++k)
    sm4 = sm1 * sm2;
  timer.stop();
  std::cout << "Eigen sparse = " << timer.value() << endl;

//   timer.reset();
//   timer.start();
//   for (int k=0; k<REPEATPRODUCT; ++k)
//     hm4 = sm1 * sm2;
//   timer.stop();
//   std::cout << "Eigen hash = " << timer.value() << endl;

  timer.reset();
  timer.start();
  for (int k=0; k<REPEATPRODUCT; ++k)
  {
    gmm::csr_matrix<float> R(rows,cols);
    gmm::copy(gmmM1, R);
    //gmm::mult(gmmM1, gmmM2, gmmT4);
  }
  timer.stop();
  std::cout << "GMM++ sparse = " << timer.value() << endl;
#endif
  //--------------------------------------------------------------------------------
  //  VARIOUS
  //--------------------------------------------------------------------------------
#if 1
//   sm3 = sm1 + m2;
  cout << m4.transpose() << "\n\n";
//   sm4 = sm1+sm2;
  cout << sm4 << "\n\n";
  cout << lm4 << endl;

  LinkedVectorMatrix<float,RowMajorBit> lm5(rows, cols);
  lm5 = lm4;
  lm5 = sm4;
  cout << endl << lm5 << endl;

  sm3 = sm4.transpose();
  cout << endl << lm5 << endl;

  cout << endl << "SM1 before random editing: " << endl << sm1 << endl;
  {
    SparseSetter<EigenSparseMatrix,RandomAccessPattern> w1(sm1);
    w1->coeffRef(4,2) = ei_random<float>();
    w1->coeffRef(2,6) = ei_random<float>();
    w1->coeffRef(0,4) = ei_random<float>();
    w1->coeffRef(9,3) = ei_random<float>();
  }
  cout << endl << "SM1 after random editing: " << endl << sm1 << endl;
#endif

  return 0;
}

