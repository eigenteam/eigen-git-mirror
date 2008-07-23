
// g++ -DNDEBUG -O3 -I.. benchCholesky.cpp  -o benchCholesky && ./benchCholesky
// options:
//  -DBENCH_GSL -lgsl /usr/lib/libcblas.so.3
//  -DEIGEN_DONT_VECTORIZE
//  -msse2
//  -DREPEAT=100
//  -DTRIES=10
//  -DSCALAR=double

#include <Eigen/Array>
#include <Eigen/Cholesky>
#include <bench/BenchUtil.h>
using namespace Eigen;

#ifndef REPEAT
#define REPEAT 10000
#endif

#ifndef TRIES
#define TRIES 4
#endif

typedef float Scalar;

template <typename MatrixType>
__attribute__ ((noinline)) void benchCholesky(const MatrixType& m)
{
  int rows = m.rows();
  int cols = m.cols();

  int repeats = (REPEAT*1000)/(rows*rows);

  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> SquareMatrixType;

  MatrixType a = MatrixType::Random(rows,cols);
  SquareMatrixType covMat =  a * a.adjoint();

  BenchTimer timerNoSqrt, timerSqrt;

  Scalar acc = 0;
  int r = ei_random<int>(0,covMat.rows()-1);
  int c = ei_random<int>(0,covMat.cols()-1);
  for (int t=0; t<TRIES; ++t)
  {
    timerNoSqrt.start();
    for (int k=0; k<repeats; ++k)
    {
      CholeskyWithoutSquareRoot<SquareMatrixType> cholnosqrt(covMat);
      acc += cholnosqrt.matrixL().coeff(r,c);
    }
    timerNoSqrt.stop();
  }

  for (int t=0; t<TRIES; ++t)
  {
    timerSqrt.start();
    for (int k=0; k<repeats; ++k)
    {
      Cholesky<SquareMatrixType> chol(covMat);
      acc += chol.matrixL().coeff(r,c);
    }
    timerSqrt.stop();
  }

  if (MatrixType::RowsAtCompileTime==Dynamic)
    std::cout << "dyn   ";
  else
    std::cout << "fixed ";
  std::cout << covMat.rows() << " \t"
            << (timerNoSqrt.value() * REPEAT) / repeats << "s \t"
            << (timerSqrt.value() * REPEAT) / repeats << "s";


  #ifdef BENCH_GSL
  if (MatrixType::RowsAtCompileTime==Dynamic)
  {
    timerSqrt.reset();

    gsl_matrix* gslCovMat = gsl_matrix_alloc(covMat.rows(),covMat.cols());
    gsl_matrix* gslCopy = gsl_matrix_alloc(covMat.rows(),covMat.cols());
    
    eiToGsl(covMat, &gslCovMat);
    for (int t=0; t<TRIES; ++t)
    {
      timerSqrt.start();
      for (int k=0; k<repeats; ++k)
      {
        gsl_matrix_memcpy(gslCopy,gslCovMat);
        gsl_linalg_cholesky_decomp(gslCopy);
        acc += gsl_matrix_get(gslCopy,r,c);
      }
      timerSqrt.stop();
    }

    std::cout << " | \t"
              << timerSqrt.value() * REPEAT / repeats << "s";

    gsl_matrix_free(gslCovMat);
  }
  #endif
  std::cout << "\n";
  // make sure the compiler does not optimize too much
  if (acc==123)
    std::cout << acc;
}

int main(int argc, char* argv[])
{
  const int dynsizes[] = {/*4,6,8,12,16,24,32,49,64,67,128,129,130,131,132,*/256,257,258,259,260,512,0};
  std::cout << "size            no sqrt         standard";
  #ifdef BENCH_GSL
  std::cout << "       GSL (standard + double + ATLAS)  ";
  #endif
  std::cout << "\n";

  for (uint i=0; dynsizes[i]>0; ++i)
    benchCholesky(Matrix<Scalar,Dynamic,Dynamic>(dynsizes[i],dynsizes[i]));

//   benchCholesky(Matrix<Scalar,2,2>());
//   benchCholesky(Matrix<Scalar,3,3>());
//   benchCholesky(Matrix<Scalar,4,4>());
//   benchCholesky(Matrix<Scalar,5,5>());
//   benchCholesky(Matrix<Scalar,6,6>());
//   benchCholesky(Matrix<Scalar,7,7>());
//   benchCholesky(Matrix<Scalar,8,8>());
//   benchCholesky(Matrix<Scalar,12,12>());
//   benchCholesky(Matrix<Scalar,16,16>());
  return 0;
}

