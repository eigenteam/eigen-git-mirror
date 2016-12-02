#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include "../../BenchTimer.h"
using namespace Eigen;

#ifndef SCALAR
#error SCALAR must be defined
#endif

typedef SCALAR Scalar;

typedef Matrix<Scalar,Dynamic,Dynamic> Mat;
typedef Matrix<Scalar,Dynamic,1>       Vec;

EIGEN_DONT_INLINE
void gemv(const Mat &A, const Vec &B, Vec &C)
{
  C.noalias() += A * B;
}

EIGEN_DONT_INLINE
double bench(long m, long n)
{
  Mat A(m,n);
  Vec B(n);
  Vec C(m);
  A.setRandom();
  B.setRandom();
  C.setZero();
  
  BenchTimer t;
  
  double up = 1e9*4/sizeof(Scalar);
  double tm0 = 4, tm1 = 10;
  if(NumTraits<Scalar>::IsComplex)
  {
    up /= 4;
    tm0 = 2;
    tm1 = 4;
  }
  
  double flops = 2. * m * n;
  long rep = std::max(1., std::min(100., up/flops) );
  long tries = std::max(tm0, std::min(tm1, up/flops) );
  
  BENCH(t, tries, rep, gemv(A,B,C));
  
  return 1e-9 * rep * flops / t.best();
}

int main(int argc, char **argv)
{
  std::vector<double> results;
  
  std::ifstream settings("gemv_settings.txt");
  long m, n;
  while(settings >> m >> n)
  {
    //std::cerr << "  Testing " << m << " " << n << " " << k << std::endl;
    results.push_back( bench(m, n) );
  }
  
  std::cout << RowVectorXd::Map(results.data(), results.size());
  
  return 0;
}
