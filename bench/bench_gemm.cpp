
// g++-4.4 bench_gemm.cpp -I .. -O2 -DNDEBUG -lrt -fopenmp && OMP_NUM_THREADS=2  ./a.out
// icpc bench_gemm.cpp -I .. -O3 -DNDEBUG -lrt -openmp  && OMP_NUM_THREADS=2  ./a.out

#include <Eigen/Core>
#include "../../eigen2/bench/BenchTimer.h"

using namespace std;
using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;
typedef Matrix<Scalar,Dynamic,Dynamic> M;

void gemm(const M& a, const M& b, M& c)
{
  c.noalias() += a * b;
}

int main(int argc, char ** argv)
{
  int rep = 2;
  int s = 1024;
  int m = s;
  int n = s;
  int p = s;
  M a(m,n); a.setOnes();
  M b(n,p); b.setOnes();
  M c(m,p); c.setOnes();

  BenchTimer t;

  BENCH(t, 5, rep, gemm(a,b,c));

  std::cerr << "cpu   " << t.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/t.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << t.total(CPU_TIMER)  << "s)\n";
  std::cerr << "real  " << t.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/t.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << t.total(REAL_TIMER) << "s)\n";

  return 0;
}

