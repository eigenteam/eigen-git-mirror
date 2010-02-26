
// g++-4.4 bench_gemm.cpp -I .. -O2 -DNDEBUG -lrt -fopenmp && OMP_NUM_THREADS=2  ./a.out
// icpc bench_gemm.cpp -I .. -O3 -DNDEBUG -lrt -openmp  && OMP_NUM_THREADS=2  ./a.out

#include <Eigen/Core>
#include <bench/BenchTimer.h>

using namespace std;
using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;
typedef Matrix<Scalar,Dynamic,Dynamic> M;

#ifdef HAVE_BLAS

extern "C" {
  #include <bench/btl/libs/C_BLAS/blas.h>

  void sgemm_kernel(int actual_mc, int cols, int actual_kc, float alpha,
                    float* blockA, float* blockB, float* res, int resStride);
  void sgemm_oncopy(int actual_kc, int cols, const float* rhs, int rhsStride, float* blockB);
  void sgemm_itcopy(int actual_kc, int cols, const float* rhs, int rhsStride, float* blockB);
}

static float fone = 1;
static float fzero = 0;
static double done = 1;
static double szero = 0;
static char notrans = 'N';
static char trans = 'T';
static char nonunit = 'N';
static char lower = 'L';
static char right = 'R';
static int intone = 1;

void blas_gemm(const MatrixXf& a, const MatrixXf& b, MatrixXf& c)
{
  int M = c.rows(); int N = c.cols(); int K = a.cols();
  int lda = a.rows(); int ldb = b.rows(); int ldc = c.rows();

  sgemm_(&notrans,&notrans,&M,&N,&K,&fone,
         const_cast<float*>(a.data()),&lda,
         const_cast<float*>(b.data()),&ldb,&fone,
         c.data(),&ldc);
}

void blas_gemm(const MatrixXd& a, const MatrixXd& b, MatrixXd& c)
{
  int M = c.rows(); int N = c.cols(); int K = a.cols();
  int lda = a.rows(); int ldb = b.rows(); int ldc = c.rows();

  dgemm_(&notrans,&notrans,&M,&N,&K,&done,
         const_cast<double*>(a.data()),&lda,
         const_cast<double*>(b.data()),&ldb,&done,
         c.data(),&ldc);
}

#endif

void gemm(const M& a, const M& b, M& c)
{
  c.noalias() += a * b;
}

int main(int argc, char ** argv)
{
  int rep = 1;    // number of repetitions per try
  int tries = 5;  // number of tries, we keep the best

  int s = 2048;
  int m = s;
  int n = s;
  int p = s;
  M a(m,n); a.setRandom();
  M b(n,p); b.setRandom();
  M c(m,p); c.setOnes();

  BenchTimer t;

  M r = c;

  // check the parallel product is correct
  #ifdef HAVE_BLAS
  blas_gemm(a,b,r);
  #else
  int procs = omp_get_max_threads();
  omp_set_num_threads(1);
  r.noalias() += a * b;
  omp_set_num_threads(procs);
  #endif
  c.noalias() += a * b;
  if(!r.isApprox(c)) std::cerr << "Warning, your parallel product is crap!\n\n";

  #ifdef HAVE_BLAS
  BENCH(t, tries, rep, blas_gemm(a,b,c));
  std::cerr << "blas  cpu   " << t.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/t.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << t.total(CPU_TIMER)  << "s)\n";
  std::cerr << "blas  real  " << t.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/t.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << t.total(REAL_TIMER) << "s)\n";
  #endif

  BENCH(t, tries, rep, gemm(a,b,c));
  std::cerr << "eigen cpu   " << t.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/t.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << t.total(CPU_TIMER)  << "s)\n";
  std::cerr << "eigen real  " << t.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/t.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << t.total(REAL_TIMER) << "s)\n";

  return 0;
}

