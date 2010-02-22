
#include <Eigen/Core>
#include <../eigen2/bench/BenchTimer.h>

extern "C"
{
  #include <bench/btl/libs/C_BLAS/blas.h>
  #include <cblas.h>
}

using namespace std;
using namespace Eigen;

#ifndef SCALAR
#define SCALAR float
#endif

typedef SCALAR Scalar;
typedef Matrix<Scalar,Dynamic,Dynamic> M;

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
//   cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, c.rows(), c.cols(), a.cols(), 1, a.data(), a.rows(), b.data(), b.rows(), 1, c.data(), c.rows());
  int M = c.rows();
  int N = c.cols();
  int K = a.cols();

  int lda = a.rows();
  int ldb = b.rows();
  int ldc = c.rows();

  sgemm_(&notrans,&notrans,&M,&N,&K,&fone,
         const_cast<float*>(a.data()),&lda,
         const_cast<float*>(b.data()),&ldb,&fzero,
         c.data(),&ldc);
}

int main(int argc, char **argv)
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

  BENCH(t, 5, rep, blas_gemm(a,b,c));

  std::cerr << "cpu   " << t.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/t.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << t.total(CPU_TIMER)  << "s)\n";
  std::cerr << "real  " << t.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/t.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << t.total(REAL_TIMER) << "s)\n";
  return 0;
}

