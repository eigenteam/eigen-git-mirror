
// g++-4.4 bench_gemm.cpp -I .. -O2 -DNDEBUG -lrt -fopenmp && OMP_NUM_THREADS=2  ./a.out
// icpc bench_gemm.cpp -I .. -O3 -DNDEBUG -lrt -openmp  && OMP_NUM_THREADS=2  ./a.out

#include <Eigen/Core>
#include <iostream>
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
  int cache_size = -1;

  bool need_help = false;
  for (int i=1; i<argc; ++i)
  {
    if(argv[i][0]=='s')
      s = atoi(argv[i]+1);
    else if(argv[i][0]=='c')
      cache_size = atoi(argv[i]+1);
    else
      need_help = true;
  }

  if(need_help)
  {
    std::cout << argv[0] << " s<matrix size> c<cache size> \n";
    return 1;
  }

  if(cache_size>0)
    setCpuCacheSizes(cache_size,32*cache_size);

  std::cout << "Matrix size = " << s << "\n";
  std::ptrdiff_t cm, cn, ck;
  getBlockingSizes<Scalar>(ck, cm, cn);
  std::cout << "blocking size = " << cm << " x " << ck << "\n";

  int m = s;
  int n = s;
  int p = s;
  M a(m,n); a.setRandom();
  M b(n,p); b.setRandom();
  M c(m,p); c.setOnes();

  M r = c;

  // check the parallel product is correct
  #ifdef EIGEN_HAS_OPENMP
  int procs = omp_get_max_threads();
  if(procs>1)
  {
    #ifdef HAVE_BLAS
    blas_gemm(a,b,r);
    #else
    omp_set_num_threads(1);
    r.noalias() += a * b;
    omp_set_num_threads(procs);
    #endif
    c.noalias() += a * b;
    if(!r.isApprox(c)) std::cerr << "Warning, your parallel product is crap!\n\n";
  }
  #endif

  #ifdef HAVE_BLAS
  BenchTimer tblas;
  BENCH(tblas, tries, rep, blas_gemm(a,b,c));
  std::cout << "blas  cpu         " << tblas.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/tblas.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << tblas.total(CPU_TIMER)  << "s)\n";
  std::cout << "blas  real        " << tblas.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/tblas.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << tblas.total(REAL_TIMER) << "s)\n";
  #endif

  BenchTimer tmt;
  BENCH(tmt, tries, rep, gemm(a,b,c));
  std::cout << "eigen cpu         " << tmt.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/tmt.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << tmt.total(CPU_TIMER)  << "s)\n";
  std::cout << "eigen real        " << tmt.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/tmt.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << tmt.total(REAL_TIMER) << "s)\n";

  #ifdef EIGEN_HAS_OPENMP
  if(procs>1)
  {
    BenchTimer tmono;
    //omp_set_num_threads(1);
    Eigen::setNbThreads(1);
    BENCH(tmono, tries, rep, gemm(a,b,c));
    std::cout << "eigen mono cpu    " << tmono.best(CPU_TIMER)/rep  << "s  \t" << (double(m)*n*p*rep*2/tmono.best(CPU_TIMER))*1e-9  <<  " GFLOPS \t(" << tmono.total(CPU_TIMER)  << "s)\n";
    std::cout << "eigen mono real   " << tmono.best(REAL_TIMER)/rep << "s  \t" << (double(m)*n*p*rep*2/tmono.best(REAL_TIMER))*1e-9 <<  " GFLOPS \t(" << tmono.total(REAL_TIMER) << "s)\n";
    std::cout << "mt speed up x" << tmono.best(CPU_TIMER) / tmt.best(REAL_TIMER)  << " => " << (100.0*tmono.best(CPU_TIMER) / tmt.best(REAL_TIMER))/procs << "%\n";
  }
  #endif

  return 0;
}

