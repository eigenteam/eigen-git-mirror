// g++ -O3 -DNDEBUG -I.. -fopenmp benchOpenMP.cpp -o benchOpenMP && ./benchOpenMP 2> /dev/null
// icpc -fast -fno-exceptions -DNDEBUG -I.. -openmp  benchOpenMP.cpp -o benchOpenMP && ./benchOpenMP 2> /dev/null

#include <omp.h>
#include "BenchUtil.h"
#include "basicbenchmark.h"

// #include <Eigen/Core>
// #include "BenchTimer.h"
//
// using namespace std;
// USING_PART_OF_NAMESPACE_EIGEN
//
// enum {LazyEval, EarlyEval, OmpEval};
//
// template<int Mode, typename MatrixType>
// double benchSingleProc(const MatrixType& mat, int iterations, int tries)  __attribute__((noinline));
//
// template<int Mode, typename MatrixType>
// double benchBasic(const MatrixType& mat, int iterations, int tries)
// {
//   const int rows = mat.rows();
//   const int cols = mat.cols();
//
//   Eigen::BenchTimer timer;
//   for(uint t=0; t<tries; ++t)
//   {
//     MatrixType I = MatrixType::identity(rows, cols);
//     MatrixType m = MatrixType::random(rows, cols);
//
//     timer.start();
//     for(int a = 0; a < iterations; a++)
//     {
//       if(Mode==LazyEval)
//         m = (I + 0.00005 * (m + m.lazyProduct(m))).eval();
//       else if(Mode==OmpEval)
//         m = (I + 0.00005 * (m + m.lazyProduct(m))).evalOMP();
//       else
//         m = I + 0.00005 * (m + m * m);
//     }
//     timer.stop();
//     cerr << m;
//   }
//   return timer.value();
// };

int main(int argc, char *argv[])
{
  // disbale floating point exceptions
  // this leads to more stable bench results
  {
    int aux;
    asm(
    "stmxcsr   %[aux]           \n\t"
    "orl       $32832, %[aux]   \n\t"
    "ldmxcsr   %[aux]           \n\t"
    : : [aux] "m" (aux));
  }

  // commented since the default setting is use as many threads as processors
  //omp_set_num_threads(omp_get_num_procs());

  std::cout << "double, fixed-size 4x4: "
    << benchBasic<LazyEval>(Matrix4d(), 10000, 10) << "s  "
    << benchBasic<OmpEval>(Matrix4d(), 10000, 10) << "s  \n";

  #define BENCH_MATRIX(TYPE, SIZE, ITERATIONS, TRIES) {\
      double single = benchBasic<LazyEval>(Matrix<TYPE,Eigen::Dynamic,Eigen::Dynamic>(SIZE,SIZE), ITERATIONS, TRIES); \
      double omp    = benchBasic<OmpEval> (Matrix<TYPE,Eigen::Dynamic,Eigen::Dynamic>(SIZE,SIZE), ITERATIONS, TRIES); \
      std::cout << #TYPE << ", " << #SIZE << "x" << #SIZE << ": " << single << "s " << omp << "s " \
        << " => x" << single/omp << " (" << omp_get_num_procs() << ")" << std::endl; \
    }

  BENCH_MATRIX(double,   32, 1000, 10);
  BENCH_MATRIX(double,  128,   10, 10);
  BENCH_MATRIX(double,  512,    1,  6);
  BENCH_MATRIX(double, 1024,    1,  4);

  return 0;
}

