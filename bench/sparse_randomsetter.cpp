
#define NOGMM
#define NOMTL

#include <map>
#include <ext/hash_map>
#include <google/dense_hash_map>
#include <google/sparse_hash_map>

#ifndef SIZE
#define SIZE 10000
#endif

#ifndef DENSITY
#define DENSITY 0.01
#endif

#ifndef REPEAT
#define REPEAT 1
#endif

#include "BenchSparseUtil.h"

#ifndef MINDENSITY
#define MINDENSITY 0.0004
#endif

#ifndef NBTRIES
#define NBTRIES 10
#endif

#define BENCH(X) \
  timer.reset(); \
  for (int _j=0; _j<NBTRIES; ++_j) { \
    timer.start(); \
    for (int _k=0; _k<REPEAT; ++_k) { \
        X  \
  } timer.stop(); }

int main(int argc, char *argv[])
{
  int rows = SIZE;
  int cols = SIZE;
  float density = DENSITY;

  EigenSparseMatrix sm1(rows,cols), sm2(rows,cols);


  int n = rows*cols*density;
  std::cout << "n = " << n << "\n";
  int dummy;
  BenchTimer t;

  t.reset(); t.start();
  for (int k=0; k<n; ++k)
    dummy = ei_random<int>(0,rows-1) + ei_random<int>(0,cols-1);
  t.stop();
  double rtime = t.value();
  std::cout << "rtime = " << rtime << " (" << dummy << ")\n\n";
  const int Bits = 6;
  for (;;)
  {
    {
      RandomSetter<EigenSparseMatrix,StdMapTraits,Bits> set1(sm1);
      t.reset(); t.start();
      for (int k=0; k<n; ++k)
        set1(ei_random<int>(0,rows-1),ei_random<int>(0,cols-1)) += 1;
      t.stop();
      std::cout << "std::map =>      \t" << t.value()-rtime
                << " nnz=" << set1.nonZeros() << "\n";getchar();
    }
    {
      RandomSetter<EigenSparseMatrix,GnuHashMapTraits,Bits> set1(sm1);
      t.reset(); t.start();
      for (int k=0; k<n; ++k)
        set1(ei_random<int>(0,rows-1),ei_random<int>(0,cols-1)) += 1;
      t.stop();
      std::cout << "gnu::hash_map => \t" << t.value()-rtime
                << " nnz=" << set1.nonZeros() << "\n";getchar();
    }
    {
      RandomSetter<EigenSparseMatrix,GoogleDenseHashMapTraits,Bits> set1(sm1);
      t.reset(); t.start();
      for (int k=0; k<n; ++k)
        set1(ei_random<int>(0,rows-1),ei_random<int>(0,cols-1)) += 1;
      t.stop();
      std::cout << "google::dense => \t" << t.value()-rtime
                << " nnz=" << set1.nonZeros() << "\n";getchar();
    }
    {
      RandomSetter<EigenSparseMatrix,GoogleSparseHashMapTraits,Bits> set1(sm1);
      t.reset(); t.start();
      for (int k=0; k<n; ++k)
        set1(ei_random<int>(0,rows-1),ei_random<int>(0,cols-1)) += 1;
      t.stop();
      std::cout << "google::sparse => \t" << t.value()-rtime
                << " nnz=" << set1.nonZeros() << "\n";getchar();
    }
    std::cout << "\n\n";
  }

  return 0;
}

