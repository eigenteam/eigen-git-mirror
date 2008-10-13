
//g++ -O3 -g0 -DNDEBUG  sparse_product.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.005 -DSIZE=10000 && ./a.out
//g++ -O3 -g0 -DNDEBUG  sparse_product.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.05 -DSIZE=2000 && ./a.out
// -DNOGMM -DNOMTL

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

  EigenSparseMatrix sm1(rows,cols), sm2(rows,cols), sm3(rows,cols), sm4(rows,cols);

  BenchTimer timer;
  for (float density = DENSITY; density>=MINDENSITY; density*=0.5)
  {
    fillMatrix(density, rows, cols, sm1);
    fillMatrix(density, rows, cols, sm2);

    // dense matrices
    #ifdef DENSEMATRIX
    {
      std::cout << "Eigen Dense\t" << density*100 << "%\n";
      DenseMatrix m1(rows,cols), m2(rows,cols), m3(rows,cols);
      eiToDense(sm1, m1);
      eiToDense(sm2, m2);

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        m3 = m1 * m2;
      timer.stop();
      std::cout << "   a * b:\t" << timer.value() << endl;

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        m3 = m1.transpose() * m2;
      timer.stop();
      std::cout << "   a' * b:\t" << timer.value() << endl;

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        m3 = m1.transpose() * m2.transpose();
      timer.stop();
      std::cout << "   a' * b':\t" << timer.value() << endl;

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        m3 = m1 * m2.transpose();
      timer.stop();
      std::cout << "   a * b':\t" << timer.value() << endl;
    }
    #endif

    // eigen sparse matrices
    {
      std::cout << "Eigen sparse\t" << density*100 << "%\n";

//       timer.reset();
//       timer.start();
      BENCH(for (int k=0; k<REPEAT; ++k) sm3 = sm1 * sm2;)
//       timer.stop();
      std::cout << "   a * b:\t" << timer.value() << endl;

      timer.reset();
      timer.start();
//       std::cerr << "transpose...\n";
//       EigenSparseMatrix sm4 = sm1.transpose();
//       std::cout << sm4.nonZeros() << " == " << sm1.nonZeros() << "\n";
//       exit(1);
//       std::cerr << "transpose OK\n";
//       std::cout << sm1 << "\n\n" << sm1.transpose() << "\n\n" << sm4.transpose() << "\n\n";
      BENCH(for (int k=0; k<REPEAT; ++k) sm3 = sm1.transpose() * sm2;)
//       timer.stop();
      std::cout << "   a' * b:\t" << timer.value() << endl;

//       timer.reset();
//       timer.start();
      BENCH( for (int k=0; k<REPEAT; ++k) sm3 = sm1.transpose() * sm2.transpose(); )
//       timer.stop();
      std::cout << "   a' * b':\t" << timer.value() << endl;

//       timer.reset();
//       timer.start();
      BENCH( for (int k=0; k<REPEAT; ++k) sm3 = sm1 * sm2.transpose(); )
//       timer.stop();
      std::cout << "   a * b' :\t" << timer.value() << endl;
    }

    // GMM++
    #ifndef NOGMM
    {
      std::cout << "GMM++ sparse\t" << density*100 << "%\n";
      GmmDynSparse  gmmT3(rows,cols);
      GmmSparse m1(rows,cols), m2(rows,cols), m3(rows,cols);
      eiToGmm(sm1, m1);
      eiToGmm(sm2, m2);

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        gmm::mult(m1, m2, gmmT3);
      timer.stop();
      std::cout << "   a * b:\t" << timer.value() << endl;

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        gmm::mult(gmm::transposed(m1), m2, gmmT3);
      timer.stop();
      std::cout << "   a' * b:\t" << timer.value() << endl;

      if (rows<500)
      {
        timer.reset();
        timer.start();
        for (int k=0; k<REPEAT; ++k)
          gmm::mult(gmm::transposed(m1), gmm::transposed(m2), gmmT3);
        timer.stop();
        std::cout << "   a' * b':\t" << timer.value() << endl;

        timer.reset();
        timer.start();
        for (int k=0; k<REPEAT; ++k)
          gmm::mult(m1, gmm::transposed(m2), gmmT3);
        timer.stop();
        std::cout << "   a * b':\t" << timer.value() << endl;
      }
      else
      {
        std::cout << "   a' * b':\t" << "forever" << endl;
        std::cout << "   a * b':\t" << "forever" << endl;
      }
    }
    #endif

    // MTL4
    #ifndef NOMTL
    {
      std::cout << "MTL4\t" << density*100 << "%\n";
      MtlSparse m1(rows,cols), m2(rows,cols), m3(rows,cols);
      eiToMtl(sm1, m1);
      eiToMtl(sm2, m2);

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        m3 = m1 * m2;
      timer.stop();
      std::cout << "   a * b:\t" << timer.value() << endl;

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        m3 = trans(m1) * m2;
      timer.stop();
      std::cout << "   a' * b:\t" << timer.value() << endl;

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        m3 = trans(m1) * trans(m2);
      timer.stop();
      std::cout << "  a' * b':\t" << timer.value() << endl;

      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        m3 = m1 * trans(m2);
      timer.stop();
      std::cout << "   a * b' :\t" << timer.value() << endl;
    }
    #endif

    std::cout << "\n\n";
  }

  return 0;
}

