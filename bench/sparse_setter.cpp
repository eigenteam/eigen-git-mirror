
//g++ -O3 -g0 -DNDEBUG  sparse_product.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.005 -DSIZE=10000 && ./a.out
//g++ -O3 -g0 -DNDEBUG  sparse_product.cpp -I.. -I/home/gael/Coding/LinearAlgebra/mtl4/ -DDENSITY=0.05 -DSIZE=2000 && ./a.out
// -DNOGMM -DNOMTL -DCSPARSE
// -I /home/gael/Coding/LinearAlgebra/CSparse/Include/ /home/gael/Coding/LinearAlgebra/CSparse/Lib/libcsparse.a
#ifndef SIZE
#define SIZE 1000000
#endif

#ifndef NBPERROW
#define NBPERROW 24
#endif

#ifndef REPEAT
#define REPEAT 1
#endif

#ifndef NOGOOGLE
#define EIGEN_GOOGLEHASH_SUPPORT
#include <google/sparse_hash_map>
#endif

#include "BenchSparseUtil.h"


#define BENCH(X) \
  timer.reset(); \
  for (int _j=0; _j<NBTRIES; ++_j) { \
    timer.start(); \
    for (int _k=0; _k<REPEAT; ++_k) { \
        X  \
  } timer.stop(); }

typedef std::vector<Vector2i> Coordinates;
typedef std::vector<float> Values;

EIGEN_DONT_INLINE Scalar* setrand_eigen_gnu_hash(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_eigen_google_dense(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_eigen_google_sparse(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_ublas_mapped(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_ublas_coord(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_ublas_compressed(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_ublas_genvec(const Coordinates& coords, const Values& vals);
EIGEN_DONT_INLINE Scalar* setrand_mtl(const Coordinates& coords, const Values& vals);

int main(int argc, char *argv[])
{
  int rows = SIZE;
  int cols = SIZE;
  //float density = float(NBPERROW)/float(SIZE);
  
  BenchTimer timer;
  Coordinates coords;
  Values values;
  for (int i=0; i<cols*NBPERROW; ++i)
  {
    coords.push_back(Vector2i(ei_random<int>(0,rows-1),ei_random<int>(0,cols-1)));
    values.push_back(ei_random<Scalar>());
  }
  std::cout << "nnz = " << coords.size()  << "\n";

    // dense matrices
    #ifdef DENSEMATRIX
    {
      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        setrand_eigen_dense(coords,values);
      timer.stop();
      std::cout << "Eigen Dense\t" << timer.value() << "\n";
    }
    #endif

    // eigen sparse matrices
    {
      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        setrand_eigen_gnu_hash(coords,values);
      timer.stop();
      std::cout << "Eigen std::map\t" << timer.value() << "\n";
    }
    #ifndef NOGOOGLE
    {
      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        setrand_eigen_google_dense(coords,values);
      timer.stop();
      std::cout << "Eigen google dense\t" << timer.value() << "\n";
    }
    {
      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        setrand_eigen_google_sparse(coords,values);
      timer.stop();
      std::cout << "Eigen google sparse\t" << timer.value() << "\n";
    }
    #endif
    
    #ifndef NOUBLAS
    {
      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        setrand_ublas_mapped(coords,values);
      timer.stop();
      std::cout << "ublas mapped\t" << timer.value() << "\n";
    }
    {
      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        setrand_ublas_genvec(coords,values);
      timer.stop();
      std::cout << "ublas vecofvec\t" << timer.value() << "\n";
    }
    /*{
      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        setrand_ublas_compressed(coords,values);
      timer.stop();
      std::cout << "ublas comp\t" << timer.value() << "\n";
    }
    {
      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        setrand_ublas_coord(coords,values);
      timer.stop();
      std::cout << "ublas coord\t" << timer.value() << "\n";
    }*/
    #endif
    
    
    // MTL4
    #ifndef NOMTL
    {
      timer.reset();
      timer.start();
      for (int k=0; k<REPEAT; ++k)
        setrand_mtl(coords,values);
      timer.stop();
      std::cout << "MTL\t" << timer.value() << "\n";
    }
    #endif

  return 0;
}

EIGEN_DONT_INLINE Scalar* setrand_eigen_gnu_hash(const Coordinates& coords, const Values& vals)
{
  using namespace Eigen;
  SparseMatrix<Scalar> mat(SIZE,SIZE);
  {
    RandomSetter<SparseMatrix<Scalar>, StdMapTraits > setter(mat);
    for (int i=0; i<coords.size(); ++i)
    {
      setter(coords[i].x(), coords[i].y()) = vals[i];
    }
//     std::cout << "check mem\n"; getchar();
  }
  return 0;//&mat.coeffRef(coords[0].x(), coords[0].y());
}

#ifndef NOGOOGLE
EIGEN_DONT_INLINE Scalar* setrand_eigen_google_dense(const Coordinates& coords, const Values& vals)
{
  using namespace Eigen;
  SparseMatrix<Scalar> mat(SIZE,SIZE);
  {
    RandomSetter<SparseMatrix<Scalar>, GoogleDenseHashMapTraits> setter(mat);
    for (int i=0; i<coords.size(); ++i)
      setter(coords[i].x(), coords[i].y()) = vals[i];
//     std::cout << "check mem\n"; getchar();
  }
  return 0;//&mat.coeffRef(coords[0].x(), coords[0].y());
}

EIGEN_DONT_INLINE Scalar* setrand_eigen_google_sparse(const Coordinates& coords, const Values& vals)
{
  using namespace Eigen;
  SparseMatrix<Scalar> mat(SIZE,SIZE);
  {
    RandomSetter<SparseMatrix<Scalar>, GoogleSparseHashMapTraits> setter(mat);
    for (int i=0; i<coords.size(); ++i)
      setter(coords[i].x(), coords[i].y()) = vals[i];
//     std::cout << "check mem\n"; getchar();
  }
  return 0;//&mat.coeffRef(coords[0].x(), coords[0].y());
}
#endif

#ifndef NOUBLAS
EIGEN_DONT_INLINE Scalar* setrand_ublas_mapped(const Coordinates& coords, const Values& vals)
{
  using namespace boost;
  using namespace boost::numeric;
  using namespace boost::numeric::ublas;
  mapped_matrix<Scalar> aux(SIZE,SIZE);
  for (int i=0; i<coords.size(); ++i)
  {
    aux(coords[i].x(), coords[i].y()) = vals[i];
  }
//   std::cout << "check mem\n"; getchar();
  compressed_matrix<Scalar> mat(aux);
  return 0;// &mat(coords[0].x(), coords[0].y());
}
/*EIGEN_DONT_INLINE Scalar* setrand_ublas_coord(const Coordinates& coords, const Values& vals)
{
  using namespace boost;
  using namespace boost::numeric;
  using namespace boost::numeric::ublas;
  coordinate_matrix<Scalar> aux(SIZE,SIZE);
  for (int i=0; i<coords.size(); ++i)
  {
    aux(coords[i].x(), coords[i].y()) = vals[i];
  }
  compressed_matrix<Scalar> mat(aux);
  return 0;//&mat(coords[0].x(), coords[0].y());
}
EIGEN_DONT_INLINE Scalar* setrand_ublas_compressed(const Coordinates& coords, const Values& vals)
{
  using namespace boost;
  using namespace boost::numeric;
  using namespace boost::numeric::ublas;
  compressed_matrix<Scalar> mat(SIZE,SIZE);
  for (int i=0; i<coords.size(); ++i)
  {
    mat(coords[i].x(), coords[i].y()) = vals[i];
  }
  return 0;//&mat(coords[0].x(), coords[0].y());
}*/
EIGEN_DONT_INLINE Scalar* setrand_ublas_genvec(const Coordinates& coords, const Values& vals)
{
  using namespace boost;
  using namespace boost::numeric;
  using namespace boost::numeric::ublas;
  
//   ublas::vector<coordinate_vector<Scalar> > foo;
  generalized_vector_of_vector<Scalar, row_major, ublas::vector<coordinate_vector<Scalar> > > aux(SIZE,SIZE);
  for (int i=0; i<coords.size(); ++i)
  {
    aux(coords[i].x(), coords[i].y()) = vals[i];
  }
  compressed_matrix<Scalar,row_major> mat(aux);
  return 0;//&mat(coords[0].x(), coords[0].y());
}
#endif

#ifndef NOMTL
EIGEN_DONT_INLINE void setrand_mtl(const Coordinates& coords, const Values& vals);
#endif

