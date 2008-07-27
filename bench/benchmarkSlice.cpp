// g++ -O3 -DNDEBUG benchmarkX.cpp -o benchmarkX && time ./benchmarkX

#include <Eigen/Array>

using namespace std;
USING_PART_OF_NAMESPACE_EIGEN

#ifndef REPEAT
#define REPEAT 10000
#endif

#ifndef SCALAR
#define SCALAR float
#endif

int main(int argc, char *argv[])
{
  typedef Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> Mat;
  Mat m(100, 100);
  m.setRandom();

  for(int a = 0; a < REPEAT; a++)
  {
    int r, c, nr, nc;
    r = Eigen::ei_random<int>(0,10);
    c = Eigen::ei_random<int>(0,10);
    nr = Eigen::ei_random<int>(50,80);
    nc = Eigen::ei_random<int>(50,80);
    m.block(r,c,nr,nc) += Mat::Ones(nr,nc);
    m.block(r,c,nr,nc) *= SCALAR(10);
    m.block(r,c,nr,nc) -= Mat::constant(nr,nc,10);
    m.block(r,c,nr,nc) /= SCALAR(10);
  }
  cout << m[0] << endl;
  return 0;
}
