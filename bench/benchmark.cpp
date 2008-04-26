// g++ -O3 -DNDEBUG -DMATSIZE=<x> benchmark.cpp -o benchmark && time ./benchmark
#include <Eigen/Core>

#ifndef MATSIZE
#define MATSIZE 3
#endif

using namespace std;
USING_PART_OF_NAMESPACE_EIGEN

#ifndef REPEAT
#define REPEAT 40000000
#endif

#ifndef SCALAR
#define SCALAR double
#endif

int main(int argc, char *argv[])
{
    Matrix4i m1, m2, m3;
    m1.setRandom();
    m2.setConstant(2);
    int s1 = 2;
    m3 = m1;
    std::cout << m1 << "\n\n";
    std::cout << m2 << "\n\n";
    m3 = m1.cwiseProduct(m2);
    std::cout << m3 << "\n==\n" << m1*s1 << "\n\n";
//     v(1,2,3,4);
//     std::cout << v * 2 << "\n";
    Matrix<SCALAR,MATSIZE,MATSIZE> I = Matrix<SCALAR,MATSIZE,MATSIZE>::ones();
    Matrix<SCALAR,MATSIZE,MATSIZE> m;
    for(int i = 0; i < MATSIZE; i++)
        for(int j = 0; j < MATSIZE; j++)
        {
            m(i,j) = (i+MATSIZE*j);
        }
    asm("#begin");
    for(int a = 0; a < REPEAT; a++)
    {
        m = I + 0.00005 * (m + m*m);
    }
    asm("#end");
    cout << m << endl;
    return 0;
}
