// g++ -O3 -DNDEBUG benchmark.cpp -o benchmark && time ./benchmark
#include <cstdlib>
#include <cmath>
#include <Eigen/Core>

//using namespace std;
USING_PART_OF_NAMESPACE_EIGEN

int main(int argc, char *argv[])
{
	Matrix3d I;
	Matrix3d m;
	for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++)
	{
		I(i,j) = (i==j);
		m(i,j) = (i+3*j);
	}
	for(int a = 0; a < 400000000; a++)
	{
		m = I + 0.00005 * (m + m*m);
	}
	std::cout << m << std::endl;
	return 0;
}
