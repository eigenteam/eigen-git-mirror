// g++ -O3 -DNDEBUG benchmarkX.cpp -o benchmarkX && time ./benchmarkX

#include <Eigen/Core>

using namespace std;
USING_PART_OF_NAMESPACE_EIGEN

int main(int argc, char *argv[])
{
	MatrixXd I = MatrixXd::identity(20,20);
	MatrixXd m(20,20);
	for(int i = 0; i < 20; i++) for(int j = 0; j < 20; j++)
	{
		m(i,j) = 0.1 * (i+20*j);
	}
	for(int a = 0; a < 100000; a++)
	{
		m = I + 0.00005 * (m + m*m);
	}
	cout << m << endl;
	return 0;
}
