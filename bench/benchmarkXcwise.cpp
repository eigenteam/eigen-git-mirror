// g++ -O3 -DNDEBUG benchmarkX.cpp -o benchmarkX && time ./benchmarkX

#include <Eigen/Core>

using namespace std;
USING_PART_OF_NAMESPACE_EIGEN

#ifndef MATTYPE
#define MATTYPE MatrixXLd
#endif

#ifndef MATSIZE
#define MATSIZE 400
#endif

#ifndef REPEAT
#define REPEAT 10000
#endif

int main(int argc, char *argv[])
{
	MATTYPE I = MATTYPE::ones(MATSIZE,MATSIZE);
	MATTYPE m(MATSIZE,MATSIZE);
	for(int i = 0; i < MATSIZE; i++) for(int j = 0; j < MATSIZE; j++)
	{
		m(i,j) = 0.1 * (i+j+1)/(MATSIZE*MATSIZE);
	}
	for(int a = 0; a < REPEAT; a++)
	{
		m = I + 0.00005 * (m + m/4);
	}
	cout << m(0,0) << endl;
	return 0;
}
