#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
   Matrix2d A;
   FullPivLU<Matrix2d> lu;
   A << 2, 1,
        2, 0.9999999999;
   lu.compute(A);
   cout << "By default, the rank of A is found to be " << lu.rank() << endl;
   cout << "Now recomputing the LU decomposition with threshold 1e-5" << endl;
   lu.setThreshold(1e-5);
   lu.compute(A);
   cout << "The rank of A is found to be " << lu.rank() << endl;
}
