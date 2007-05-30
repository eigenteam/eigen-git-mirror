#include <iostream>

#include <tvmet/Matrix.h>

using namespace std;

int main()
{
  tvmet::Matrix<double,3,2>	B;
  tvmet::Matrix<double,3,3>	D;

  B =
   -0.05,  0,
    0,     0.05,
    0.05, -0.05;
  D =
    2000, 1000, 0,
    1000, 2000, 0,
    0,    0,    500;

  cout << "B = " << B << endl;
  cout << "D = " << D << endl;

  {
    tvmet::Matrix<double,2,2>	K;

    K = trans(B) * D * B;
    cout << "K = " << K << endl;
  }

  {
    tvmet::Matrix<double,2,2>	K;

    K = tvmet::Matrix<double,2,3>(trans(B) * D) * B;
    cout << "K = " << K << endl;
  }
}
