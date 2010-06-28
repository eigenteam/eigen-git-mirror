#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  MatrixXf m(3,3), n(2,2);
  
  m << 1,2,3,
       4,5,6,
       7,8,9;
       
  // assignment through a block operation,
  //  block as rvalue
  n = m.block(0,0,2,2);
  
  //print n
  cout << "n = " << endl << n << endl << endl;
  
  
  n << 1,1,
       1,1;
        
  // block as lvalue
  m.block(0,0,2,2) = n;
  
  //print m
  cout << "m = " << endl << m << endl;
}
