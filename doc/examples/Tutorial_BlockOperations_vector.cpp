#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  VectorXf v(6);
  
  v << 1, 2, 3, 4, 5, 6;

  //print first three elements
  cout << "-- head(3) --" << endl
    << v.head(3) << endl << endl;
  
  //print last three elements
  cout << "-- tail(3) --" << endl
    << v.tail(3) << endl << endl;
    
  //print between 2nd and 5th elem. inclusive
  cout << "-- segment(1,4) --" << endl
    << v.segment(1,4) << endl;
}
