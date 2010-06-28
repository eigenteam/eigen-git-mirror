#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  MatrixXf m(4,4);
  
  m << 1, 2, 3, 4,
       5, 6, 7, 8,
       9, 10,11,12,
       13,14,15,16;

  //print first two columns
  cout << "-- leftCols(2) --" << endl
    << m.leftCols(2) << endl << endl;
  
  //print last two rows
  cout << "-- bottomRows(2) --" << endl
    << m.bottomRows(2) << endl << endl;
    
  //print top-left 2x3 corner
  cout << "-- topLeftCorner(2,3) --" << endl
    << m.topLeftCorner(2,3) << endl;
}
