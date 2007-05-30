/*
 * This is a reused case study used as example from Rusell Edwards.
 * Works only with > release-1-2-0 due to a missed
 * product/operator*(Matrix,XprVector) on releases prior.
 * It shows a possible use of chained expressions.
 */
#include <iostream>

#include <tvmet/Matrix.h>
#include <tvmet/Vector.h>

using namespace std;
using namespace tvmet;

int main() {
  Matrix<float,3,3>		eigenvecs;
  Matrix<float,3,3>		M;

  eigenvecs = 1,2,3,4,5,6,7,8,9;
  M = 10,20,30,40,50,60,70,80,90;

  Vector<float,3> ev0( M * col(eigenvecs, 0));

  cout << "ev0 = " << ev0 << endl;
}
