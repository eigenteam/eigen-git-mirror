// Small bench routine for Eigen available in Eigen
// (C) Desire NUENTSA WAKAM, INRIA

#include <iostream>
#include <fstream>
#include <iomanip>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/SparseLU>

using namespace std;
using namespace Eigen;

int main(int argc, char **args)
{
  SparseMatrix<double, ColMajor> A; 
  typedef SparseMatrix<double, ColMajor>::Index Index;
  typedef Matrix<double, Dynamic, Dynamic> DenseMatrix;
  typedef Matrix<double, Dynamic, 1> DenseRhs;
  VectorXd b, x, tmp;
  SparseLU<SparseMatrix<double, ColMajor>, AMDOrdering<int> >   solver;
  ifstream matrix_file; 
  string line;
  int  n;
  
  // Set parameters
  /* Fill the matrix with sparse matrix stored in Matrix-Market coordinate column-oriented format */
  if (argc < 2) assert(false && "please, give the matrix market file ");
  loadMarket(A, args[1]);
  cout << "End charging matrix " << endl;
  bool iscomplex=false, isvector=false;
  int sym;
  getMarketHeader(args[1], sym, iscomplex, isvector);
  if (iscomplex) { cout<< " Not for complex matrices \n"; return -1; }
  if (isvector) { cout << "The provided file is not a matrix file\n"; return -1;}
  if (sym != 0) { // symmetric matrices, only the lower part is stored
    SparseMatrix<double, ColMajor> temp; 
    temp = A;
    A = temp.selfadjointView<Lower>();
  }
  n = A.cols();
  /* Fill the right hand side */

  if (argc > 2)
    loadMarketVector(b, args[2]);
  else 
  {
    b.resize(n);
    tmp.resize(n);
//       tmp.setRandom();
    for (int i = 0; i < n; i++) tmp(i) = i; 
    b = A * tmp ;
  }

  /* Compute the factorization */
//   solver.isSymmetric(true);
  solver.compute(A);
  
  solver._solve(b, x);
  /* Check the accuracy */
  VectorXd tmp2 = b - A*x;
  double tempNorm = tmp2.norm()/b.norm();
  cout << "Relative norm of the computed solution : " << tempNorm <<"\n";
  
  return 0;
}