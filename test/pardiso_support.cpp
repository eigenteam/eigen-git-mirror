/* 
   Intel Copyright (C) ....
*/

#include "sparse_solver.h"
#include <Eigen/PARDISOSupport>

template<typename T> void test_pardiso_T()
{
  PardisoLLT < SparseMatrix<T, RowMajor> > pardiso_llt;
  PardisoLDLT< SparseMatrix<T, RowMajor> > pardiso_ldlt;
  PardisoLU  < SparseMatrix<T, RowMajor> > pardiso_lu;

  check_sparse_spd_solving(pardiso_llt);
  check_sparse_spd_solving(pardiso_ldlt);
  check_sparse_square_solving(pardiso_lu);
}

void test_pardiso_support()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(test_pardiso_T<float>());
    CALL_SUBTEST_2(test_pardiso_T<double>());
    CALL_SUBTEST_3(test_pardiso_T< std::complex<float> >());
    CALL_SUBTEST_4(test_pardiso_T< std::complex<double> >());
  }
}
