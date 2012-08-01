// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef EIGEN_SPARSELU_UTILS_H
#define EIGEN_SPARSELU_UTILS_H



template <typename IndexVector, typename ScalarVector>
void LU_countnz(const int n, int& nnzL, int& nnzU, LU_GlobalLU_t<IndexVector, ScalarVector>& glu)
{
 IndexVector& xsup = glu.xsup; 
 IndexVector& xlsub = glu.xlsub; 
 nnzL = 0; 
 nnzU = (glu.xusub)(n); 
 int nsuper = (glu.supno)(n); 
 int jlen; 
 int i, j, fsupc;
 if (n <= 0 ) return; 
 // For each supernode
 for (i = 0; i <= nsuper; i++)
 {
   fsupc = xsup(i); 
   jlen = xlsub(fsupc+1) - xlsub(fsupc); 
   
   for (j = fsupc; j < xsup(i+1); j++)
   {
     nnzL += jlen; 
     nnzU += j - fsupc + 1; 
     jlen--; 
   }
 }
 
}
/**
 * \brief Fix up the data storage lsub for L-subscripts. 
 * 
 * It removes the subscripts sets for structural pruning, 
 * and applies permutation to the remaining subscripts
 * 
 */
template <typename IndexVector, typename ScalarVector>
void LU_fixupL(const int n, const IndexVector& perm_r, LU_GlobalLU_t<IndexVector, ScalarVector>& glu)
{
  int fsupc, i, j, k, jstart; 
  IndexVector& xsup = glu.xsup; 
  IndexVector& lsub = glu.lsub; 
  IndexVector& xlsub = glu.xlsub; 
  
  int nextl = 0; 
  int nsuper = (glu.supno)(n); 
  
  // For each supernode 
  for (i = 0; i <= nsuper; i++)
  {
    fsupc = xsup(i); 
    jstart = xlsub(fsupc); 
    xlsub(fsupc) = nextl; 
    for (j = jstart; j < xlsub(fsupc + 1); j++)
    {
      lsub(nextl) = perm_r(lsub(j)); // Now indexed into P*A
      nextl++;
    }
    for (k = fsupc+1; k < xsup(i+1); k++)
      xlsub(k) = nextl; // other columns in supernode i
  }
  
  xlsub(n) = nextl; 
}

#endif
