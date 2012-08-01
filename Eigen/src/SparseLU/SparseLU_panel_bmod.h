// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* 
 
 * NOTE: This file is the modified version of [s,d,c,z]panel_bmod.c file in SuperLU 
 
 * -- SuperLU routine (version 3.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
 *
 * Copyright (c) 1994 by Xerox Corporation.  All rights reserved.
 *
 * THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
 * EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.
 *
 * Permission is hereby granted to use or copy this program for any
 * purpose, provided the above notices are retained on all copies.
 * Permission to modify the code and to distribute modified code is
 * granted, provided the above notices are retained, and a notice that
 * the code was modified is included with the above copyright notice.
 */
#ifndef SPARSELU_PANEL_BMOD_H
#define SPARSELU_PANEL_BMOD_H
/**
 * \brief Performs numeric block updates (sup-panel) in topological order.
 * 
 * Before entering this routine, the original nonzeros in the panel
 * were already copied i nto the spa[m,w] ... FIXME to be checked
 * 
 * \param m number of rows in the matrix
 * \param w Panel size
 * \param jcol Starting  column of the panel
 * \param nseg Number of segments in the U part
 * \param dense Store the full representation of the panel 
 * \param tempv working array 
 * \param segrep segment representative... first row in the segment
 * \param repfnz First nonzero rows
 * \param glu Global LU data. 
 * 
 * 
 */
template <typename DenseIndexBlock, typename IndexVector, typename ScalarVector>
void LU_panel_bmod(const int m, const int w, const int jcol, const int nseg, ScalarVector& dense, ScalarVector& tempv, DenseIndexBlock& segrep, DenseIndexBlock& repfnz, LU_GlobalLU_t<IndexVector,ScalarVector>& glu)
{
  typedef typename ScalarVector::Scalar Scalar; 
  IndexVector& xsup = glu.xsup; 
  IndexVector& supno = glu.supno; 
  IndexVector& lsub = glu.lsub; 
  IndexVector& xlsub = glu.xlsub; 
  IndexVector& xlusup = glu.xlusup; 
  ScalarVector& lusup = glu.lusup; 
  
  int ksub,jj,nextl_col; 
  int fsupc, nsupc, nsupr, nrow; 
  int krep, kfnz; 
  int lptr; // points to the row subscripts of a supernode 
  int luptr; // ...
  int segsize,no_zeros ; 
  // For each nonz supernode segment of U[*,j] in topological order
  int k = nseg - 1; 
  for (ksub = 0; ksub < nseg; ksub++)
  { // For each updating supernode
  
    /* krep = representative of current k-th supernode
     * fsupc =  first supernodal column
     * nsupc = number of columns in a supernode
     * nsupr = number of rows in a supernode
     */
    krep = segrep(k); k--; 
    fsupc = xsup(supno(krep)); 
    nsupc = krep - fsupc + 1; 
    nsupr = xlsub(fsupc+1) - xlsub(fsupc); 
    nrow = nsupr - nsupc; 
    lptr = xlsub(fsupc); 
    // NOTE : Unlike the original implementation in SuperLU, the present implementation
    // does not include a 2-D block update. 
    
    // Sequence through each column in the panel
    for (jj = jcol; jj < jcol + w; jj++)
    {
      nextl_col = (jj-jcol) * m; 
      VectorBlock<IndexVector> repfnz_col(repfnz, nextl_col, m); // First nonzero column index for each row
      VectorBlock<ScalarVector> dense_col(dense, nextl_col, m); // Scatter/gather entire matrix column from/to here
      
      kfnz = repfnz_col(krep); 
      if ( kfnz == IND_EMPTY ) 
        continue; // skip any zero segment
      
      segsize = krep - kfnz + 1;
      luptr = xlusup(fsupc);    
      
      // NOTE : Unlike the original implementation in SuperLU, 
      // there is no update feature for  col-col, 2col-col ... 
      
      // Perform a trianglar solve and block update, 
      // then scatter the result of sup-col update to dense[]
      no_zeros = kfnz - fsupc; 
            if(segsize==1)  LU_kernel_bmod<1>::run(segsize, dense_col, tempv, lusup, luptr, nsupr, nrow, lsub, lptr, no_zeros);
      else  if(segsize==2)  LU_kernel_bmod<2>::run(segsize, dense_col, tempv, lusup, luptr, nsupr, nrow, lsub, lptr, no_zeros);
      else  if(segsize==3)  LU_kernel_bmod<3>::run(segsize, dense_col, tempv, lusup, luptr, nsupr, nrow, lsub, lptr, no_zeros);
      else                  LU_kernel_bmod<Dynamic>::run(segsize, dense_col, tempv, lusup, luptr, nsupr, nrow, lsub, lptr, no_zeros); 
    } // End for each column in the panel 
    
  } // End for each updating supernode
}
#endif