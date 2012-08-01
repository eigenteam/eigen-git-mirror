// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* 
 
 * NOTE: This file is the modified version of xcolumn_bmod.c file in SuperLU 
 
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
#ifndef SPARSELU_COLUMN_BMOD_H
#define SPARSELU_COLUMN_BMOD_H

/**
 * \brief Performs numeric block updates (sup-col) in topological order
 * 
 * \param jcol current column to update
 * \param nseg Number of segments in the U part
 * \param dense Store the full representation of the column
 * \param tempv working array 
 * \param segrep segment representative ...
 * \param repfnz ??? First nonzero column in each row ???  ...
 * \param fpanelc First column in the current panel
 * \param glu Global LU data. 
 * \return 0 - successful return 
 *         > 0 - number of bytes allocated when run out of space
 * 
 */
template <typename IndexVector, typename ScalarVector, typename BlockIndexVector, typename BlockScalarVector>
int LU_column_bmod(const int jcol, const int nseg, BlockScalarVector& dense, ScalarVector& tempv, BlockIndexVector& segrep, BlockIndexVector& repfnz, int fpanelc, LU_GlobalLU_t<IndexVector, ScalarVector>& glu)
{
  typedef typename IndexVector::Scalar Index; 
  typedef typename ScalarVector::Scalar Scalar; 
  int  jsupno, k, ksub, krep, ksupno; 
  int lptr, nrow, isub, irow, nextlu, new_next, ufirst; 
  int fsupc, nsupc, nsupr, luptr, kfnz, no_zeros; 
  /* krep = representative of current k-th supernode
    * fsupc =  first supernodal column
    * nsupc = number of columns in a supernode
    * nsupr = number of rows in a supernode
    * luptr = location of supernodal LU-block in storage
    * kfnz = first nonz in the k-th supernodal segment
    * no_zeros = no lf leading zeros in a supernodal U-segment
    */
  IndexVector& xsup = glu.xsup; 
  IndexVector& supno = glu.supno; 
  IndexVector& lsub = glu.lsub; 
  IndexVector& xlsub = glu.xlsub; 
  IndexVector& xlusup = glu.xlusup; 
  ScalarVector& lusup = glu.lusup; 
  Index& nzlumax = glu.nzlumax; 
  
  jsupno = supno(jcol);
  // For each nonzero supernode segment of U[*,j] in topological order 
  k = nseg - 1; 
  int d_fsupc; // distance between the first column of the current panel and the 
               // first column of the current snode
  int fst_col; // First column within small LU update
  int segsize; 
  for (ksub = 0; ksub < nseg; ksub++)
  {
    krep = segrep(k); k--; 
    ksupno = supno(krep); 
    if (jsupno != ksupno )
    {
      // outside the rectangular supernode 
      fsupc = xsup(ksupno); 
      fst_col = std::max(fsupc, fpanelc); 
      
      // Distance from the current supernode to the current panel; 
      // d_fsupc = 0 if fsupc > fpanelc
      d_fsupc = fst_col - fsupc; 
      
      luptr = xlusup(fst_col) + d_fsupc; 
      lptr = xlsub(fsupc) + d_fsupc; 
      
      kfnz = repfnz(krep); 
      kfnz = std::max(kfnz, fpanelc); 
      
      segsize = krep - kfnz + 1; 
      nsupc = krep - fst_col + 1; 
      nsupr = xlsub(fsupc+1) - xlsub(fsupc); 
      nrow = nsupr - d_fsupc - nsupc; 
      
      // NOTE  Unlike the original implementation in SuperLU, the only feature  
      // available here is a sup-col update. 
      
      // Perform a triangular solver and block update, 
      // then scatter the result of sup-col update to dense
      no_zeros = kfnz - fst_col; 
      if(segsize==1)
        LU_kernel_bmod<1>::run(segsize, dense, tempv, lusup, luptr, nsupr, nrow, lsub, lptr, no_zeros);
      else
        LU_kernel_bmod<Dynamic>::run(segsize, dense, tempv, lusup, luptr, nsupr, nrow, lsub, lptr, no_zeros);
    } // end if jsupno 
  } // end for each segment
  
  // Process the supernodal portion of  L\U[*,j]
  nextlu = xlusup(jcol); 
  fsupc = xsup(jsupno);
  
  // copy the SPA dense into L\U[*,j]
  int mem; 
  new_next = nextlu + xlsub(fsupc + 1) - xlsub(fsupc); 
  while (new_next > nzlumax )
  {
    mem = LUMemXpand<ScalarVector>(glu.lusup, nzlumax, nextlu, LUSUP, glu.num_expansions);  
    if (mem) return mem; 
  }
  
  for (isub = xlsub(fsupc); isub < xlsub(fsupc+1); isub++)
  {
    irow = lsub(isub);
    lusup(nextlu) = dense(irow);
    dense(irow) = Scalar(0.0); 
    ++nextlu; 
  }
  
  xlusup(jcol + 1) = nextlu;  // close L\U(*,jcol); 
  
  /* For more updates within the panel (also within the current supernode),
   * should start from the first column of the panel, or the first column
   * of the supernode, whichever is bigger. There are two cases:
   *  1) fsupc < fpanelc, then fst_col <-- fpanelc
   *  2) fsupc >= fpanelc, then fst_col <-- fsupc
   */
  fst_col = std::max(fsupc, fpanelc); 
  
  if (fst_col  < jcol)
  {
    // Distance between the current supernode and the current panel
    // d_fsupc = 0 if fsupc >= fpanelc
    d_fsupc = fst_col - fsupc; 
    
    lptr = xlsub(fsupc) + d_fsupc; 
    luptr = xlusup(fst_col) + d_fsupc; 
    nsupr = xlsub(fsupc+1) - xlsub(fsupc); // leading dimension
    nsupc = jcol - fst_col; // excluding jcol 
    nrow = nsupr - d_fsupc - nsupc; 
    
    // points to the beginning of jcol in snode L\U(jsupno) 
    ufirst = xlusup(jcol) + d_fsupc; 
    Map<Matrix<Scalar,Dynamic,Dynamic>, 0,  OuterStride<> > A( &(lusup.data()[luptr]), nsupc, nsupc, OuterStride<>(nsupr) ); 
    VectorBlock<ScalarVector> u(lusup, ufirst, nsupc); 
    u = A.template triangularView<UnitLower>().solve(u); 
    
    new (&A) Map<Matrix<Scalar,Dynamic,Dynamic>, 0, OuterStride<> > ( &(lusup.data()[luptr+nsupc]), nrow, nsupc, OuterStride<>(nsupr) ); 
    VectorBlock<ScalarVector> l(lusup, ufirst+nsupc, nrow); 
    l.noalias() -= A * u;
    
  } // End if fst_col
  return 0; 
}
#endif