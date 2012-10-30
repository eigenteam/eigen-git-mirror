// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* 
 
 * NOTE: This file is the modified version of [s,d,c,z]snode_bmod.c file in SuperLU 
 
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
#ifndef SPARSELU_SNODE_BMOD_H
#define SPARSELU_SNODE_BMOD_H
template <typename Scalar, typename Index>
int SparseLUBase<Scalar,Index>::LU_snode_bmod (const int jcol, const int fsupc, ScalarVector& dense, GlobalLU_t& glu)
{  
  /* lsub : Compressed row subscripts of ( rectangular supernodes )
   * xlsub :  xlsub[j] is the starting location of the j-th column in lsub(*)
   * lusup : Numerical values of the rectangular supernodes
   * xlusup[j] is the starting location of the j-th column in lusup(*)
   */
  int nextlu = glu.xlusup(jcol); // Starting location of the next column to add 
  int irow, isub; 
  // Process the supernodal portion of L\U[*,jcol]
  for (isub = glu.xlsub(fsupc); isub < glu.xlsub(fsupc+1); isub++)
  {
    irow = glu.lsub(isub); 
    glu.lusup(nextlu) = dense(irow);
    dense(irow) = 0;
    ++nextlu;
  }
  // Make sure the leading dimension is a multiple of the underlying packet size
  // so that fast fully aligned kernels can be enabled:
  {
    Index lda = nextlu-glu.xlusup(jcol);
    int offset = internal::first_multiple<Index>(lda, internal::packet_traits<Scalar>::size) - lda;
    if(offset)
    {
      glu.lusup.segment(nextlu,offset).setZero();
      nextlu += offset;
    }
  }
  glu.xlusup(jcol + 1) = nextlu; // Initialize xlusup for next column ( jcol+1 )
  
  if (fsupc < jcol ){
    int luptr = glu.xlusup(fsupc); // points to the first column of the supernode
    int nsupr = glu.xlsub(fsupc + 1) -glu.xlsub(fsupc); //Number of rows in the supernode
    int nsupc = jcol - fsupc; // Number of columns in the supernodal portion of L\U[*,jcol]
    int ufirst = glu.xlusup(jcol); // points to the beginning of column jcol in supernode L\U(jsupno)
    int lda = glu.xlusup(jcol+1) - ufirst;
    
    int nrow = nsupr - nsupc; // Number of rows in the off-diagonal blocks
    
    // Solve the triangular system for U(fsupc:jcol, jcol) with L(fspuc:jcol, fsupc:jcol)
    Map<Matrix<Scalar,Dynamic,Dynamic>,0,OuterStride<> > A( &(glu.lusup.data()[luptr]), nsupc, nsupc, OuterStride<>(lda) );
    VectorBlock<ScalarVector> u(glu.lusup, ufirst, nsupc);
    u = A.template triangularView<UnitLower>().solve(u); // Call the Eigen dense triangular solve interface
    
    // Update the trailing part of the column jcol U(jcol:jcol+nrow, jcol) using L(jcol:jcol+nrow, fsupc:jcol) and U(fsupc:jcol)
    new (&A) Map<Matrix<Scalar,Dynamic,Dynamic>,0,OuterStride<> > ( &(glu.lusup.data()[luptr+nsupc]), nrow, nsupc, OuterStride<>(lda) ); 
    VectorBlock<ScalarVector> l(glu.lusup, ufirst+nsupc, nrow); 
    l.noalias() -= A * u; 
  }
  return 0;
}
#endif