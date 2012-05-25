// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

/* 
 
 * NOTE: This file is the modified version of dsnode_bmod.c file in SuperLU 
 
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
template <typename VectorType>
int SparseLU::LU_dsnode_bmod (const int jcol, const int jsupno, const int fsupc, 
                              VectorType& dense, VectorType& tempv)
{
  VectorXi& lsub = m_Glu.lsub; // Compressed row subscripts of ( rectangular supernodes ??)
  VectorXi& xlsub = m_Glu.xlsub; // xlsub[j] is the starting location of the j-th column in lsub(*)
  Scalar* lusup = m_Glu.lusup.data(); // Numerical values of the rectangular supernodes
  VectorXi& xlusup = m_Glu.xlusup; // xlusup[j] is the starting location of the j-th column in lusup(*)
  
  int nextlu = xlusup(jcol); // Starting location of the next column to add 
  int irow; 
  // Process the supernodal portion of L\U[*,jcol]
  for (int isub = xlsub(fsupc); isub < xlsub(fsupc+1); isub++)
  {
    irow = lsub(isub); 
    lusup(nextlu) = dense(irow);
    dense(irow) = 0;
    ++nextlu;
  }
  xlusup(jcol + 1) = nextlu; // Initialize xlusup for next column ( jcol+1 )
  
  if (fsupc < jcol ){
    int luptr = xlusup(fsupc); // points to the first column of the supernode
    int nsupr = xlsub(fsupc + 1) -xlsub(fsupc); //Number of rows in the supernode
    int nsupc = jcol - fsupc; // Number of columns in the supernodal portion of L\U[*,jcol]
    int ufirst = xlusup(jcol); // points to the beginning of column jcol in supernode L\U(jsupno)
    
    int nrow = nsupr - nsupc; // Number of rows in the off-diagonal blocks
    int incx = 1, incy = 1; 
    Scalar alpha = Scalar(-1.0); 
    Scalar beta = Scalar(1.0);
    // Solve the triangular system for U(fsupc:jcol, jcol) with L(fspuc..., fsupc:jcol)
    //BLASFUNC(trsv)("L", "N", "U", &nsupc, &(lusup[luptr]), &nsupr, &(lusup[ufirst]), &incx);
    Map<Matrix<Scalar,Dynamic,Dynamic>, 0, OuterStride<> > A( &(lusup[luptr]), nsupc, nsupc, OuterStride<>(nsupr) );
    Map<Matrix<Scalar,Dynamic,1> > l(&(lusup[ufirst]), nsupc);
    l = A.triangularView<Lower>().solve(l);
    // Update the trailing part of the column jcol U(jcol:jcol+nrow, jcol) using L(jcol:jcol+nrow, fsupc:jcol) and U(fsupc:jcol)
    BLASFUNC(gemv)("N", &nrow, &nsupc, &alpha, &lusup[luptr+nsupc], &nsupr, &lusup[ufirst], &incx, &beta, &lusup[ufirst+nsupc], &incy);
    
    return 0;
}
#endif