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
 
 * NOTE: This file is the modified version of dpivotL.c file in SuperLU 
 
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
#ifndef SPARSELU_PIVOTL_H
#define SPARSELU_PIVOTL_H
/**
 * \brief Performs the numerical pivotin on the current column of L, and the CDIV operation.
 * 
 * Here is the pivot policy :
 * (1) 
 * 
 * \param jcol The current column of L
 * \param pivrow [out] The pivot row
 * 
 * 
 */
int SparseLU::LU_pivotL(const int jcol, Index& pivrow)
{
  // Initialize pointers 
  VectorXi& lsub = m_Glu.lsub; // Compressed row subscripts of ( rectangular supernodes ??)
  VectorXi& xlsub = m_Glu.xlsub; // xlsub[j] is the starting location of the j-th column in lsub(*)
  Scalar* lusup = m_Glu.lusup.data(); // Numerical values of the rectangular supernodes
  VectorXi& xlusup = m_Glu.xlusup; // xlusup[j] is the starting location of the j-th column in lusup(*)
  
  Index fsupc = (m_Glu.xsup)((m_Glu.supno)(jcol)); // First column in the supernode containing the column jcol
  Index nsupc = jcol - fsupc; // Number of columns in the supernode portion, excluding jcol; nsupc >=0
  Index lptr = xlsub(fsupc); // pointer to the starting location of the row subscripts for this supernode portion
  Index nsupr = xlsub(fsupc+1) - lptr; // Number of rows in the supernode
  Scalar* lu_sup_ptr = &(lusup[xlusup(fsupc)]); // Start of the current supernode
  Scalar* lu_col_ptr = &(lusup[xlusup(jcol)]); // Start of jcol in the supernode
  Index* lsub_ptr = &(lsub.data()[lptr]); // Start of row indices of the supernode
  
  // Determine the largest abs numerical value for partial pivoting 
  Index diagind = m_iperm_c(jcol); // diagonal index 
  Scalar pivmax = 0.0; 
  Index pivptr = nsupc; 
  Index diag = -1; 
  Index old_pivptr = nsupc; 
  Scalar rtemp;
  for (isub = nsupc; isub < nsupr; ++isub) {
    rtemp = std::abs(lu_col_ptr[isub]);
    if (rtemp > pivmax) {
      pivmax = rtemp; 
      pivptr = isub;
    } 
    if (lsub_ptr[isub] == diagind) diag = isub;
  }
  
  // Test for singularity
  if ( pivmax == 0.0 ) {
    pivrow = lsub_ptr[pivptr];
    m_perm_r(pivrow) = jcol;
    return (jcol+1);
  }
  
  Scalar thresh = m_diagpivotthresh * pivmax; 
  
  // Choose appropriate pivotal element 
  
  {
    // Test if the diagonal element can be used as a pivot (given the threshold value)
    if (diag >= 0 ) 
    {
      // Diagonal element exists
      rtemp = std::abs(lu_col_ptr[diag]);
      if (rtemp != Scalar(0.0) && rtemp >= thresh) pivptr = diag;
    }
    pivrow = lsub_ptr[pivptr];
  }
  
  // Record pivot row
  perm_r(pivrow) = jcol; 
  // Interchange row subscripts
  if (pivptr != nsupc )
  {
    std::swap( lsub_ptr(pivptr), lsub_ptr(nsupc) );
    // Interchange numerical values as well, for the two rows in the whole snode
    // such that L is indexed the same way as A
    for (icol = 0; icol <= nsupc; icol++)
    {
      itemp = pivptr + icol * nsupr; 
      std::swap(lu_sup_ptr[itemp], lu_sup_ptr[nsupc + icol * nsupr]);
    }
  }
  // cdiv operations
  Scalar temp = Scalar(1.0) / lu_col_ptr[nsupc];
  for (k = nsupc+1; k < nsupr; k++)
    lu_col_ptr[k] *= temp; 
  return 0;
}
#endif