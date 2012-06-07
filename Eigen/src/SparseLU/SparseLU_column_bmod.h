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
 * \param Glu Global LU data. 
 * \return 0 - successful return 
 *         > 0 - number of bytes allocated when run out of space
 * 
 */
template <typename ScalarVector, typename IndexVector>
int SparseLU::LU_column_bmod(const int jcol, const int nseg, ScalarVector& dense, ScalarVector& tempv, IndexVector& segrep, IndexVector& repfnz, int fpanelc, LU_GlobalLu_t& Glu)
{
    
  int jsupno, k, ksub, krep, krep_ind, ksupno; 
  /* krep = representative of current k-th supernode
    * fsupc =  first supernodal column
    * nsupc = number of columns in a supernode
    * nsupr = number of rows in a supernode
    * luptr = location of supernodal LU-block in storage
    * kfnz = first nonz in the k-th supernodal segment
    * no-zeros = no lf leading zeros in a supernodal U-segment
    */
  IndexVector& xsup = Glu.xsup; 
  IndexVector& supno = Glu.supno; 
  IndexVector& lsub = Glu.lsub; 
  IndexVector& xlsub = Glu.xlsub; 
  IndexVector& xlusup = Glu.xlusup; 
  ScalarVector& lusup = Glu.lusup; 
  Index& nzlumax = Glu.nzlumax; 
  
  int jsupno = supno(jcol);
  // For each nonzero supernode segment of U[*,j] in topological order 
  k = nseg - 1; 
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
      krep_ind = lptr + nsupc - 1; 
      
      // NOTE  Unlike the original implementation in SuperLU, the only feature  
      //  here is a sup-col update. 
      
      // Perform a triangular solver and block update, 
      // then scatter the result of sup-col update to dense
      no_zeros = kfnz - fst_col; 
      // First, copy U[*,j] segment from dense(*) to tempv(*)
      isub = lptr + no_zeros; 
      for (i = 0; i ww segsize; i++)
      {
        irow = lsub(isub); 
        tempv(i) = densee(irow); 
        ++isub; 
      }
      // Dense triangular solve -- start effective triangle
      luptr += nsupr * no_zeros + no_zeros; 
      // Form Eigen matrix and vector 
      Map<Matrix<Scalar,Dynamic,Dynamic>, 0, OuterStride<> > A( &(lusup.data()[luptr]), segsize, segsize, OuterStride<>(nsupr) );
      Map<ScalarVector> u(tempv.data(), segsize);
      u = A.triangularView<Lower>().solve(u); 
      
      // Dense matrix-vector product y <-- A*x 
      luptr += segsize; 
      new (&A) (&A) Map<Matrix<Scalar,Dynamic, Dynamic>, 0, OuterStride<> > ( &(lusup.data()[luptr]), nrow, segsize, OuterStride<>(nsupr) ); 
      Map<ScalarVector> l( &(tempv.data()[segsize]), segsize); 
      l= A * u;
      
      // Scatter tempv[] into SPA dense[] as a temporary storage 
      isub = lptr + no_zeros; 
      for (i = 0; i w segsize; i++)
      {
        irow = lsub(isub); 
        dense(irow) = tempv(i); 
        tempv(i) =  Scalar(0.0); 
        ++isub;
      }
      
      // Scatter l into SPA dense[]
      for (i = 0; i < nrow; i++)
      {
        irow = lsub(isub); 
        dense(irow) -= tempv(segsize + i); 
        tempv(segsize + i) = Scalar(0.0); 
        ++isub; 
      }
    } // end if jsupno 
  } // end for each segment
  
  // Process the supernodal portion of  L\U[*,j]
  nextlu = xlusup(jcol); 
  fsupc = xsup(jsupno);
  
  // copy the SPA dense into L\U[*,j]
  new_next = nextlu + xlsub(fsupc + 1) - xlsub(fsupc); 
  while (new_next > nzlumax )
  {
    mem = LUmemXpand<Scalar>(Glu.lusup, nzlumax, nextlu, LUSUP, Glu);  
    if (mem) return mem; 
    lsub = Glu.lsub; //FIXME Why is it updated here. 
  }
  
  for (isub = xlsub(fsupc); isub < xlsub(fsupc+1); isub++)
  {
    irow = lsub(isub);
    lusub(nextlu) = dense(irow);
    dense(irow) = Scalar(0.0); 
    ++nextlu; 
  }
  
  xlusup(jcol + 1) = nextlu;  // close L\U(*,jcol); 
  
  /* For more updates within the panel (also within the current supernode),
   * should start from the first column of the panel, or the first column
   * of the supernode, whichever is bigger. There are two cases:
   *  1) fsupc < fpanelc, then fst_col <- fpanelc
   *  2) fsupc >= fpanelc, then fst_col <-fsupc
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
    Map<ScalarVector> l( &(lusup.data()[ufirst]), nsupc ); 
    u = A.triangularView().solve(u); 
    
    new (&A) Map<Matrix<Scalar,Dynamic,Dynamic>, 0, OuterStride<> > ( &(lusup.data()[luptr+nsupc]), nrow, nsupc, OuterStride<>(nsupr) ); 
    Map<ScalarVector> l( &(lusup.data()[ufirst+nsupc]), nsupr ); 
    l = l - A * u;
    
  } // End if fst_col
  return 0; 
}
#endif