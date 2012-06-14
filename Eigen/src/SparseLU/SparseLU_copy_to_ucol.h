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
 
 * NOTE: This file is the modified version of xcopy_to_ucol.c file in SuperLU 
 
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
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
#ifndef SPARSELU_COPY_TO_UCOL_H
#define SPARSELU_COPY_TO_UCOL_H

/**
 * \brief Performs numeric block updates (sup-col) in topological order
 * 
 * \param jcol current column to update
 * \param nseg Number of segments in the U part
 * \param segrep segment representative ...
 * \param repfnz First nonzero column in each row  ...
 * \param perm_r Row permutation 
 * \param dense Store the full representation of the column
 * \param glu Global LU data. 
 * \return 0 - successful return 
 *         > 0 - number of bytes allocated when run out of space
 * 
 */
template <typename IndexVector, typename ScalarVector, typename SegRepType, typename RepfnzType, typename DenseType>
int LU_copy_to_ucol(const int jcol, const int nseg, SegRepType& segrep, RepfnzType& repfnz ,IndexVector& perm_r, DenseType& dense, LU_GlobalLU_t<IndexVector, ScalarVector>& glu)
{ 
  typedef typename IndexVector::Scalar Index; 
  typedef typename ScalarVector::Scalar Scalar; 
  Index ksub, krep, ksupno; 
  
  IndexVector& xsup = glu.xsup; 
  IndexVector& supno = glu.supno; 
  IndexVector& lsub = glu.lsub; 
  IndexVector& xlsub = glu.xlsub; 
  ScalarVector& ucol = glu.ucol; 
  IndexVector& usub = glu.usub; 
  IndexVector& xusub = glu.xusub;
  Index& nzumax = glu.nzumax; 
  
  Index jsupno = supno(jcol);
  
  // For each nonzero supernode segment of U[*,j] in topological order 
  int k = nseg - 1, i; 
  Index nextu = xusub(jcol); 
  Index kfnz, isub, segsize; 
  Index new_next,irow; 
  Index fsupc, mem; 
  for (ksub = 0; ksub < nseg; ksub++)
  {
    krep = segrep(k); k--; 
    ksupno = supno(krep); 
    if (jsupno != ksupno ) // should go into ucol(); 
    {
      kfnz = repfnz(krep); 
      if (kfnz != IND_EMPTY)
      { // Nonzero U-segment 
        fsupc = xsup(ksupno); 
        isub = xlsub(fsupc) + kfnz - fsupc; 
        segsize = krep - kfnz + 1; 
        new_next = nextu + segsize; 
        while (new_next > nzumax) 
        {
          mem = LUMemXpand<ScalarVector>(ucol, nzumax, nextu, UCOL, glu.num_expansions); 
          if (mem) return mem; 
          mem = LUMemXpand<IndexVector>(usub, nzumax, nextu, USUB, glu.num_expansions); 
          if (mem) return mem; 
          
        }
        
        for (i = 0; i < segsize; i++)
        {
          irow = lsub(isub); 
          usub(nextu) = perm_r(irow); // Unlike teh L part, the U part is stored in its final order
          ucol(nextu) = dense(irow); 
          dense(irow) = Scalar(0.0); 
          nextu++;
          isub++;
        }
        
      } // end nonzero U-segment 
      
    } // end if jsupno 
    
  } // end for each segment
  xusub(jcol + 1) = nextu; // close U(*,jcol)
  return 0; 
}

#endif