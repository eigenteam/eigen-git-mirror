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
 
 * NOTE: This file is the modified version of xpruneL.c file in SuperLU 
 
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
#ifndef SPARSELU_PRUNEL_H
#define SPARSELU_PRUNEL_H
/**
 * \brief Prunes the L-structure.
 *
 * It prunes the L-structure  of supernodes whose L-structure constains the current pivot row "pivrow"
 * 
 * 
 * \param jcol The current column of L
 * \param [in]perm_r Row permutation
 * \param [out]pivrow  The pivot row
 * \param nseg Number of segments ???
 * \param segrep 
 * \param repfnz
 * \param [out]xprune 
 * \param Glu Global LU data
 * 
 */
template <typename VectorType>
void SparseLU::LU_pruneL(const int jcol, const VectorXi& perm_r, const int pivrow, const int nseg, const VectorXi& segrep, VectorXi& repfnz, VectorXi& xprune, GlobalLU_t& Glu)
{
  // Initialize pointers 
  VectorXi& xsup = Glu.xsup; 
  VectorXi& supno = Glu.supno; 
  VectorXi& lsub = Glu.lsub; 
  VectorXi& xlsub = Glu.xlsub; 
  VectorType& lusup = Glu.lusup;
  VectorXi& xlusup = Glu.xlusup; 
  
  // For each supernode-rep irep in U(*,j]
  int jsupno = supno(jcol); 
  int i,irep,irep1; 
  bool movnum, do_prune = false; 
  int kmin, kmax, ktemp, minloc, maxloc; 
  for (i = 0; i < nseg; i++)
  {
    irep = segrep(i); 
    irep1 = irep + 1; 
    do_prune = false; 
    
    // Don't prune with a zero U-segment 
    if (repfnz(irep) == IND_EMPTY) continue; 
    
    // If a snode overlaps with the next panel, then the U-segment
    // is fragmented into two parts -- irep and irep1. We should let 
    // pruning occur at the rep-column in irep1s snode. 
    if (supno(irep) == supno(irep1) continue; // don't prune 
    
    // If it has not been pruned & it has a nonz in row L(pivrow,i)
    if (supno(irep) != jsupno )
    {
      if ( xprune (irep) >= xlsub(irep1)
      {
        kmin = xlsub(irep);
        kmax = xlsub(irep1) - 1; 
        for (krow = kmin; krow <= kmax; krow++)
        {
          if (lsub(krow) == pivrow) 
          {
            do_prune = true; 
            break; 
          }
        }
      }
      
      if (do_prune) 
      {
        // do a quicksort-type partition
        // movnum=true means that the num values have to be exchanged
        movnum = false; 
        if (irep == xsup(supno(irep)) ) // Snode of size 1 
          movnum = true; 
        
        while (kmin <= kmax)
        {
          if (perm_r(lsub(kmax)) == IND_EMPTY)
            kmax--; 
          else if ( perm_r(lsub(kmin)) != IND_EMPTY)
            kmin--;
          else 
          {
            // kmin below pivrow (not yet pivoted), and kmax
            // above pivrow: interchange the two suscripts
            ktemp = lsub(kmin);
            lsub(kmin) = lsub(kmax);
            lsub(kmax) = ktemp; 
            
            // If the supernode has only one column, then we 
            // only keep one set of subscripts. For any subscript
            // intercnahge performed, similar interchange must be 
            // done on the numerical values. 
            if (movnum) 
            {
              minloc = xlusup(irep) + ( kmin - xlsub(irep) ); 
              maxloc = xlusup(irep) + ( kmax - xlsub(irep) ); 
              std::swap(lusup(minloc), lusup(maxloc)); 
            }
            kmin++;
            kmax--;
          }
        } // end while 
        
        xprune(irep) = kmin; 
      } // end if do_prune 
    } // end pruning 
  } // End for each U-segment
}
#endif