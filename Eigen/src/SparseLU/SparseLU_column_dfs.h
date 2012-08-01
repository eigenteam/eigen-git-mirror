// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* 
 
 * NOTE: This file is the modified version of [s,d,c,z]column_dfs.c file in SuperLU 
 
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
#ifndef SPARSELU_COLUMN_DFS_H
#define SPARSELU_COLUMN_DFS_H
/**
 * \brief Performs a symbolic factorization on column jcol and decide the supernode boundary
 * 
 * A supernode representative is the last column of a supernode.
 * The nonzeros in U[*,j] are segments that end at supernodes representatives. 
 * The routine returns a list of the supernodal representatives 
 * in topological order of the dfs that generates them. 
 * The location of the first nonzero in each supernodal segment 
 * (supernodal entry location) is also returned. 
 * 
 * \param m number of rows in the matrix
 * \param jcol Current column 
 * \param perm_r Row permutation
 * \param maxsuper 
 * \param [in,out] nseg Number of segments in current U[*,j] - new segments appended
 * \param lsub_col defines the rhs vector to start the dfs
 * \param [in,out] segrep Segment representatives - new segments appended 
 * \param repfnz
 * \param xprune 
 * \param marker
 * \param parent
 * \param xplore
 * \param glu global LU data 
 * \return 0 success
 *         > 0 number of bytes allocated when run out of space
 * 
 */
template <typename IndexVector, typename ScalarVector, typename BlockIndexVector>
int LU_column_dfs(const int m, const int jcol, IndexVector& perm_r, int maxsuper, int& nseg,  BlockIndexVector& lsub_col, IndexVector& segrep, BlockIndexVector& repfnz, IndexVector& xprune, IndexVector& marker, IndexVector& parent, IndexVector& xplore, LU_GlobalLU_t<IndexVector, ScalarVector>& glu)
{
  typedef typename IndexVector::Scalar Index; 
  typedef typename ScalarVector::Scalar Scalar; 
  
  int jsuper, nsuper, nextl; 
  int krow; // Row index of the current element 
  int kperm; // permuted row index
  int krep; // Supernode reprentative of the current row
  int k, kmark; 
  int chperm, chmark, chrep, oldrep, kchild; 
  int myfnz; // First nonzero element in the current column
  int xdfs, maxdfs, kpar;
  int mem; 
  // Initialize pointers 
  IndexVector& xsup = glu.xsup; 
  IndexVector& supno = glu.supno; 
  IndexVector& lsub = glu.lsub; 
  IndexVector& xlsub = glu.xlsub; 
  Index& nzlmax = glu.nzlmax; 
  
  int jcolm1 = jcol - 1; 
  int jcolp1 = jcol + 1;
  nsuper = supno(jcol); 
  jsuper = nsuper; 
  nextl = xlsub(jcol); 
  VectorBlock<IndexVector> marker2(marker, 2*m, m); 
  int fsupc, jptr, jm1ptr, ito, ifrom, istop; 
  // For each nonzero in A(*,jcol) do dfs 
  for (k = 0; lsub_col[k] != IND_EMPTY; k++) 
  {
    krow = lsub_col(k); 
    lsub_col(k) = IND_EMPTY; 
    kmark = marker2(krow); 
    
    // krow was visited before, go to the next nonz; 
    if (kmark == jcol) continue; 
    
    // For each unmarker nbr krow of jcol
    marker2(krow) = jcol; 
    kperm = perm_r(krow); 
    
    if (kperm == IND_EMPTY ) 
    {
      //  krow is in L: place it in structure of L(*,jcol)
      lsub(nextl++) = krow; // krow is indexed into A
      if ( nextl >= nzlmax )
      {
        mem = LUMemXpand<IndexVector>(lsub, nzlmax, nextl, LSUB, glu.num_expansions); 
        if ( mem ) return mem; 
      }
      if (kmark != jcolm1) jsuper = IND_EMPTY; // Row index subset testing
    }
    else 
    {
      // krow is in U : if its supernode-rep krep
      // has been explored, update repfnz(*)
      krep = xsup(supno(kperm)+1) - 1;
      myfnz = repfnz(krep); 
      
      if (myfnz != IND_EMPTY )
      {
        // visited before 
        if (myfnz > kperm) repfnz(krep) = kperm; 
        // continue; 
      }
      else 
      {
        // otherwise, perform dfs starting at krep
        oldrep = IND_EMPTY; 
        parent(krep) = oldrep; 
        repfnz(krep) = kperm; 
        xdfs = xlsub(krep); 
        maxdfs = xprune(krep); 
        
        do
        {
          // For each unmarked kchild of krep 
          while (xdfs < maxdfs)
          {
            kchild = lsub(xdfs); 
            xdfs++; 
            chmark = marker2(kchild); 
            
            if (chmark != jcol) 
            {
             // Not reached yet 
              marker2(kchild) = jcol; 
              chperm = perm_r(kchild); 
              
              if (chperm == IND_EMPTY)
              {
                // if kchild is in L: place it in L(*,k)
                lsub(nextl++) = kchild; 
                if (nextl >= nzlmax)
                {
                   mem = LUMemXpand<IndexVector>(lsub, nzlmax, nextl, LSUB, glu.num_expansions); 
                   if (mem) return mem; 
                }
                if (chmark != jcolm1) jsuper = IND_EMPTY; 
              } 
              else 
              {
                // if kchild is in U : 
                // chrep = its supernode-rep. If its rep has been explored, 
                // update its repfnz
                chrep = xsup(supno(chperm)+1) - 1; 
                myfnz = repfnz(chrep); 
                if (myfnz != IND_EMPTY) 
                {
                  // Visited before 
                  if ( myfnz > chperm) repfnz(chrep) = chperm; 
                }
                else 
                {
                  // continue dfs at super-rep of kchild 
                  xplore(krep) = xdfs; 
                  oldrep = krep; 
                  krep = chrep; // Go deeped down G(L^t)
                  parent(krep) = oldrep; 
                  repfnz(krep) = chperm; 
                  xdfs = xlsub(krep); 
                  maxdfs = xprune(krep); 
                } // else myfnz 
              } // else for chperm 
              
            } // if chmark 
            
          } // end while 
          
          // krow has no more unexplored nbrs; 
          // place supernode-rep krep in postorder DFS.
          // backtrack dfs to its parent
          
          segrep(nseg) = krep; 
          ++nseg; 
          kpar = parent(krep); // Pop from stack, mimic recursion
          if (kpar == IND_EMPTY) break; // dfs done 
          krep = kpar; 
          xdfs = xplore(krep); 
          maxdfs = xprune(krep); 
          
        } while ( kpar != IND_EMPTY); 
        
      } // else myfnz
      
    } // else kperm 
    
  } // for each nonzero ... 
  
  // check to see if j belongs in the same supernode as j-1
  if ( jcol == 0 )
  { // Do nothing for column 0 
    nsuper = supno(0) = 0 ;
  }
  else 
  {
    fsupc = xsup(nsuper); 
    jptr = xlsub(jcol); // Not yet compressed
    jm1ptr = xlsub(jcolm1); 
    
    // Use supernodes of type T2 : see SuperLU paper
    if ( (nextl-jptr != jptr-jm1ptr-1) ) jsuper = IND_EMPTY;
    
    // Make sure the number of columns in a supernode doesn't
    // exceed threshold
    if ( (jcol - fsupc) >= maxsuper) jsuper = IND_EMPTY; 
    
    /* If jcol starts a new supernode, reclaim storage space in
     * lsub from previous supernode. Note we only store 
     * the subscript set of the first and last columns of 
     * a supernode. (first for num values, last for pruning)
     */
    if (jsuper == IND_EMPTY)
    { // starts a new supernode 
      if ( (fsupc < jcolm1-1) ) 
      { // >= 3 columns in nsuper
        ito = xlsub(fsupc+1);
        xlsub(jcolm1) = ito; 
        istop = ito + jptr - jm1ptr; 
        xprune(jcolm1) = istop; // intialize xprune(jcol-1)
        xlsub(jcol) = istop; 
        
        for (ifrom = jm1ptr; ifrom < nextl; ++ifrom, ++ito)
          lsub(ito) = lsub(ifrom); 
        nextl = ito;  // = istop + length(jcol)
      }
      nsuper++; 
      supno(jcol) = nsuper; 
    } // if a new supernode 
  } // end else:  jcol > 0
  
  // Tidy up the pointers before exit
  xsup(nsuper+1) = jcolp1; 
  supno(jcolp1) = nsuper; 
  xprune(jcol) = nextl;  // Intialize upper bound for pruning
  xlsub(jcolp1) = nextl; 
  
  return 0; 
}
#endif