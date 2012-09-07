// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/* 
 
 * NOTE: This file is the modified version of [s,d,c,z]snode_dfs.c file in SuperLU 
 
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
#ifndef SPARSELU_SNODE_DFS_H
#define SPARSELU_SNODE_DFS_H
  /**
   * \brief Determine the union of the row structures of those columns within the relaxed snode.
 *    NOTE: The relaxed snodes are leaves of the supernodal etree, therefore, 
 *    the portion outside the rectangular supernode must be zero.
 * 
 * \param jcol start of the supernode
 * \param kcol end of the supernode
 * \param asub Row indices
 * \param colptr Pointer to the beginning of each column
 * \param xprune (out) The pruned tree ??
 * \param marker (in/out) working vector
 * \return 0 on success, > 0 size of the memory when memory allocation failed
 */
  template <typename MatrixType, typename IndexVector, typename ScalarVector>
  int LU_snode_dfs(const int jcol, const int kcol,const MatrixType& mat,  IndexVector& xprune, IndexVector& marker, LU_GlobalLU_t<IndexVector, ScalarVector>& glu)
  {
    typedef typename IndexVector::Scalar Index; 
    int mem; 
    Index nsuper = ++glu.supno(jcol); // Next available supernode number
    int nextl = glu.xlsub(jcol); //Index of the starting location of the jcol-th column in lsub
    int krow,kmark; 
    for (int i = jcol; i <=kcol; i++)
    {
      // For each nonzero in A(*,i)
      for (typename MatrixType::InnerIterator it(mat, i); it; ++it)
      {
        krow = it.row(); 
        kmark = marker(krow);
        if ( kmark != kcol )
        {
          // First time to visit krow
          marker(krow) = kcol; 
          glu.lsub(nextl++) = krow; 
          if( nextl >= glu.nzlmax )
          {
            mem = LUMemXpand<IndexVector>(glu.lsub, glu.nzlmax, nextl, LSUB, glu.num_expansions);
            if (mem) return mem; // Memory expansion failed... Return the memory allocated so far
          }
        }
      }
      glu.supno(i) = nsuper;
    }
    
    // If supernode > 1, then make a copy of the subscripts for pruning
    if (jcol < kcol)
    {
      Index new_next = nextl + (nextl - glu.xlsub(jcol));
      while (new_next > glu.nzlmax)
      {
        mem = LUMemXpand<IndexVector>(glu.lsub, glu.nzlmax, nextl, LSUB, glu.num_expansions);
        if (mem) return mem; // Memory expansion failed... Return the memory allocated so far
      }
      Index ifrom, ito = nextl; 
      for (ifrom = glu.xlsub(jcol); ifrom < nextl;)
        glu.lsub(ito++) = glu.lsub(ifrom++);
      for (int i = jcol+1; i <=kcol; i++) glu.xlsub(i) = nextl;
      nextl = ito;
    }
    glu.xsup(nsuper+1) = kcol + 1; // Start of next available supernode
    glu.supno(kcol+1) = nsuper;
    xprune(kcol) = nextl;
    glu.xlsub(kcol+1) = nextl;
    return 0;
  }
#endif