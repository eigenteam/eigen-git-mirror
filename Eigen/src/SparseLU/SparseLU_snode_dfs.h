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
 
 * NOTE: This file is the modified version of dsnode_dfs.c file in SuperLU 
 
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
#ifdef EIGEN_SNODE_DFS_H
#define EIGEN_SNODE_DFS_H
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
 */
  template <typename Index>
  int SparseLU::LU_snode_dfs(const int jcol, const int kcol, const VectorXi* asub, const VectorXi* colptr, 
                    VectorXi& xprune, VectorXi& marker, LU_GlobalLu_t *m_Glu)
  {
    VectorXi& xsup = m_Glu->xsup; 
    VectorXi& supno = m_Glu->supno; // Supernode number corresponding to this column
    VectorXi& lsub = m_Glu->lsub;
    VectorXi& xlsub = m_Glu->xlsub;
    
    int nsuper = ++supno(jcol); // Next available supernode number
    register int nextl = xlsub(jcol); //Index of the starting location of the jcol-th column in lsub
    register int i,k; 
    int krow,kmark; 
    for (i = jcol; i <=kcol; i++)
    {
      // For each nonzero in A(*,i)
      for (k = colptr(i); k < colptr(i+1); k++)
      {
        krow = asub(k); 
        kmark = marker(krow);
        if ( kmark != kcol )
        {
          // First time to visit krow
          marker(krow) = kcol; 
          lsub(nextl++) = krow; 
          if( nextl >= nzlmax )
          {
            m_Glu->lsub = LUMemXpand<Index>(jcol, nextl, LSUB, nzlmax);
            m_Glu->nzlmax = nzlmax;
            lsub = m_Glu->lsub;
          }
        }
      }
      supno(i) = nsuper; 
    }
    
    // If supernode > 1, then make a copy of the subscripts for pruning
    if (jcol < kcol)
    {
      int new_next = nextl + (nextl - xlsub(jcol));
      while (new_next > nzlmax)
      {
        m_Glu->lsub = LUMemXpand<Index>(jcol, nextl, LSUB, &nzlmax);
        m_Glu->nzlmax= nzlmax;
        lsub = m_Glu->lsub;
      }
      int ifrom, ito = nextl; 
      for (ifrom = xlsub(jcol); ifrom < nextl;)
        lsub(ito++) = lsub(ifrom++);
      for (i = jcol+1; i <=kcol; i++)xlsub(i) = nextl;
      nextl = ito;
    }
    xsup(nsuper+1) = kcol + 1; // Start of next available supernode
    supno(kcol+1) = nsuper;
    xprune(kcol) = nextl;
    xlsub(kcol+1) = nextl;
    return 0;
  }
   
  
#endif