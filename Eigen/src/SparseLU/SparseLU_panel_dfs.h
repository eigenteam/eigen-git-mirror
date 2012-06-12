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
 
 * NOTE: This file is the modified version of xpanel_dfs.c file in SuperLU 
 
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
#ifndef SPARSELU_PANEL_DFS_H
#define SPARSELU_PANEL_DFS_H
/**
 * \brief Performs a symbolic factorization on a panel of columns [jcol, jcol+w)
 * 
 * A supernode representative is the last column of a supernode.
 * The nonzeros in U[*,j] are segments that end at supernodes representatives
 * 
 * The routine returns a list of the supernodal representatives 
 * in topological order of the dfs that generates them. This list is 
 * a superset of the topological order of each individual column within 
 * the panel.
 * The location of the first nonzero in each supernodal segment 
 * (supernodal entry location) is also returned. Each column has 
 * a separate list for this purpose. 
 * 
 * Two markers arrays are used for dfs :
 *    marker[i] == jj, if i was visited during dfs of current column jj;
 *    marker1[i] >= jcol, if i was visited by earlier columns in this panel; 
 * 
 * \param [in]m number of rows in the matrix
 * \param [in]w Panel size
 * \param [in]jcol Starting  column of the panel
 * \param [in]A Input matrix in column-major storage
 * \param [in]perm_r Row permutation
 * \param [out]nseg Number of U segments
 * \param [out]dense Accumulate the column vectors of the panel
 * \param [out]panel_lsub Subscripts of the row in the panel 
 * \param [out]segrep Segment representative i.e first nonzero row of each segment
 * \param [out]repfnz First nonzero location in each row
 * \param [out]xprune 
 * \param [out]marker 
 * 
 * 
 */
template <typename MatrixType, typename IndexVector, typename ScalarVector>
void SparseLU::LU_panel_dfs(const int m, const int w, const int jcol, MatrixType& A, IndexVector& perm_r, int& nseg, ScalarVector& dense,  IndexVector& panel_lsub, IndexVector& segrep, IndexVector& repfnz, IndexVector& xprune, IndexVector& marker, IndexVector& parent, IndexVector& xplore, LU_GlobalLU_t& Glu)
{
  
  int jj; // Index through each column in the panel 
  int nextl_col; // Next available position in panel_lsub[*,jj] 
  int krow; // Row index of the current element 
  int kperm; // permuted row index
  int krep; // Supernode representative of the current row
  int kmark; 
  int chperm, chmark, chrep, oldrep, kchild; 
  int myfnz; // First nonzero element in the current column
  int xdfs, maxdfs, kpar;
  
  // Initialize pointers 
//   IndexVector& marker1 = marker.block(m, m); 
  VectorBlock<IndexVector> marker1(marker, m, m); 
  nseg = 0; 
  IndexVector& xsup = Glu.xsup; 
  IndexVector& supno = Glu.supno; 
  IndexVector& lsub = Glu.lsub; 
  IndexVector& xlsub = Glu.xlsub; 
  // For each column in the panel 
  for (jj = jcol; jj < jcol + w; jj++) 
  {
    nextl_col = (jj - jcol) * m; 
    
    VectorBlock<IndexVector> repfnz_col(repfnz, nextl_col, m); // First nonzero location in each row
    VectorBlock<IndexVector> dense_col(dense,nextl_col, m); // Accumulate a column vector here
    
    
    // For each nnz in A[*, jj] do depth first search
    for (MatrixType::InnerIterator it(A, jj); it; ++it)
    {
      krow = it.row(); 
      dense_col(krow) = it.val(); 
      kmark = marker(krow); 
      if (kmark == jj) 
        continue; // krow visited before, go to the next nonzero
      
      // For each unmarked krow of jj
      marker(krow) = jj; 
      kperm = perm_r(krow); 
      if (kperm == IND_EMPTY ) {
        // krow is in L : place it in structure of L(*, jj)
        panel_lsub(nextl_col++) = krow;  // krow is indexed into A
      }
      else 
      {
        // krow is in U : if its supernode-representative krep
        // has been explored, update repfnz(*)
        krep = xsup(supno(kperm)+1) - 1; 
        myfnz = repfnz_col(krep); 
        
        if (myfnz != IND_EMPTY )
        {
          // Representative visited before
          if (myfnz > kperm ) repfnz_col(krep) = kperm; 
          
        }
        else 
        {
          // Otherwise, perform dfs starting at krep
          oldrep = IND_EMPTY; 
          parent(krep) = oldrep; 
          repfnz_col(krep) = kperm; 
          xdfs =  xlsub(krep); 
          maxdfs = xprune(krep); 
          
          do 
          {
            // For each unmarked kchild of krep
            while (xdfs < maxdfs) 
            {
              kchild = lsub(xdfs); 
              xdfs++; 
              chmark = marker(kchild); 
              
              if (chmark != jj ) 
              {
                marker(kchild) = jj; 
                chperm = perm_r(kchild); 
                
                if (chperm == IND_EMPTY) 
                {
                  // case kchild is in L: place it in L(*, j)
                  panel_lsub(nextl_col++) = kchild; 
                }
                else
                {
                  // case kchild is in U :
                  // chrep = its supernode-rep. If its rep has been explored, 
                  // update its repfnz(*)
                  chrep = xsup(supno(chperm)+1) - 1; 
                  myfnz = repfnz_col(chrep); 
                  
                  if (myfnz != IND_EMPTY) 
                  { // Visited before 
                    if (myfnz > chperm) 
                      repfnz_col(chrep) = chperm; 
                  }
                  else 
                  { // Cont. dfs at snode-rep of kchild
                    xplore(krep) = xdfs; 
                    oldrep = krep; 
                    krep = chrep; // Go deeper down G(L)
                    parent(krep) = oldrep; 
                    repfnz_col(krep) = chperm; 
                    xdfs = xlsub(krep); 
                    maxdfs = xprune(krep); 
                    
                  } // end if myfnz != -1
                } // end if chperm == -1 
                    
              } // end if chmark !=jj
            } // end while xdfs < maxdfs
            
            // krow has no more unexplored nbrs :
            //    Place snode-rep krep in postorder DFS, if this 
            //    segment is seen for the first time. (Note that 
            //    "repfnz(krep)" may change later.)
            //    Baktrack dfs to its parent
            if (marker1(krep) < jcol )
            {
              segrep(nseg) = krep; 
              ++nseg; 
              marker1(krep) = jj; 
            }
            
            kpar = parent(krep); // Pop recursion, mimic recursion 
            if (kpar == IND_EMPTY) 
              break; // dfs done 
            krep = kpar; 
            xdfs = xplore(krep); 
            maxdfs = xprune(krep); 

          } while (kpar != IND_EMPTY); // Do until empty stack 
          
        } // end if (myfnz = -1)

      } // end if (kperm == -1) 

    }// end for nonzeros in column jj
    
  } // end for column jj
  
}
#endif