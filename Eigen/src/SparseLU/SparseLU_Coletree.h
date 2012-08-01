// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


/* 
 
 * NOTE: This file is the modified version of sp_coletree.c file in SuperLU 
 
 * -- SuperLU routine (version 3.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * August 1, 2008
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
#ifndef SPARSELU_COLETREE_H
#define SPARSELU_COLETREE_H
/** Find the root of the tree/set containing the vertex i : Use Path halving */ 
template<typename IndexVector>
int etree_find (int i, IndexVector& pp)
{
  int p = pp(i); // Parent 
  int gp = pp(p); // Grand parent 
  while (gp != p) 
  {
    pp(i) = gp; // Parent pointer on find path is changed to former grand parent
    i = gp; 
    p = pp(i);
    gp = pp(p);
  }
  return p; 
}

/** Compute the column elimination tree of a sparse matrix
  * NOTE : The matrix is supposed to be in column-major format. 
  * 
  */
template<typename MatrixType, typename IndexVector>
int LU_sp_coletree(const MatrixType& mat, IndexVector& parent)
{
  int nc = mat.cols(); // Number of columns 
  int nr = mat.rows(); // Number of rows 
  
  IndexVector root(nc); // root of subtree of etree 
  root.setZero();
  IndexVector pp(nc); // disjoint sets 
  pp.setZero(); // Initialize disjoint sets 
  IndexVector firstcol(nr); // First nonzero column in each row 
  
  //Compute first nonzero column in each row 
  int row,col; 
  firstcol.setConstant(nc);  //for (row = 0; row < nr; firstcol(row++) = nc); 
  for (col = 0; col < nc; col++)
  {
    for (typename MatrixType::InnerIterator it(mat, col); it; ++it)
    { // Is it necessary to browse the whole matrix, the lower part should do the job ??
      row = it.row();
      firstcol(row) = std::min(firstcol(row), col);
    }
  }
  /* Compute etree by Liu's algorithm for symmetric matrices,
          except use (firstcol[r],c) in place of an edge (r,c) of A.
    Thus each row clique in A'*A is replaced by a star
    centered at its first vertex, which has the same fill. */
  int rset, cset, rroot; 
  for (col = 0; col < nc; col++) 
  {
    pp(col) = col; 
    cset = col; 
    root(cset) = col; 
    parent(col) = nc; 
    for (typename MatrixType::InnerIterator it(mat, col); it; ++it)
    { //  A sequence of interleaved find and union is performed 
      row = firstcol(it.row());
      if (row >= col) continue; 
      rset = etree_find(row, pp); // Find the name of the set containing row
      rroot = root(rset);
      if (rroot != col) 
      {
        parent(rroot) = col; 
        pp(cset) = rset; 
        cset = rset; 
        root(cset) = col; 
      }
    }
  }
  return 0;  
}

/** 
  * Depth-first search from vertex n.  No recursion.
  * This routine was contributed by Cédric Doucet, CEDRAT Group, Meylan, France.
*/
template<typename IndexVector>
void LU_nr_etdfs (int n, IndexVector& parent, IndexVector& first_kid, IndexVector& next_kid, IndexVector& post, int postnum)
{
  int current = n, first, next;
  while (postnum != n) 
  {
    // No kid for the current node
    first = first_kid(current);
    
    // no kid for the current node
    if (first == -1) 
    {
      // Numbering this node because it has no kid 
      post(current) = postnum++;
      
      // looking for the next kid 
      next = next_kid(current); 
      while (next == -1) 
      {
        // No more kids : back to the parent node
        current = parent(current); 
        // numbering the parent node 
        post(current) = postnum++;
        
        // Get the next kid 
        next = next_kid(current); 
      }
      // stopping criterion 
      if (postnum == n+1) return; 
      
      // Updating current node 
      current = next; 
    }
    else 
    {
      current = first; 
    }
  }
}


/**
  * Post order a tree 
  * \param parent Input tree
  * \param post postordered tree
  */
template<typename IndexVector>
void LU_TreePostorder(int n, IndexVector& parent, IndexVector& post)
{
  IndexVector first_kid, next_kid; // Linked list of children 
  int postnum; 
  // Allocate storage for working arrays and results 
  first_kid.resize(n+1); 
  next_kid.setZero(n+1);
  post.setZero(n+1);
  
  // Set up structure describing children
  int v, dad; 
  first_kid.setConstant(-1); 
  for (v = n-1; v >= 0; v--) 
  {
    dad = parent(v);
    next_kid(v) = first_kid(dad); 
    first_kid(dad) = v; 
  }
  
  // Depth-first search from dummy root vertex #n
  postnum = 0; 
  LU_nr_etdfs(n, parent, first_kid, next_kid, post, postnum);
}

#endif