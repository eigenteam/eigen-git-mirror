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

/** Compute the column elimination tree of a sparse matrix
  * NOTE : The matrix is supposed to be in column-major format. 
  * 
  */
template<typename MatrixType>
int LU_sp_coletree(const MatrixType& mat, VectorXi& parent)
{
  int nc = mat.cols(); // Number of columns 
  int nr = mat.rows(); // Number of rows 
  
  VectorXi root(nc); // root of subtree of etree 
  root.setZero();
  VectorXi pp(nc); // disjoint sets 
  pp.setZero(); // Initialize disjoint sets 
  VectorXi firstcol(nr); // First nonzero column in each row 
  firstcol.setZero(); 
  
  //Compute firstcol[row]
  int row,col; 
  firstcol.setConstant(nc);  //for (row = 0; row < nr; firstcol(row++) = nc); 
  for (col = 0; col < nc; col++)
  {
    for (typename MatrixType::InnerIterator it(mat, col); it; ++it)
    { // Is it necessary to brows the whole matrix, the lower part should do the job ??
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
    pp(col) = cset = col; // Initially, each element is in its own set 
    root(cset) = col; 
    parent(col) = nc; 
    for (typename MatrixType::InnerIterator it(mat, col); it; ++it)
    { //  A sequence of interleaved find and union is performed 
      row = firstcol(it.row());
      if (row >= col) continue; 
      rset = internal::etree_find(row, pp); // Find the name of the set containing row
      rroot = root(rset);
      if (rroot != col) 
      {
        parent(rroot) = col; 
        pp(cset) = cset = rset; // Get the union of cset and rset 
        root(cset) = col; 
      }
    }
  }
  return 0;  
}

/** Find the root of the tree/set containing the vertex i : Use Path halving */ 
int etree_find (int i, VectorXi& pp)
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

/**
  * Post order a tree
  */
VectorXi TreePostorder(int n, VectorXi& parent)
{
  VectorXi first_kid, next_kid; // Linked list of children 
  VectorXi post; // postordered etree
  int postnum; 
  // Allocate storage for working arrays and results 
  first_kid.resize(n+1); 
  next_kid.setZero(n+1);
  post.setZero(n+1);
  
  // Set up structure describing children
  int v, dad; 
  first_kid.setConstant(-1); 
  for (v = n-1, v >= 0; v--) 
  {
    dad = parent(v);
    next_kid(v) = first_kid(dad); 
    first_kid(dad) = v; 
  }
  
  // Depth-first search from dummy root vertex #n
  postnum = 0; 
  internal::nr_etdfs(n, parent, first_kid, next_kid, post, postnum);
  return post; 
}
/** 
  * Depth-first search from vertex n.  No recursion.
  * This routine was contributed by Cédric Doucet, CEDRAT Group, Meylan, France.
*/
void nr_etdfs (int n, int *parent, int* first_kid, int *next_kid, int *post, int postnum)
{
  int current = n, first, next;
  while (postnum != n) 
  {
    // No kid for the current node
    first = first_kid(current);
    
    // no first kid for the current node
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
      if (postnum==n+1) return; 
      
      // Updating current node 
      current = next; 
    }
    else 
    {
      current = first; 
    }
  }
}

#endif