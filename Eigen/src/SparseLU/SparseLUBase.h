// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SPARSELUBASE_H
#define SPARSELUBASE_H
/**
 * Base class for sparseLU
 */
template <typename Scalar, typename Index>
struct SparseLUBase
{
  typedef Matrix<Scalar,Dynamic,1> ScalarVector;
  typedef Matrix<Index,Dynamic,1> IndexVector; 
  typedef typename ScalarVector::RealScalar RealScalar; 
  typedef VectorBlock<Matrix<Scalar,Dynamic,1> > BlockScalarVector;
  typedef VectorBlock<Matrix<Index,Dynamic,1> > BlockIndexVector;
//   typedef Ref<Matrix<Scalar,Dynamic,1> > BlockScalarVector;
//   typedef Ref<Matrix<Index,Dynamic,1> > BlockIndexVector;
  typedef LU_GlobalLU_t<IndexVector, ScalarVector> GlobalLU_t; 
  typedef SparseMatrix<Scalar,ColMajor,Index> MatrixType; 
  
  static int etree_find (int i, IndexVector& pp); 
  static int LU_sp_coletree(const MatrixType& mat, IndexVector& parent);
  static void LU_nr_etdfs (int n, IndexVector& parent, IndexVector& first_kid, IndexVector& next_kid, IndexVector& post, int postnum);
  static void LU_TreePostorder(int n, IndexVector& parent, IndexVector& post);
  template <typename VectorType>
  static int expand(VectorType& vec, int& length, int nbElts, int keep_prev, int& num_expansions);
  static int LUMemInit(int m, int n, int annz, int lwork, int fillratio, int panel_size,  GlobalLU_t& glu); 
  template <typename VectorType>
  static int LUMemXpand(VectorType& vec, int& maxlen, int nbElts, LU_MemType memtype, int& num_expansions);
  static void LU_heap_relax_snode (const int n, IndexVector& et, const int relax_columns, IndexVector& descendants, IndexVector& relax_end); 
  static void LU_relax_snode (const int n, IndexVector& et, const int relax_columns, IndexVector& descendants, IndexVector& relax_end); 
  static int LU_snode_dfs(const int jcol, const int kcol,const MatrixType& mat,  IndexVector& xprune, IndexVector& marker, LU_GlobalLU_t<IndexVector, ScalarVector>& glu); 
  static int LU_snode_bmod (const int jcol, const int fsupc, ScalarVector& dense, GlobalLU_t& glu);
  static int LU_pivotL(const int jcol, const RealScalar diagpivotthresh, IndexVector& perm_r, IndexVector& iperm_c, int& pivrow, GlobalLU_t& glu);
  template <typename Traits>
  static void LU_dfs_kernel(const int jj, IndexVector& perm_r,
                   int& nseg, IndexVector& panel_lsub, IndexVector& segrep,
                   Ref<IndexVector> repfnz_col, IndexVector& xprune, Ref<IndexVector> marker, IndexVector& parent,
                   IndexVector& xplore, GlobalLU_t& glu, int& nextl_col, int krow, Traits& traits);
  static void LU_panel_dfs(const int m, const int w, const int jcol, MatrixType& A, IndexVector& perm_r, int& nseg, ScalarVector& dense, IndexVector& panel_lsub, IndexVector& segrep, IndexVector& repfnz, IndexVector& xprune, IndexVector& marker, IndexVector& parent, IndexVector& xplore, GlobalLU_t& glu);
   
  static void LU_panel_bmod(const int m, const int w, const int jcol, const int nseg, ScalarVector& dense, ScalarVector& tempv, IndexVector& segrep, IndexVector& repfnz, LU_perfvalues& perfv, GlobalLU_t& glu);
  static int LU_column_dfs(const int m, const int jcol, IndexVector& perm_r, int maxsuper, int& nseg,  BlockIndexVector& lsub_col, IndexVector& segrep, BlockIndexVector& repfnz, IndexVector& xprune, IndexVector& marker, IndexVector& parent, IndexVector& xplore, GlobalLU_t& glu);
  static int LU_column_bmod(const int jcol, const int nseg, BlockScalarVector& dense, ScalarVector& tempv, BlockIndexVector& segrep, BlockIndexVector& repfnz, int fpanelc, GlobalLU_t& glu); 
  static int LU_copy_to_ucol(const int jcol, const int nseg, IndexVector& segrep, BlockIndexVector& repfnz ,IndexVector& perm_r, BlockScalarVector& dense, GlobalLU_t& glu); 
  static void LU_pruneL(const int jcol, const IndexVector& perm_r, const int pivrow, const int nseg, const IndexVector& segrep, BlockIndexVector& repfnz, IndexVector& xprune, GlobalLU_t& glu);
  static void LU_countnz(const int n, int& nnzL, int& nnzU, GlobalLU_t& glu); 
  static void LU_fixupL(const int n, const IndexVector& perm_r, GlobalLU_t& glu); 

}; 

#include "SparseLU_Coletree.h"
#include "SparseLU_Memory.h"
#include "SparseLU_heap_relax_snode.h"
#include "SparseLU_relax_snode.h"
#include "SparseLU_snode_dfs.h"
#include "SparseLU_snode_bmod.h"
#include "SparseLU_pivotL.h"
#include "SparseLU_panel_dfs.h"
#include "SparseLU_kernel_bmod.h"
#include "SparseLU_panel_bmod.h"
#include "SparseLU_column_dfs.h"
#include "SparseLU_column_bmod.h"
#include "SparseLU_copy_to_ucol.h"
#include "SparseLU_pruneL.h"
#include "SparseLU_Utils.h"

#endif
