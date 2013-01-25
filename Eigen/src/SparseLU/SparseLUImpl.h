// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SPARSELU_IMPL_H
#define SPARSELU_IMPL_H

namespace Eigen {
namespace internal {
  
/** \ingroup SparseLU_Module
  * \class SparseLUImpl
  * Base class for sparseLU
  */
template <typename Scalar, typename Index>
class SparseLUImpl
{
  public:
    typedef Matrix<Scalar,Dynamic,1> ScalarVector;
    typedef Matrix<Index,Dynamic,1> IndexVector; 
    typedef typename ScalarVector::RealScalar RealScalar; 
    typedef Ref<Matrix<Scalar,Dynamic,1> > BlockScalarVector;
    typedef Ref<Matrix<Index,Dynamic,1> > BlockIndexVector;
    typedef LU_GlobalLU_t<IndexVector, ScalarVector> GlobalLU_t; 
    typedef SparseMatrix<Scalar,ColMajor,Index> MatrixType; 
    
  protected:
     template <typename VectorType>
     int expand(VectorType& vec, int& length, int nbElts, int keep_prev, int& num_expansions);
     int memInit(int m, int n, int annz, int lwork, int fillratio, int panel_size,  GlobalLU_t& glu); 
     template <typename VectorType>
     int memXpand(VectorType& vec, int& maxlen, int nbElts, MemType memtype, int& num_expansions);
     void heap_relax_snode (const int n, IndexVector& et, const int relax_columns, IndexVector& descendants, IndexVector& relax_end); 
     void relax_snode (const int n, IndexVector& et, const int relax_columns, IndexVector& descendants, IndexVector& relax_end); 
     int snode_dfs(const int jcol, const int kcol,const MatrixType& mat,  IndexVector& xprune, IndexVector& marker, GlobalLU_t& glu); 
     int snode_bmod (const int jcol, const int fsupc, ScalarVector& dense, GlobalLU_t& glu);
     int pivotL(const int jcol, const RealScalar diagpivotthresh, IndexVector& perm_r, IndexVector& iperm_c, int& pivrow, GlobalLU_t& glu);
     template <typename Traits>
     void dfs_kernel(const int jj, IndexVector& perm_r,
                    int& nseg, IndexVector& panel_lsub, IndexVector& segrep,
                    Ref<IndexVector> repfnz_col, IndexVector& xprune, Ref<IndexVector> marker, IndexVector& parent,
                    IndexVector& xplore, GlobalLU_t& glu, int& nextl_col, int krow, Traits& traits);
     void panel_dfs(const int m, const int w, const int jcol, MatrixType& A, IndexVector& perm_r, int& nseg, ScalarVector& dense, IndexVector& panel_lsub, IndexVector& segrep, IndexVector& repfnz, IndexVector& xprune, IndexVector& marker, IndexVector& parent, IndexVector& xplore, GlobalLU_t& glu);
    
     void panel_bmod(const int m, const int w, const int jcol, const int nseg, ScalarVector& dense, ScalarVector& tempv, IndexVector& segrep, IndexVector& repfnz, GlobalLU_t& glu);
     int column_dfs(const int m, const int jcol, IndexVector& perm_r, int maxsuper, int& nseg,  BlockIndexVector lsub_col, IndexVector& segrep, BlockIndexVector repfnz, IndexVector& xprune, IndexVector& marker, IndexVector& parent, IndexVector& xplore, GlobalLU_t& glu);
     int column_bmod(const int jcol, const int nseg, BlockScalarVector dense, ScalarVector& tempv, BlockIndexVector segrep, BlockIndexVector repfnz, int fpanelc, GlobalLU_t& glu); 
     int copy_to_ucol(const int jcol, const int nseg, IndexVector& segrep, BlockIndexVector repfnz ,IndexVector& perm_r, BlockScalarVector dense, GlobalLU_t& glu); 
     void pruneL(const int jcol, const IndexVector& perm_r, const int pivrow, const int nseg, const IndexVector& segrep, BlockIndexVector repfnz, IndexVector& xprune, GlobalLU_t& glu);
     void countnz(const int n, int& nnzL, int& nnzU, GlobalLU_t& glu); 
     void fixupL(const int n, const IndexVector& perm_r, GlobalLU_t& glu); 
     
     template<typename , typename >
     friend struct column_dfs_traits;
}; 

} // end namespace internal
} // namespace Eigen

#endif
