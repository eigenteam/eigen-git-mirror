 
// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012  Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
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

#ifndef EIGEN_ORDERING_H
#define EIGEN_ORDERING_H

#include "Amd.h"
namespace Eigen {
  
#include "Eigen_Colamd.h"

namespace internal {
    
/** \internal
  * \ingroup OrderingMethods_Module
  * \returns the symmetric pattern A^T+A from the input matrix A. 
  * FIXME: The values should not be considered here
  */
template<typename MatrixType> 
void ordering_helper_at_plus_a(const MatrixType& mat, MatrixType& symmat)
{
  MatrixType C;
  C = mat.transpose(); // NOTE: Could be  costly
  for (int i = 0; i < C.rows(); i++) 
  {
      for (typename MatrixType::InnerIterator it(C, i); it; ++it)
        it.valueRef() = 0.0;
  }
  symmat = C + mat; 
}
    
}

/** \ingroup OrderingMethods_Module
  * \class AMDOrdering
  *
  * Functor computing the \em approximate \em minimum \em degree ordering
  * If the matrix is not structurally symmetric, an ordering of A^T+A is computed
  * \tparam  Index The type of indices of the matrix 
  * \sa COLAMDOrdering
  */
template <typename Index>
class AMDOrdering
{
  public:
    typedef PermutationMatrix<Dynamic, Dynamic, Index> PermutationType;
    
    /** Compute the permutation vector from a sparse matrix
     * This routine is much faster if the input matrix is column-major     
     */
    template <typename MatrixType>
    void operator()(const MatrixType& mat, PermutationType& perm)
    {
      // Compute the symmetric pattern
      SparseMatrix<typename MatrixType::Scalar, ColMajor, Index> symm;
      internal::ordering_helper_at_plus_a(mat,symm); 
    
      // Call the AMD routine 
      //m_mat.prune(keep_diag());
      internal::minimum_degree_ordering(symm, perm);
    }
    
    /** Compute the permutation with a selfadjoint matrix */
    template <typename SrcType, unsigned int SrcUpLo> 
    void operator()(const SparseSelfAdjointView<SrcType, SrcUpLo>& mat, PermutationType& perm)
    { 
      SparseMatrix<typename SrcType::Scalar, ColMajor, Index> C; C = mat;
      
      // Call the AMD routine 
      // m_mat.prune(keep_diag()); //Remove the diagonal elements 
      internal::minimum_degree_ordering(C, perm);
    }
};

/** \ingroup OrderingMethods_Module
  * \class NaturalOrdering
  *
  * Functor computing the natural ordering (identity)
  * 
  * \note Returns an empty permutation matrix
  * \tparam  Index The type of indices of the matrix 
  */
template <typename Index>
class NaturalOrdering
{
  public:
    typedef PermutationMatrix<Dynamic, Dynamic, Index> PermutationType;
    
    /** Compute the permutation vector from a column-major sparse matrix */
    template <typename MatrixType>
    void operator()(const MatrixType& /*mat*/, PermutationType& perm)
    {
      perm.resize(0); 
    }
    
};

/** \ingroup OrderingMethods_Module
  * \class COLAMDOrdering
  *
  * Functor computing the \em column \em approximate \em minimum \em degree ordering 
  * The matrix should be in column-major format
  */
template<typename Index>
class COLAMDOrdering
{
  public:
    typedef PermutationMatrix<Dynamic, Dynamic, Index> PermutationType; 
    typedef Matrix<Index, Dynamic, 1> IndexVector;
    
    /** Compute the permutation vector form a sparse matrix */
    template <typename MatrixType>
    void operator() (const MatrixType& mat, PermutationType& perm)
    {
      int m = mat.rows();
      int n = mat.cols();
      int nnz = mat.nonZeros();
      // Get the recommended value of Alen to be used by colamd
      int Alen = internal::colamd_recommended(nnz, m, n); 
      // Set the default parameters
      double knobs [COLAMD_KNOBS]; 
      int stats [COLAMD_STATS];
      internal::colamd_set_defaults(knobs);
      
      int info;
      IndexVector p(n+1), A(Alen); 
      for(int i=0; i <= n; i++)   p(i) = mat.outerIndexPtr()[i];
      for(int i=0; i < nnz; i++)  A(i) = mat.innerIndexPtr()[i];
      // Call Colamd routine to compute the ordering 
      info = internal::colamd(m, n, Alen, A.data(), p.data(), knobs, stats); 
      eigen_assert( info && "COLAMD failed " );
      
      perm.resize(n);
      for (int i = 0; i < n; i++) perm.indices()(p(i)) = i;
    }
};

} // end namespace Eigen

#endif
