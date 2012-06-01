// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>
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

#ifndef EIGEN_SPARSELU_MATRIX_H
#define EIGEN_SPARSELU_MATRIX_H

/** \ingroup SparseLU_Module
 * \brief a class to manipulate the L supernodal factor from the SparseLU factorization
 * 
 * This class  contain the data to easily store 
 * and manipulate the supernodes during the factorization and solution phase of Sparse LU. 
 * Only the lower triangular matrix has supernodes.
 * 
 * NOTE : This class corresponds to the SCformat structure in SuperLU
 * 
 */
/* TO DO
 * InnerIterator as for sparsematrix 
 * SuperInnerIterator to iterate through all supernodes 
 * Function for triangular solve
 */
template <typename _Scalar, typename _Index>
class SuperNodalMatrix
{
  public:
    typedef typename _Scalar Scalar; 
    typedef typename _Index Index; 
  public:
    SuperNodalMatrix()
    {
      
    }
    SuperNodalMatrix(Index m, Index n, Index nnz, Scalar *nzval, Index* nzval_colptr, Index* rowind, 
             Index* rowind_colptr, Index* col_to_sup, Index* sup_to_col ):m_row(m),m_col(n),m_nnz(nnz),
             m_nzval(nzval),m_nzval_colptr(nzval_colptr),m_rowind(rowind), 
             m_rowind_colptr(rowind_colptr),m_col_to_sup(col_to_sup),m_sup_to_col(sup_to_col)
    {
      
    }
    
    ~SuperNodalMatrix()
    {
      
    }
    void setInfos(Index m, Index n, Index nnz, Scalar *nzval, Index* nzval_colptr, Index* rowind, 
             Index* rowind_colptr, Index* col_to_sup, Index* sup_to_col )
    {
      m_row = m;
      m_col = n; 
      m_nnz = nnz; 
      m_nzval = nzval; 
      m_nzval_colptr = nzval_colptr; 
      m_rowind = rowind; 
      m_rowind_colptr = rowind_colptr; 
      m_col_to_sup = col_to_sup; 
      m_sup_to_col = sup_to_col; 
      
    }
    SuperNodalMatrix(SparseMatrix& mat); 
    
    class InnerIterator
    {
      public:
        
      protected:
        
    }: 
  protected:
    Index m_row; // Number of rows
    Index m_col; // Number of columns 
    Index m_nnz; // Number of nonzero values 
    Index m_nsupper; // Index of the last supernode
    Scalar* m_nzval; //array of nonzero values packed by (supernode ??) column
    Index* m_nzval_colptr; //nzval_colptr[j] Stores the location in nzval[] which starts column j 
    Index* m_rowind; // Array of compressed row indices of rectangular supernodes
    Index* m_rowind_colptr; //rowind_colptr[j] stores the location in rowind[] which starts column j
    Index *m_col_to_sup; // col_to_sup[j] is the supernode number to which column j belongs
    Index *m_sup_to_col; //sup_to_col[s] points to the starting column of the s-th supernode
    
  private :
    SuperNodalMatrix(SparseMatrix& ) {}
};

SuperNodalMatrix::SuperNodalMatrix(SparseMatrix& mat)
{
  
}
#endif