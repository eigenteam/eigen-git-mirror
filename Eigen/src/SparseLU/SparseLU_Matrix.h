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
 * \brief a class to manipulate the supernodal matrices in the SparseLU factorization
 * 
 * This class extends the class SparseMatrix and should contain the data to easily store 
 * and manipulate the supernodes during the factorization and solution phase of Sparse LU. 
 * Only the lower triangular matrix has supernodes.
 * 
 * NOTE : This class corresponds to the SCformat structure in SuperLU
 * 
 */

template <typename _Scalar, typename _Index>
class SuperNodalMatrix
{
  public:
    SCMatrix()
    {
      
    }
    
    ~SCMatrix()
    {
      
    }
    operator SparseMatrix(); 
    
  protected:
    Index nnz; // Number of nonzero values 
    Index nsupper; // Index of the last supernode
    Scalar *nzval; //array of nonzero values packed by (supernode ??) column
    Index *nzval_colptr; //nzval_colptr[j] Stores the location in nzval[] which starts column j 
    Index *rowind; // Array of compressed row indices of rectangular supernodes
    Index rowind_colptr; //rowind_colptr[j] stores the location in rowind[] which starts column j
    Index *col_to_sup; // col_to_sup[j] is the supernode number to which column j belongs
    Index *sup_to_col; //sup_to_col[s] points to the starting column of the s-th supernode
    // Index *nzval_colptr corresponds to m_outerIndex in SparseMatrix
    
  private :
    SuperNodalMatrix(SparseMatrix& ) {}
};

SuperNodalMatrix::operator SparseMatrix() 
{
  
}
#endif