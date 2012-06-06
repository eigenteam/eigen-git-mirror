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
             Index* rowind_colptr, Index* col_to_sup, Index* sup_to_col )
    {
      setInfos(m, n, nnz, nzval, nzval_colptr, rowind, rowind_colptr, col_to_sup, sup_to_col);
    }
    
    ~SuperNodalMatrix()
    {
      
    }
    /**
     * Set appropriate pointers for the lower triangular supernodal matrix
     * These infos are available at the end of the numerical factorization
     * FIXME This class will be modified such that it can be use in the course 
     * of the factorization.
     */
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
    
    /**
     * Number of rows
     */
    int rows()
    {
      return m_row;
    }
    
    /**
     * Number of columns
     */
    int cols()
    {
      return m_col;
    }
    
    /**
     * Return the array of nonzero values packed by column
     * 
     * The size is nnz
     */
    Scalar* valuePtr()
    {
      return m_nzval; 
    }
    
    /**
     * Return the pointers to the beginning of each column in \ref outerIndexPtr()
     */
    Index* colIndexPtr()
    {
      return m_nzval_colptr; 
    }
    
    /**
     * Return the array of compressed row indices of all supernodes
     */
    Index* rowIndex()
    {
      return m_rowind; 
    }
    /**
     * Return the location in \em rowvaluePtr() which starts each column
     */
    Index* rowIndexPtr()
    {
      return m_rowind_colptr; 
    }
    /** 
     * Return the array of column-to-supernode mapping 
     */
    Index colToSup()
    {
      return m_col_to_sup;       
    }
    /**
     * Return the array of supernode-to-column mapping
     */
    Index supToCol()
    {
      return m_sup_to_col;
    }
    
  
    class InnerIterator; 
    class SuperNodeIterator;
    
  protected:
    Index m_row; // Number of rows
    Index m_col; // Number of columns 
    Index m_nnz; // Number of nonzero values 
    Index m_nsuper; // Number of supernodes 
    Scalar* m_nzval; //array of nonzero values packed by column
    Index* m_nzval_colptr; //nzval_colptr[j] Stores the location in nzval[] which starts column j 
    Index* m_rowind; // Array of compressed row indices of rectangular supernodes
    Index* m_rowind_colptr; //rowind_colptr[j] stores the location in rowind[] which starts column j
    Index *m_col_to_sup; // col_to_sup[j] is the supernode number to which column j belongs
    Index *m_sup_to_col; //sup_to_col[s] points to the starting column of the s-th supernode
    
  private :
};

/**
  * \brief InnerIterator class to iterate over nonzero values in the triangular supernodal matrix
  * 
  */
template<typename Scalar, typename Index>
class SuperNodalMatrix::InnerIterator
{
  public:
     InnerIterator(const SuperNodalMatrix& mat, Index outer)
      : m_matrix(mat),
        m_outer(outer), 
        m_idval(mat.colIndexPtr()[outer]),
        m_startval(m_idval),
        m_endval(mat.colIndexPtr()[outer+1])
        m_idrow(mat.rowIndexPtr()[outer]),
        m_startidrow(m_idrow),
        m_endidrow(mat.rowIndexPtr()[outer+1])
    {}
    inline InnerIterator& operator++()
    { 
      m_idval++; 
      m_idrow++ ;
      return *this; 
    }
    inline Scalar value() const { return m_matrix.valuePtr()[m_idval]; }
    
    inline Scalar& valueRef() { return const_cast<Scalar&>(m_matrix.valuePtr()[m_idval]; }
    
    inline Index index() const { return m_matrix.rowIndex()[m_idrow]; }
    inline Index row() const { return index(); }
    inline Index col() const { return m_outer; }
    
    inline Index supIndex() const { return m_matrix.colToSup()[m_outer]; }
    
    inline operator bool() const 
    { 
      return ( (m_idval < m_endval) && (m_idval > m_startval) && 
                (m_idrow < m_endidrow) && (m_idrow > m_startidrow) ); 
    }
    
  protected:
        const SuperNodalMatrix& m_matrix; // Supernodal lower triangular matrix 
        const Index m_outer; // Current column 
        Index m_idval; //Index to browse the values in the current column
        const Index m_startval; // Start of the column value 
        const Index m_endval; // End of the column value 
        Index m_idrow;  //Index to browse the row indices 
        const Index m_startidrow; // Start of the row indices of the current column value
        const Index m_endidrow; // End of the row indices of the current column value
};
/**
 * \brief Iterator class to iterate over nonzeros Supernodes in the triangular supernodal matrix
 * 
 * The final goal is to use this class when dealing with supernodes during numerical factorization
 */
template<typename Scalar, typename Index>
class SuperNodalMatrix::SuperNodeIterator
{
  public: 
    SuperNodeIterator(const SuperNodalMatrix& mat)
    {
      
    }
    SuperNodeIterator(const SuperNodalMatrix& mat, Index supno)
    {
      
    }
    
    /*
     * Available Methods : 
     * Browse all supernodes (operator ++ )
     * Number of supernodes 
     * Columns of the current supernode
     * triangular matrix of the current supernode 
     * rectangular part of the current supernode
     */
  protected:
    const SuperNodalMatrix& m_matrix; // Supernodal lower triangular matrix 
    Index m_idsup;  // Index to browse all supernodes
    const Index m_nsuper; // Number of all supernodes
    Index m_startidsup; 
    Index m_endidsup; 
  
}; 
#endif