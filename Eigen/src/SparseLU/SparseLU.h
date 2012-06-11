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


#ifndef EIGEN_SPARSE_LU
#define EIGEN_SPARSE_LU

namespace Eigen {
  
template <typename _MatrixType>
class SparseLU;

#include <Ordering.h>
#include <SparseLU_Structs.h>
#include <SparseLU_Memory.h>
#include <SparseLU_Utils.h>
#include <SuperNodalMatrix.h>

#include <SparseLU_Coletree.h>
#include <SparseLU_heap_relax_snode.h>
#include <SparseLU_relax_snode.h>
/**
 * \ingroup SparseLU_Module
 * \brief Sparse supernodal LU factorization for general matrices
 * 
 * This class implements the supernodal LU factorization for general matrices. 
 * 
 * \tparam _MatrixType The type of the sparse matrix. It must be a column-major SparseMatrix<>
 */
template <typename _MatrixType>
class SparseLU
{
  public:
    typedef _MatrixType MatrixType; 
    typedef typename MatrixType::Scalar Scalar; 
    typedef typename MatrixType::Index Index; 
    typedef SparseMatrix<Scalar,ColMajor,Index> NCMatrix;
    typedef SuperNodalMatrix<Scalar, Index> SCMatrix; 
    typedef GlobalLU_t<Scalar, Index> LU_GlobalLU_t;
    typedef Matrix<Scalar,Dynamic,1> ScalarVector;
    typedef Matrix<Index,Dynamic,1> IndexVector;
    typedef PermutationMatrix<Dynamic, Dynamic, Index> PermutationType;
  public:
    SparseLU():m_isInitialized(true),m_symmetricmode(false),m_fact(DOFACT),m_diagpivotthresh(1.0)
    {
      initperfvalues(); 
    }
    SparseLU(const MatrixType& matrix):SparseLU()
    {
      compute(matrix);
    }
    
    ~SparseLU()
    {
      // Free all explicit dynamic pointers 
    }
    
    void analyzePattern (const MatrixType& matrix);
    void factorize (const MatrixType& matrix);
    
    /**
     * Compute the symbolic and numeric factorization of the input sparse matrix.
     * The input matrix should be in column-major storage. 
     */
    void compute (const MatrixType& matrix)
    {
      // Analyze 
      analyzePattern(matrix); 
      //Factorize
      factorize(matrix);
    }
    template<typename Rhs, typename Dest>
    bool SparseLU::_solve(const MatrixBase<Rhs> &b, MatrixBase<Dest> &dest) const
    
    /** Indicate that the pattern of the input matrix is symmetric */
    void isSymmetric(bool sym)
    {
      m_symmetricmode = sym;
    }
    
    /** Set the threshold used for a diagonal entry to be an acceptable pivot. */
    void diagPivotThresh(RealScalar thresh)
    {
      m_diagpivotthresh = thresh; 
    }
      
  
    /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const internal::solve_retval<SparseLU, Rhs> solve(const MatrixBase<Rhs>& b) const 
    {
      eigen_assert(m_factorizationIsOk && "SparseLU is not initialized."); 
      eigen_assert(rows()==b.rows()
                    && "SparseLU::solve(): invalid number of rows of the right hand side matrix b");
          return internal::solve_retval<SuperLUBase, Rhs>(*this, b.derived());
    }
    
  protected:
    // Functions 
    void initperfvalues(); 
    template <typename IndexVector, typename ScalarVector>
    int LU_snode_dfs(const int jcol, const int kcol, const IndexVector* asub, 
                               const IndexVector* colptr, IndexVector& xprune, IndexVector& marker, LU_GlobalLU_t& glu);
    
  template <typename Index, typename ScalarVector>
  int LU_dsnode_bmod (const Index jcol, const Index jsupno, const Index fsupc, 
                      ScalarVector& dense, ScalarVector& tempv, LU_GlobalLu_t& Glu);
    
    // Variables 
    mutable ComputationInfo m_info;
    bool m_isInitialized;
    bool m_factorizationIsOk;
    bool m_analysisIsOk;
    fact_t m_fact; 
    NCMatrix m_mat; // The input (permuted ) matrix 
    SCMatrix m_Lstore; // The lower triangular matrix (supernodal)
    NCMatrix m_Ustore; // The upper triangular matrix
    PermutationType m_perm_c; // Column permutation 
    PermutationType m_perm_r ; // Row permutation
    IndexVector m_etree; // Column elimination tree 
    
    ScalarVector m_work; // Scalar work vector 
    IndexVector m_iwork; //Index work vector 
    static LU_GlobalLU_t m_glu; // persistent data to facilitate multiple factors 
                               // should be defined as a class member
    // SuperLU/SparseLU options 
    bool m_symmetricmode;
    
    // values for performance 
    int m_panel_size; // a panel consists of at most <panel_size> consecutive columns
    int m_relax; // To control degree of relaxing supernodes. If the number of nodes (columns) 
                 // in a subtree of the elimination tree is less than relax, this subtree is considered 
                 // as one supernode regardless of the row structures of those columns
    int m_maxsuper; // The maximum size for a supernode in complete LU
    int m_rowblk; // The minimum row dimension for 2-D blocking to be used;
    int m_colblk; // The minimum column dimension for 2-D blocking to be used;
    int m_fillfactor; // The estimated fills factors for L and U, compared with A
    RealScalar m_diagpivotthresh; // Specifies the threshold used for a diagonal entry to be an acceptable pivot
    int m_nnzL, m_nnzU; // Nonzeros in L and U factors 
  
  private:
    // Copy constructor 
    SparseLU (SparseLU& ) {}
  
}; // End class SparseLU

/* Set the  default values for performance    */
void SparseLU::initperfvalues()
{
  m_panel_size = 12; 
  m_relax = 1; 
  m_maxsuper = 100; 
  m_rowblk = 200; 
  m_colblk = 60; 
  m_fillfactor = 20; 
}


/** 
 * Compute the column permutation to minimize the fill-in (file amd.c )
 * 
 *  - Apply this permutation to the input matrix - 
 * 
 *  - Compute the column elimination tree on the permuted matrix (file Eigen_Coletree.h)
 * 
 *  - Postorder the elimination tree and the column permutation (file Eigen_Coletree.h)
 * 
 */
template <typename MatrixType, typename OrderingType>
void SparseLU::analyzePattern(const MatrixType& mat)
{
  
  //TODO  It is possible as in SuperLU to compute row and columns scaling vectors to equilibrate the matrix mat.
  
  // Compute the fill-reducing ordering  
  // TODO Currently, the only  available ordering method is AMD. 
  
  OrderingType ord(mat); 
  m_perm_c = ord.get_perm(); 
  //FIXME Check the right semantic behind m_perm_c
  // that is, column j of mat goes to column m_perm_c(j) of mat * m_perm_c; 
  
 
  // Apply the permutation to the column of the input  matrix
  m_mat = mat * m_perm_c;  //FIXME Check if this is valid, check as well how to permute only the index
    
  // Compute the column elimination tree of the permuted matrix 
  if (m_etree.size() == 0)  m_etree.resize(m_mat.cols());
  LU_sp_coletree(m_mat, m_etree); 
    
  // In symmetric mode, do not do postorder here
  if (!m_symmetricmode) {
    IndexVector post, iwork; 
    // Post order etree
    LU_TreePostorder(m_mat.cols(), m_etree, post); 
      
    // Renumber etree in postorder 
    iwork.resize(n+1);
    for (i = 0; i < n; ++i) iwork(post(i)) = post(m_etree(i));
    m_etree = iwork;
    
    // Postmultiply A*Pc by post, i.e reorder the matrix according to the postorder of the etree
    PermutationType post_perm(post);
    //m_mat = m_mat * post_perm; // FIXME This should surely be in factorize()  
  
    // Composition of the two permutations
    m_perm_c = m_perm_c * post_perm;
  } // end postordering 
  
  m_analysisIsok = true; 
}

/** 
 *  - Numerical factorization 
 *  - Interleaved with the symbolic factorization 
 * \tparam MatrixType The type of the matrix, it should be a column-major sparse matrix
 * \return info where
 *  : successful exit
 *    = 0: successful exit
 *    > 0: if info = i, and i is
 *       <= A->ncol: U(i,i) is exactly zero. The factorization has
 *          been completed, but the factor U is exactly singular,
 *          and division by zero will occur if it is used to solve a
 *          system of equations.
 *       > A->ncol: number of bytes allocated when memory allocation
 *         failure occurred, plus A->ncol. If lwork = -1, it is
 *         the estimated amount of space needed, plus A->ncol.  
 */
template <typename MatrixType>
void  SparseLU::factorize(const MatrixType& matrix)
{
  
  eigen_assert(m_analysisIsok && "analyzePattern() should be called first"); 
  eigen_assert((matrix.rows() == matrix.cols()) && "Only for squared matrices");
  
  // Apply the column permutation computed in analyzepattern()
  m_mat = matrix * m_perm_c; 
  m_mat.makeCompressed(); 
  
  int m = m_mat.rows();
  int n = m_mat.cols();
  int nnz = m_mat.nonZeros();
  int maxpanel = m_panel_size * m;
  // Allocate storage common to the factor routines
  int lwork = 0;
  int info = LUMemInit(m, n, nnz, m_work, m_iwork, lwork, m_fillratio, m_panel_size, m_maxsuper, m_rowblk, m_glu); 
  if (info) 
  {
    std::cerr << "UNABLE TO ALLOCATE WORKING MEMORY\n\n" ;
    m_factorizationIsOk = false;
    return ; 
  }
  
  
  // Set up pointers for integer working arrays 
  int idx = 0; 
  VectorBlock<IndexVector> segrep(m_iwork, idx, m); 
  idx += m; 
  VectorBlock<IndexVector> parent(m_iwork, idx, m);
  idx += m;
  VectorBlock<IndexVector> xplore(m_iwork, idx, m); 
  idx += m; 
  VectorBlock<IndexVector> repnfnz(m_iwork, idx, maxpanel);
  idx += maxpanel;
  VectorBlock<IndexVector> panel_lsub(m_iwork, idx, maxpanel)
  idx += maxpanel; 
  VectorBlock<IndexVector> xprune(m_iwork, idx, n);
  idx += n; 
  VectorBlock<IndexVector> marker(m_iwork, idx, m * LU_NO_MARKER); 
  
  repfnz.setConstant(-1); 
  panel_lsub.setConstant(-1);
  
  // Set up pointers for scalar working arrays 
  VectorBlock<ScalarVector> dense(m_work, 0, maxpanel); 
  dense.setZero();
  VectorBlock<ScalarVector> tempv(m_work, maxpanel, LU_NUM_TEMPV(m, m_panel_size, m_maxsuper, m_rowblk) );
  tempv.setZero();
  
  // Setup Permutation vectors
  // Compute the inverse of perm_c
  PermutationType iperm_c (m_perm_c.inverse() );
  
  // Identify initial relaxed snodes
  IndexVector relax_end(n);
  if ( m_symmetricmode = true ) 
    internal::LU_heap_relax_snode(n, m_etree, m_relax, marker, relax_end);
  else
    internal::LU_relax_snode(n, m_etree, m_relax, marker, relax_end);
  
  m_perm_r.setConstant(-1);
  marker.setConstant(-1);
  
  IndexVector& xsup = m_glu.xsup; 
  IndexVector& supno = m_glu.supno; 
  IndexVector& xlsub = m_glu.xlsub;
  IndexVector& xlusup = m_glu.xlusup;
  IndexVector& xusub = m_glu.xusub;
  Index& nzlumax = m_glu.nzlumax; 
    
  supno(0) = IND_EMPTY; 
  xsup(0) = xlsub(0) = xusub(0) = xlusup(0) = 0;
  int panel_size = m_panel_size; 
  int wdef = m_panel_size; // upper bound on panel width
  
  // Work on one 'panel' at a time. A panel is one of the following :
  //  (a) a relaxed supernode at the bottom of the etree, or
  //  (b) panel_size contiguous columns, <panel_size> defined by the user
  register int jcol,kcol; 
  IndexVector panel_histo(n);
  Index nextu, nextlu, jsupno, fsupc, new_next;
  Index pivrow; // Pivotal row number in the original row matrix
  int nseg1; // Number of segments in U-column above panel row jcol
  int nseg; // Number of segments in each U-column 
  int irep,ir; 
  for (jcol = 0; jcol < n; )
  {
    if (relax_end(jcol) != IND_EMPTY) 
    { // Starting a relaxed node from jcol
      kcol = relax_end(jcol); // End index of the relaxed snode 
      
      // Factorize the relaxed supernode(jcol:kcol)
      // First, determine the union of the row structure of the snode 
      info = LU_snode_dfs(jcol, kcol, m_mat.innerIndexPtr(), m_mat.outerIndexPtr(), xprune, marker); 
      if ( info ) 
      {
        m_info = NumericalIssue; 
        m_factorizationIsOk = false; 
        std::cerr << "MEMORY ALLOCATION FAILED IN SNODE_DFS() \n";
        return; 
      }
      nextu = xusub(jcol); //starting location of column jcol in ucol
      nextlu = xlusup(jcol); //Starting location of column jcol in lusup (rectangular supernodes)
      jsupno = supno(jcol); // Supernode number which column jcol belongs to 
      fsupc = xsup(jsupno); //First column number of the current supernode
      new_next = nextlu + (xlsub(fsupc+1)-xlsub(fsupc)) * (kcol - jcol + 1);
      while (new_next > nzlumax ) 
      {
        mem = LUMemXpand<Scalar>(lusup, nzlumax, nextlu, LUSUP, m_glu);
        if (mem) 
        {
          std::cerr << "MEMORY ALLOCATION FAILED FOR L FACTOR \n"; 
          m_factorizationIsOk = false; 
          return; 
        }
      }
      
      // Now, left-looking factorize each column within the snode
      for (icol = jcol; icol<=kcol; icol++){
        xusub(icol+1) = nextu;
        // Scatter into SPA dense(*)
        for (typename MatrixType::InnerIterator it(m_mat, icol); it; ++it)
          dense(it.row()) = it.val();
        
        // Numeric update within the snode 
        LU_snode_bmod(icol, jsupno, fsupc, dense, tempv); 
        
        // Eliminate the current column 
        info = LU_pivotL(icol, m_diagpivotthresh, m_perm_r, m_iperm_c, pivrow, m_glu); 
        if ( info ) 
        {
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
      }
      jcol = icol; // The last column te be eliminated
    }
    else 
    { // Work on one panel of panel_size columns
      
      // Adjust panel size so that a panel won't overlap with the next relaxed snode. 
      panel_size = w_def;
      for (k = jcol + 1; k < std::min(jcol+panel_size, n); k++)
      {
        if (relax_end(k) != IND_EMPTY) 
        {
          panel_size = k - jcol; 
          break; 
        }
      }
      if (k == n) 
        panel_size = n - jcol; 
        
      // Symbolic outer factorization on a panel of columns 
      LU_panel_dfs(m, panel_size, jcol, m_mat, m_perm_r, nseg1, dense, panel_lsub, segrep, repfnz, xprune, marker, parent, xplore, m_glu); 
      
      // Numeric sup-panel updates in topological order 
      LU_panel_bmod(m, panel_size, jcol, nseg1, dense, tempv, segrep, repfnz, m_glu); 
      
      // Sparse LU within the panel, and below the panel diagonal 
      for ( jj = jcol, j< jcol + panel_size; jj++) 
      {
        k = (jj - jcol) * m; // Column index for w-wide arrays 
        
        nseg = nseg1; // begin after all the panel segments
        //Depth-first-search for the current column
        VectorBlock<IndexVector> panel_lsubk(panel_lsub, k, m); //FIXME
        VectorBlock<IndexVector> repfnz_k(repfnz, k, m); //FIXME 
        info = LU_column_dfs(m, jj, perm_r, nseg, panel_lsub(k), segrep, repfnz_k, xprune, marker, parent, xplore, m_glu); 
        if ( !info ) 
        {
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
        // Numeric updates to this column 
        VectorBlock<IndexVector> dense_k(dense, k, m); //FIXME 
        VectorBlock<IndexVector> segrep_k(segrep, nseg1, m) // FIXME Check the length
        info = LU_column_bmod(jj, (nseg - nseg1), dense_k, tempv, segrep_k, repfnz_k, jcol, m_glu); 
        if ( info ) 
        {
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
        
        // Copy the U-segments to ucol(*)
        //FIXME Check that repfnz_k, dense_k... have stored references to modified columns
        info = LU_copy_to_col(jj, nseg, segrep, repfnz_k, perm_r, dense_k, m_glu); 
        if ( info ) 
        {
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
        
        // Form the L-segment 
        info = LU_pivotL(jj, m_diagpivotthresh, m_perm_r, iperm_c, pivrow, m_glu);
        if ( info ) 
        {
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
        
        // Prune columns (0:jj-1) using column jj
        LU_pruneL(jj, m_perm_r, pivrow, nseg, segrep, repfnz_k, xprune, m_glu); 
        
        // Reset repfnz for this column 
        for (i = 0; i < nseg; i++)
        {
          irep = segrep(i); 
          repfnz(irep) = IND_EMPTY; 
        }
      } // end SparseLU within the panel  
      jcol += panel_size;  // Move to the next panel
    } // end else 
  } // end for -- end elimination 
  
  // Adjust row permutation in the case of rectangular matrices
  if (m > n ) 
  {
    k = 0; 
    for (i = 0; i < m; ++i)
    {
      if ( perm_r(i) == IND_EMPTY )
      {
        perm_r(i) = n + k; 
        ++k; 
      }
    }
  }
  // Count the number of nonzeros in factors 
  LU_countnz(n, xprune, m_nnzL, m_nnzU, m_glu); 
  // Apply permutation  to the L subscripts 
  LU_fixupL(n, m_perm_r, m_glu); 
  
  // Free work space iwork and work 
  //...
  
  // Create supernode matrix L 
  m_Lstore.setInfos(m, n, m_nnzL, Glu.lusup, Glu.xlusup, Glu.lsub, Glu.xlsub, Glu.supno; Glu.xsup); 
  // Create the column major upper sparse matrix  U
  new (&m_Ustore) Map<SparseMatrix<Scalar, ColumnMajor> > ( m, n, m_nnzU, Glu.xusub.data(), Glu.usub.data(), Glu.ucol.data() ); //FIXME 
  this.m_Ustore = m_Ustore; 
  
  m_info = Success;
  m_factorizationIsOk = ok;
}

template<typename Rhs, typename Dest>
bool SparseLU::_solve(const MatrixBase<Rhs> &b, MatrixBase<Dest> &x) const
{
  eigen_assert(m_isInitialized && "The matrix should be factorized first");
  EIGEN_STATIC_ASSERT((Dest::Flags&RowMajorBit)==0,
                     THIS_METHOD_IS_ONLY_FOR_COLUMN_MAJOR_MATRICES);
  
  x = b; /* on return, x is overwritten by the computed solution */
  
  int nrhs = b.cols(); 
  
  // Permute the right hand side to form Pr*B
  x = m_perm_r * x; 
  
  // Forward solve PLy = Pb; 
  Index fsupc; // First column of the current supernode 
  Index istart; // Pointer index to the subscript of the current column
  Index nsupr; // Number of rows in the current supernode
  Index nsupc; // Number of columns in the current supernode
  Index nrow; // Number of rows in the non-diagonal part of the supernode
  Index luptr; // Pointer index to the current nonzero value
  Index iptr; // row index pointer iterator
  Index irow; //Current index row
  Scalar * Lval = m_Lstore.valuePtr(); // Nonzero values 
  Matrix<Scalar,Dynamic,Dynamic> work(n,nrhs); // working vector
  work.setZero();
  int j; 
  for (k = 0; k <= m_Lstore.nsuper(); k ++)
  {
    fsupc = m_Lstore.sup_to_col()[k]; 
    istart = m_Lstore.rowIndexPtr()[fsupc]; 
    nsupr = m_Lstore..rowIndexPtr()[fsupc+1] - istart; 
    nsupc = m_Lstore.sup_to_col()[k+1] - fsupc; 
    nrow = nsupr - nsupc; 
    
    if (nsupc == 1 )
    {
      for (j = 0; j < nrhs; j++)
      {
        luptr = m_Lstore.colIndexPtr()[fsupc];  //FIXME Should be outside the for loop
        for (iptr = istart+1; iptr < m_Lstore.rowIndexPtr()[fsupc+1]; iptr++)
        {
          irow = m_Lstore.rowIndex()[iptr]; 
          ++luptr; 
          x(irow, j) -= x(fsupc, j) * Lval[luptr]; 
        }
      }
    }
    else 
    {
      // The supernode has more than one column 
      
      // Triangular solve 
      luptr = m_Lstore.colIndexPtr()[fsupc]; //FIXME Should be outside the loop 
      Map<Matrix<Scalar,Dynamic,Dynamic>, 0, OuterStride<> > A( &(Lval[luptr]), nsupc, nsupc, OuterStride<>(nsupr) ); 
//       Map<Matrix<Scalar,Dynamic,Dynamic>, 0, OuterStride > u( &(x(fsupc,0)), nsupc, nrhs, OuterStride<>(x.rows()) );
      Matrix<Scalar,Dynamic,Dynamic>& u = x.block(fsupc, 0, nsupc, nrhs); //FIXME Check this 
      u = A.triangularView<Lower>().solve(u); 
      
      // Matrix-vector product 
      new (&A) Map<Matrix<Scalar,Dynamic,Dynamic>, 0, OuterStride<> > ( &(Lval[luptr+nsupc]), nrow, nsupc, OuterStride<>(nsupr) ); 
      work.block(0, 0, nrow, nrhs) = A * u; 
      
      //Begin Scatter 
      for (j = 0; j < nrhs; j++)
      {
        iptr = istart + nsupc; 
        for (i = 0; i < nrow; i++)
        {
          irow = m_Lstore.rowIndex()[iptr]; 
          x(irow, j) -= work(i, j); // Scatter operation
          work(i, j) = Scalar(0); 
          iptr++;
        }
      }
    }
  } // end for all supernodes
  
  // Back solve Ux = y
  for (k = m_Lstore.nsuper(); k >= 0; k--)
  {
    fsupc = m_Lstore.sup_to_col()[k];
    istart = m_Lstore.rowIndexPtr()[fsupc];
    nsupr = m_Lstore..rowIndexPtr()[fsupc+1] - istart; 
    nsupc = m_Lstore.sup_to_col()[k+1] - fsupc; 
    luptr = m_Lstore.colIndexPtr()[fsupc]; 
    
    if (nsupc == 1)
    {
      for (j = 0; j < nrhs; j++)
      {
	x(fsupc, j) /= Lval[luptr]; 
      }
    }
    else 
    {
      Map<Matrix<Scalar,Dynamic,Dynamic>, 0, OuterStride<> > A( &(Lval[luptr]), nsupc, nsupc, OuterStride<>(nsupr) ); 
      Matrix<Scalar,Dynamic,Dynamic>& u = x.block(fsupc, 0, nsupc, nrhs); 
      u = A.triangularView<Upper>().solve(u); 
    }
    
    for (j = 0; j < nrhs; ++j)
    {
      for (jcol = fsupc; jcol < fsupc + nsupc; jcol++)
      {
	for (i = m_Ustore.outerIndexPtr()[jcol]; i < m_Ustore.outerIndexPtr()[jcol]; i++)
	{
	  irow = m_Ustore.InnerIndices()[i]; 
	  x(irow, j) -= x(irow, jcol) * m_Ustore.Values()[i];
	}
      }
    }
  } // End For U-solve
  
  // Permute back the solution 
  x = x * m_perm_c; 
  
  return true; 
}

namespace internal {
  
template<typename _MatrixType, typename Derived, typename Rhs>
struct solve_retval<SparseLU<_MatrixType,Derived>, Rhs>
  : solve_retval_base<SparseLU<_MatrixType,Derived>, Rhs>
{
  typedef SparseLU<_MatrixType,Derived> Dec;
  EIGEN_MAKE_SOLVE_HELPERS(Dec,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dec().derived()._solve(rhs(),dst);
  }
};

} // end namespace internal



} // End namespace Eigen 
#endif