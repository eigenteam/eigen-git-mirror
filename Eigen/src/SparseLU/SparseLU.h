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
#include <SparseLU_Utils.h>
#include <SuperNodalMatrix.h>
#include <SparseLU_Structs.h>
#include <SparseLU_Memory.h>
#include <SparseLU_Coletree.h>

template <typename _MatrixType>
class SparseLU
{
  public:
    typedef _MatrixType MatrixType; 
    typedef typename MatrixType::Scalar Scalar; 
    typedef typename MatrixType::Index Index; 
    typedef SparseMatrix<Scalar,ColMajor,Index> NCMatrix;
    typedef SuperNodalMatrix<Scalar, Index> SCMatrix; 
    typedef GlobalLU_t<Scalar, Index> Eigen_GlobalLU_t;
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
      
    }
    
    void analyzePattern (const MatrixType& matrix);
    void factorize (const MatrixType& matrix);
    void compute (const MatrixType& matrix);
    
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
  protected:
    // Functions 
    void initperfvalues(); 
    
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
    PermutationType m_iperm_c; // Column permutation 
    PermutationType m_perm_r ; // Row permutation
    PermutationType m_iperm_r ; // Inverse row permutation
    IndexVector m_etree; // Column elimination tree 
    
    ScalarVector m_work; //
    IndexVector m_iwork; //
    static Eigen_GlobalLU_t m_Glu; // persistent data to facilitate multiple factors 
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
    int nnzL, nnzU; // Nonzeros in L and U factors 
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
 *  - Apply this permutation to the input matrix - 
 *  - Compute the column elimination tree on the permuted matrix (file Eigen_Coletree.h)
 *  - Postorder the elimination tree and the column permutation (file Eigen_Coletree.h)
 *  - 
 */
template <typename MatrixType>
void SparseLU::analyzePattern(const MatrixType& mat)
{
  // Compute the column permutation 
  AMDordering amd(mat); 
  m_perm_c = amd.get_perm_c(); 
  // Apply the permutation to the column of the input  matrix
  m_mat = mat * m_perm_c;  //how is the permutation represented ???
    
  // Compute the column elimination tree of the permuted matrix 
  if (m_etree.size() == 0)  m_etree.resize(m_mat.cols());
  internal::sp_coletree(m_mat, m_etree); 
    
  // In symmetric mode, do not do postorder here
  if (m_symmetricmode ==  false) {
    IndexVector post, iwork; 
    // Post order etree
    post = internal::TreePostorder(m_mat.cols(), m_etree); 
      
    // Renumber etree in postorder 
    iwork.resize(n+1);
    for (i = 0; i < n; ++i) iwork(post(i)) = post(m_etree(i));
    m_etree = iwork; 
      
    // Postmultiply A*Pc by post, 
    // i.e reorder the matrix according to the postorder of the etree
    // FIXME Check if this is available : constructor from a vector
    PermutationType post_perm(post); 
    m_mat = m_mat * post_perm; 
  
    // Product of m_perm_c  and post
    for (i = 0; i < n; ++i) iwork(i) = m_perm_c(post_perm.indices()(i));
    m_perm_c = iwork; 
  } // end postordering 
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
  
  // Allocate storage common to the factor routines
  int lwork = 0;
  int info = LUMemInit(lwork); 
  eigen_assert ( (info == 0) && "Unable to allocate memory for the factors"); 
  
  int m = m_mat.rows();
  int n = m_mat.cols();
  int maxpanel = m_panel_size * m;
  
  // Set up pointers for integer working arrays 
  Map<IndexVector> segrep(&m_iwork(0), m); //
  Map<IndexVector> parent(&segrep(0) + m, m); //
  Map<IndexVector> xplore(&parent(0) + m, m); //
  Map<IndexVector> repfnz(&xplore(0) + m, maxpanel); // 
  Map<IndexVector> panel_lsub(&repfnz(0) + maxpanel, maxpanel);//
  Map<IndexVector> xprune(&panel_lsub(0) + maxpanel, n); //
  Map<IndexVector> marker(&xprune(0)+n, m * LU_NO_MARKER); //
  repfnz.setConstant(-1); 
  panel_lsub.setConstant(-1);
  
  // Set up pointers for scalar working arrays 
  ScalarVector dense(maxpanel);
  dense.setZero();
  ScalarVector tempv(LU_NUM_TEMPV(m,m_panel_size,m_maxsuper,m_rowblk); 
  tempv.setZero();
  
  // Setup Permutation vectors
  PermutationType iperm_r; // inverse of perm_r
  if (m_fact = SamePattern_SameRowPerm)
    iperm_r = m_perm_r.inverse();
  // Compute the inverse of perm_c
  PermutationType iperm_c;
  iperm_c = m_perm_c.inverse();
  
  // Identify initial relaxed snodes
  IndexVector relax_end(n);
  if ( m_symmetricmode = true ) 
    LU_heap_relax_snode(n, m_etree, m_relax, marker, relax_end);
  else
    LU_relax_snode(n, m_etree, m_relax, marker, relax_end);
  
  m_perm_r.setConstant(-1);
  marker.setConstant(-1);
  
  IndexVector& xsup = m_Glu.xsup; 
  IndexVector& supno = m_GLu.supno; 
  IndexVector& xlsub = m_Glu.xlsub;
  IndexVector& xlusup = m_GLu.xlusup;
  IndexVector& xusub = m_Glu.xusub;
  Index& nzlumax = m_Glu.nzlumax; 
    
  supno(0) = IND_EMPTY; 
  xsup(0) = xlsub(0) = xusub(0) = xlusup(0);
  int panel_size = m_panel_size; 
  int wdef = panel_size; // upper bound on panel width
  
  // Work on one 'panel' at a time. A panel is one of the following :
  //  (a) a relaxed supernode at the bottom of the etree, or
  //  (b) panel_size contiguous columns, <panel_size> defined by the user
  register int jcol,kcol; 
  int min_mn = std::min(m,n);
  IndexVector panel_histo(n);
  Index nextu, nextlu, jsupno, fsupc, new_next;
  int pivrow; // Pivotal row number in the original row matrix
  int nseg1; // Number of segments in U-column above panel row jcol
  int nseg; // Number of segments in each U-column 
  int irep,ir; 
  for (jcol = 0; jcol < min_mn; )
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
        return; 
      }
      nextu = xusub(jcol); //starting location of column jcol in ucol
      nextlu = xlusup(jcol); //Starting location of column jcol in lusup (rectangular supernodes)
      jsupno = supno(jcol); // Supernode number which column jcol belongs to 
      fsupc = xsup(jsupno); //First column number of the current supernode
      new_next = nextlu + (xlsub(fsupc+1)-xlsub(fsupc)) * (kcol - jcol + 1);
      nzlumax = m_Glu.nzlumax;
      while (new_next > nzlumax ) 
      {
        mem = LUMemXpand<Scalar>(lusup, nzlumax, nextlu, LUSUP, m_Glu);
        if (mem) 
        {
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
        info = LU_pivotL(icol, m_diagpivotthresh, m_perm_r, m_iperm_c, pivrow, m_Glu); 
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
      for (k = jcol + 1; k < std::min(jcol+panel_size, min_mn); k++)
      {
        if (relax_end(k) != IND_EMPTY) 
        {
          panel_size = k - jcol; 
          break; 
        }
      }
      if (k == min_mn) 
        panel_size = min_mn - jcol; 
        
      // Symbolic outer factorization on a panel of columns 
      LU_panel_dfs(m, panel_size, jcol, m_mat, m_perm_r, nseg1, dense, panel_lsub, segrep, repfnz, xprune, marker, parent, xplore, m_Glu); 
      
      // Numeric sup-panel updates in topological order 
      LU_panel_bmod(m, panel_size, jcol, nseg1, dense, tempv, segrep, repfnz, m_Glu); 
      
      // Sparse LU within the panel, and below the panel diagonal 
      for ( jj = jcol, j< jcol + panel_size; jj++) 
      {
        k = (jj - jcol) * m; // Column index for w-wide arrays 
        
        nseg = nseg1; // begin after all the panel segments
        //Depth-first-search for the current column
        VectorBlock<IndexVector> panel_lsubk(panel_lsub, k, m); //FIXME
        VectorBlock<IndexVector> repfnz_k(repfnz, k, m); //FIXME 
        info = LU_column_dfs(m, jj, perm_r, nseg, panel_lsub(k), segrep, repfnz_k, xprune, marker, parent, xplore, m_Glu); 
        if ( !info ) 
        {
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
        // Numeric updates to this column 
        VectorBlock<IndexVector> dense_k(dense, k, m); //FIXME 
        VectorBlock<IndexVector> segrep_k(segrep, nseg1, m) // FIXME Check the length
        info = LU_column_bmod(jj, (nseg - nseg1), dense_k, tempv, segrep_k, repfnz_k, jcol, m_Glu); 
        if ( info ) 
        {
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
        
        // Copy the U-segments to ucol(*)
        //FIXME Check that repfnz_k, dense_k... have stored references to modified columns
        info = LU_copy_to_col(jj, nseg, segrep, repfnz_k, perm_r, dense_k, m_Glu); 
        if ( info ) 
        {
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
        
        // Form the L-segment 
        info = LU_pivotL(jj, m_diagpivotthresh, m_perm_r, iperm_c, pivrow, m_Glu);
        if ( info ) 
        {
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
        
        // Prune columns (0:jj-1) using column jj
        LU_pruneL(jj, m_perm_r, pivrow, nseg, segrep, repfnz_k, xprune, m_Glu); 
        
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
  LU_countnz(min_mn, xprune, m_nnzL, m_nnzU, m_Glu); 
  // Apply permutation  to the L subscripts 
  LU_fixupL(min_mn, m_perm_r, m_Glu); 
  
  // Free work space iwork and work 
  //...
  
  // Create supernode matrix L 
  m_Lstore.setInfos(m, min_mn, nnzL, Glu.lusup, Glu.xlusup, Glu.lsub, Glu.xlsub, Glu.supno; Glu.xsup); 
  // Create the column major upper sparse matrix  U
  // ?? Use the MappedSparseMatrix class ??
  new (&m_Ustore) Map<SparseMatrix<Scalar, ColumnMajor> > ( m, min_mn, nnzU, Glu.xusub.data(), Glu.usub.data(), Glu.ucol.data() ); 
  m_info = Success;
  m_factorizationIsOk = ok;
}


} // End namespace Eigen 
#endif