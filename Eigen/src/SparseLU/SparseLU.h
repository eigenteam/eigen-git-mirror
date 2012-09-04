// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef EIGEN_SPARSE_LU_H
#define EIGEN_SPARSE_LU_H

namespace Eigen {


// Data structure needed by all routines 
#include "SparseLU_Structs.h"
#include "SparseLU_Matrix.h"

/**
 * \ingroup SparseLU_Module
 * \brief Sparse supernodal LU factorization for general matrices
 * 
 * This class implements the supernodal LU factorization for general matrices. 
 * 
 * \tparam _MatrixType The type of the sparse matrix. It must be a column-major SparseMatrix<>
 */
template <typename _MatrixType, typename _OrderingType>
class SparseLU
{
  public:
    typedef _MatrixType MatrixType; 
    typedef _OrderingType OrderingType;
    typedef typename MatrixType::Scalar Scalar; 
    typedef typename MatrixType::RealScalar RealScalar; 
    typedef typename MatrixType::Index Index; 
    typedef SparseMatrix<Scalar,ColMajor,Index> NCMatrix;
    typedef SuperNodalMatrix<Scalar, Index> SCMatrix; 
    typedef Matrix<Scalar,Dynamic,1> ScalarVector;
    typedef Matrix<Index,Dynamic,1> IndexVector;
    typedef PermutationMatrix<Dynamic, Dynamic, Index> PermutationType;
  public:
    SparseLU():m_isInitialized(true),m_Ustore(0,0,0,0,0,0),m_symmetricmode(false),m_diagpivotthresh(1.0)
    {
      initperfvalues(); 
    }
    SparseLU(const MatrixType& matrix):m_isInitialized(true),m_Ustore(0,0,0,0,0,0),m_symmetricmode(false),m_diagpivotthresh(1.0)
    {
      initperfvalues(); 
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
    
    inline Index rows() const { return m_mat.rows(); }
    inline Index cols() const { return m_mat.cols(); }
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
     
    /** Return the number of nonzero elements in the L factor */
    int nnzL()
    {
      if (m_factorizationIsOk)
        return m_nnzL; 
      else
      {
        std::cerr<<"Numerical factorization should be done before\n"; 
        return 0; 
      }
    }
    /** Return the number of nonzero elements in the U factor */
    int nnzU()
    {
      if (m_factorizationIsOk)
        return m_nnzU; 
      else
      {
        std::cerr<<"Numerical factorization should be done before\n"; 
        return 0; 
      }
    }
    /** \returns the solution X of \f$ A X = B \f$ using the current decomposition of A.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const internal::solve_retval<SparseLU, Rhs> solve(const MatrixBase<Rhs>& B) const 
    {
      eigen_assert(m_factorizationIsOk && "SparseLU is not initialized."); 
      eigen_assert(rows()==B.rows()
                    && "SparseLU::solve(): invalid number of rows of the right hand side matrix B");
          return internal::solve_retval<SparseLU, Rhs>(*this, B.derived());
    }

    
     /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was succesful,
      *          \c NumericalIssue if the PaStiX reports a problem
      *          \c InvalidInput if the input matrix is invalid
      *
      * \sa iparm()          
      */
    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_info;
    }

    template<typename Rhs, typename Dest>
    bool _solve(const MatrixBase<Rhs> &B, MatrixBase<Dest> &_X) const
    {
      Dest& X(_X.derived());
      eigen_assert(m_factorizationIsOk && "The matrix should be factorized first");
      EIGEN_STATIC_ASSERT((Dest::Flags&RowMajorBit)==0,
                        THIS_METHOD_IS_ONLY_FOR_COLUMN_MAJOR_MATRICES);
      
      
      int nrhs = B.cols(); 
      Index n = B.rows(); 
      
      // Permute the right hand side to form X = Pr*B
      // on return, X is overwritten by the computed solution
      X.resize(n,nrhs);
      for(int j = 0; j < nrhs; ++j)
        X.col(j) = m_perm_r * B.col(j); 
      
      //Forward substitution with L 
      m_Lstore.solveInPlace(X);
      
      // Backward solve with U
      for (int k = m_Lstore.nsuper(); k >= 0; k--)
      {
        Index fsupc = m_Lstore.supToCol()[k];
        Index istart = m_Lstore.rowIndexPtr()[fsupc];
        Index nsupr = m_Lstore.rowIndexPtr()[fsupc+1] - istart; 
        Index nsupc = m_Lstore.supToCol()[k+1] - fsupc; 
        Index luptr = m_Lstore.colIndexPtr()[fsupc]; 
        
        if (nsupc == 1)
        {
          for (int j = 0; j < nrhs; j++)
          {
            X(fsupc, j) /= m_Lstore.valuePtr()[luptr]; 
          }
        }
        else 
        {
          Map<const Matrix<Scalar,Dynamic,Dynamic>, 0, OuterStride<> > A( &(m_Lstore.valuePtr()[luptr]), nsupc, nsupc, OuterStride<>(nsupr) ); 
          Map< Matrix<Scalar,Dynamic,Dynamic>, 0, OuterStride<> > U (&(X(fsupc,0)), nsupc, nrhs, OuterStride<>(n) ); 
          U = A.template triangularView<Upper>().solve(U); 
        }
        
        for (int j = 0; j < nrhs; ++j)
        {
          for (int jcol = fsupc; jcol < fsupc + nsupc; jcol++)
          {
            typename MappedSparseMatrix<Scalar>::InnerIterator it(m_Ustore, jcol);
            for ( ; it; ++it)
            {
              Index irow = it.index(); 
              X(irow, j) -= X(jcol, j) * it.value();
            }
          }
        }
      } // End For U-solve
      
      // Permute back the solution 
      for (int j = 0; j < nrhs; ++j)
        X.col(j) = m_perm_c.inverse() * X.col(j); 
      
      return true; 
    }

  protected:
    // Functions 
    void initperfvalues()
    {
      m_perfv.panel_size = 12; 
      m_perfv.relax = 6; 
      m_perfv.maxsuper = 100; 
      m_perfv.rowblk = 200; 
      m_perfv.colblk = 60; 
      m_perfv.fillfactor = 20;  
    }
      
    // Variables 
    mutable ComputationInfo m_info;
    bool m_isInitialized;
    bool m_factorizationIsOk;
    bool m_analysisIsOk;
    NCMatrix m_mat; // The input (permuted ) matrix 
    SCMatrix m_Lstore; // The lower triangular matrix (supernodal)
    MappedSparseMatrix<Scalar> m_Ustore; // The upper triangular matrix
    PermutationType m_perm_c; // Column permutation 
    PermutationType m_perm_r ; // Row permutation
    IndexVector m_etree; // Column elimination tree 
    
    LU_GlobalLU_t<IndexVector, ScalarVector> m_glu; // persistent data to facilitate multiple factors 
                               // FIXME All fields of this struct can be defined separately as class members
                               
    // SuperLU/SparseLU options 
    bool m_symmetricmode;
    
    // values for performance 
    LU_perfvalues m_perfv; 
    RealScalar m_diagpivotthresh; // Specifies the threshold used for a diagonal entry to be an acceptable pivot
    int m_nnzL, m_nnzU; // Nonzeros in L and U factors 
  
  private:
    // Copy constructor 
    SparseLU (SparseLU& ) {}
  
}; // End class SparseLU


// Functions needed by the anaysis phase
#include "SparseLU_Coletree.h"
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
void SparseLU<MatrixType, OrderingType>::analyzePattern(const MatrixType& mat)
{
  
  //TODO  It is possible as in SuperLU to compute row and columns scaling vectors to equilibrate the matrix mat.
  
  OrderingType ord; 
  ord(mat,m_perm_c);
  //FIXME Check the right semantic behind m_perm_c
  // that is, column j of mat goes to column m_perm_c(j) of mat * m_perm_c; 

  
  // Apply the permutation to the column of the input  matrix
//   m_mat = mat * m_perm_c.inverse(); //FIXME It should be less expensive here to permute only the structural pattern of the matrix
   
  //First copy the whole input matrix. 
  m_mat = mat;
  m_mat.Uncompress(); //NOTE: The effect of this command is only to create the InnerNonzeros pointers. FIXME : This vector is filled but not subsequently used.  
  //Then, permute only the column pointers
  for (int i = 0; i < mat.cols(); i++)
  {
    m_mat.outerIndexPtr()[m_perm_c.indices()(i)] = mat.outerIndexPtr()[i]; 
    m_mat.innerNonZeroPtr()[m_perm_c.indices()(i)] = mat.outerIndexPtr()[i+1] - mat.outerIndexPtr()[i]; 
  }
    
  // Compute the column elimination tree of the permuted matrix 
  /*if (m_etree.size() == 0)  */m_etree.resize(m_mat.cols());
  
  LU_sp_coletree(m_mat, m_etree); 
     
  // In symmetric mode, do not do postorder here
  if (!m_symmetricmode) {
    IndexVector post, iwork; 
    // Post order etree
    LU_TreePostorder(m_mat.cols(), m_etree, post); 
      
   
    // Renumber etree in postorder 
    int m = m_mat.cols(); 
    iwork.resize(m+1);
    for (int i = 0; i < m; ++i) iwork(post(i)) = post(m_etree(i));
    m_etree = iwork;
    
    // Postmultiply A*Pc by post, i.e reorder the matrix according to the postorder of the etree
    PermutationType post_perm(m); //FIXME Use directly a constructor with post
    for (int i = 0; i < m; i++) 
      post_perm.indices()(i) = post(i); 
        
    // Combine the two permutations : postorder the permutation for future use
    m_perm_c = post_perm * m_perm_c;
    
  } // end postordering 
  
  m_analysisIsOk = true; 
}

// Functions needed by the numerical factorization phase 
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
template <typename MatrixType, typename OrderingType>
void SparseLU<MatrixType, OrderingType>::factorize(const MatrixType& matrix)
{
  
  eigen_assert(m_analysisIsOk && "analyzePattern() should be called first"); 
  eigen_assert((matrix.rows() == matrix.cols()) && "Only for squared matrices");
  
  typedef typename IndexVector::Scalar Index; 
  
  
  // Apply the column permutation computed in analyzepattern()
  //   m_mat = matrix * m_perm_c.inverse(); 
  m_mat = matrix;
  m_mat.Uncompress(); //NOTE: The effect of this command is only to create the InnerNonzeros pointers.
  //Then, permute only the column pointers
  for (int i = 0; i < matrix.cols(); i++)
  {
    m_mat.outerIndexPtr()[m_perm_c.indices()(i)] = matrix.outerIndexPtr()[i]; 
    m_mat.innerNonZeroPtr()[m_perm_c.indices()(i)] = matrix.outerIndexPtr()[i+1] - matrix.outerIndexPtr()[i]; 
  }
  
  int m = m_mat.rows();
  int n = m_mat.cols();
  int nnz = m_mat.nonZeros();
  int maxpanel = m_perfv.panel_size * m;
  // Allocate working storage common to the factor routines
  int lwork = 0;
  int info = LUMemInit(m, n, nnz, lwork, m_perfv.fillfactor, m_perfv.panel_size, m_glu); 
  if (info) 
  {
    std::cerr << "UNABLE TO ALLOCATE WORKING MEMORY\n\n" ;
    m_factorizationIsOk = false;
    return ; 
  }
  
  // Set up pointers for integer working arrays 
  IndexVector segrep(m); segrep.setZero();
  IndexVector parent(m); parent.setZero();
  IndexVector xplore(m); xplore.setZero();
  IndexVector repfnz(maxpanel);
  IndexVector panel_lsub(maxpanel);
  IndexVector xprune(n); xprune.setZero();
  IndexVector marker(m*LU_NO_MARKER); marker.setZero();
  
  repfnz.setConstant(-1); 
  panel_lsub.setConstant(-1);
  
  // Set up pointers for scalar working arrays 
  ScalarVector dense; 
  dense.setZero(maxpanel);
  ScalarVector tempv; 
  tempv.setZero(LU_NUM_TEMPV(m, m_perfv.panel_size, m_perfv.maxsuper, m_perfv.rowblk) );
  
  // Compute the inverse of perm_c
  PermutationType iperm_c(m_perm_c.inverse()); 
  
  // Identify initial relaxed snodes
  IndexVector relax_end(n);
  if ( m_symmetricmode == true ) 
    LU_heap_relax_snode<IndexVector>(n, m_etree, m_perfv.relax, marker, relax_end);
  else
    LU_relax_snode<IndexVector>(n, m_etree, m_perfv.relax, marker, relax_end);
  
  
  m_perm_r.resize(m); 
  m_perm_r.indices().setConstant(-1);
  marker.setConstant(-1);
  
  IndexVector& xsup = m_glu.xsup; 
  IndexVector& supno = m_glu.supno; 
  IndexVector& xlsub = m_glu.xlsub;
  IndexVector& xlusup = m_glu.xlusup;
  IndexVector& xusub = m_glu.xusub;
  ScalarVector& lusup = m_glu.lusup; 
  Index& nzlumax = m_glu.nzlumax; 
    
  supno(0) = IND_EMPTY; xsup.setConstant(0);
  xsup(0) = xlsub(0) = xusub(0) = xlusup(0) = Index(0);
  
  // Work on one 'panel' at a time. A panel is one of the following :
  //  (a) a relaxed supernode at the bottom of the etree, or
  //  (b) panel_size contiguous columns, <panel_size> defined by the user
  int jcol,kcol; 
  IndexVector panel_histo(n);
  Index nextu, nextlu, jsupno, fsupc, new_next;
  Index pivrow; // Pivotal row number in the original row matrix
  int nseg1; // Number of segments in U-column above panel row jcol
  int nseg; // Number of segments in each U-column 
  int irep, icol; 
  int i, k, jj; 
  for (jcol = 0; jcol < n; )
  {
    if (relax_end(jcol) != IND_EMPTY) 
    { // Starting a relaxed node from jcol
      kcol = relax_end(jcol); // End index of the relaxed snode 
      
      // Factorize the relaxed supernode(jcol:kcol)
      // First, determine the union of the row structure of the snode 
      info = LU_snode_dfs(jcol, kcol, m_mat, xprune, marker, m_glu); 
      if ( info ) 
      {
        std::cerr << "MEMORY ALLOCATION FAILED IN SNODE_DFS() \n";
        m_info = NumericalIssue; 
        m_factorizationIsOk = false; 
        return; 
      }
      nextu = xusub(jcol); //starting location of column jcol in ucol
      nextlu = xlusup(jcol); //Starting location of column jcol in lusup (rectangular supernodes)
      jsupno = supno(jcol); // Supernode number which column jcol belongs to 
      fsupc = xsup(jsupno); //First column number of the current supernode
      new_next = nextlu + (xlsub(fsupc+1)-xlsub(fsupc)) * (kcol - jcol + 1);
      int mem; 
      while (new_next > nzlumax ) 
      {
        mem = LUMemXpand(lusup, nzlumax, nextlu, LUSUP, m_glu.num_expansions);
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
          dense(it.row()) = it.value();
        
        // Numeric update within the snode 
        LU_snode_bmod(icol, fsupc, dense, m_glu); 
        
        // Eliminate the current column 
        info = LU_pivotL(icol, m_diagpivotthresh, m_perm_r.indices(), iperm_c.indices(), pivrow, m_glu); 
        if ( info ) 
        {
          m_info = NumericalIssue; 
          std::cerr<< "THE MATRIX IS STRUCTURALLY SINGULAR ... ZERO COLUMN AT " << info <<std::endl; 
          m_factorizationIsOk = false; 
          return; 
        }
      }
      jcol = icol; // The last column te be eliminated
    }
    else 
    { // Work on one panel of panel_size columns
      
      // Adjust panel size so that a panel won't overlap with the next relaxed snode. 
      int panel_size = m_perfv.panel_size; // upper bound on panel width
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
      LU_panel_dfs(m, panel_size, jcol, m_mat, m_perm_r.indices(), nseg1, dense, panel_lsub, segrep, repfnz, xprune, marker, parent, xplore, m_glu); 
      
      // Numeric sup-panel updates in topological order 
      LU_panel_bmod(m, panel_size, jcol, nseg1, dense, tempv, segrep, repfnz, m_perfv, m_glu); 
      
      // Sparse LU within the panel, and below the panel diagonal 
      for ( jj = jcol; jj< jcol + panel_size; jj++) 
      {
        k = (jj - jcol) * m; // Column index for w-wide arrays 
        
        nseg = nseg1; // begin after all the panel segments
        //Depth-first-search for the current column
        VectorBlock<IndexVector> panel_lsubk(panel_lsub, k, m);
        VectorBlock<IndexVector> repfnz_k(repfnz, k, m); 
        info = LU_column_dfs(m, jj, m_perm_r.indices(), m_perfv.maxsuper, nseg, panel_lsubk, segrep, repfnz_k, xprune, marker, parent, xplore, m_glu); 
        if ( info ) 
        {
          std::cerr << "UNABLE TO EXPAND MEMORY IN COLUMN_DFS() \n";
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
        // Numeric updates to this column 
        VectorBlock<ScalarVector> dense_k(dense, k, m); 
        VectorBlock<IndexVector> segrep_k(segrep, nseg1, m-nseg1); 
        info = LU_column_bmod(jj, (nseg - nseg1), dense_k, tempv, segrep_k, repfnz_k, jcol, m_glu); 
        if ( info ) 
        {
          std::cerr << "UNABLE TO EXPAND MEMORY IN COLUMN_BMOD() \n";
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
        
        // Copy the U-segments to ucol(*)
        info = LU_copy_to_ucol(jj, nseg, segrep, repfnz_k ,m_perm_r.indices(), dense_k, m_glu); 
        if ( info ) 
        {
          std::cerr << "UNABLE TO EXPAND MEMORY IN COPY_TO_UCOL() \n";
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
        
        // Form the L-segment 
        info = LU_pivotL(jj, m_diagpivotthresh, m_perm_r.indices(), iperm_c.indices(), pivrow, m_glu);
        if ( info ) 
        {
          std::cerr<< "THE MATRIX IS STRUCTURALLY SINGULAR ... ZERO COLUMN AT " << info <<std::endl; 
          m_info = NumericalIssue; 
          m_factorizationIsOk = false; 
          return; 
        }
        
        // Prune columns (0:jj-1) using column jj
        LU_pruneL(jj, m_perm_r.indices(), pivrow, nseg, segrep, repfnz_k, xprune, m_glu); 
        
        // Reset repfnz for this column 
        for (i = 0; i < nseg; i++)
        {
          irep = segrep(i); 
          repfnz_k(irep) = IND_EMPTY; 
        }
      } // end SparseLU within the panel  
      jcol += panel_size;  // Move to the next panel
    } // end else 
  } // end for -- end elimination 
  
  // Count the number of nonzeros in factors 
  LU_countnz(n, m_nnzL, m_nnzU, m_glu); 
  // Apply permutation  to the L subscripts 
  LU_fixupL(n, m_perm_r.indices(), m_glu); 
  
  
  
  // Create supernode matrix L 
  m_Lstore.setInfos(m, n, m_glu.lusup, m_glu.xlusup, m_glu.lsub, m_glu.xlsub, m_glu.supno, m_glu.xsup); 
  // Create the column major upper sparse matrix  U; 
  new (&m_Ustore) MappedSparseMatrix<Scalar> ( m, n, m_nnzU, m_glu.xusub.data(), m_glu.usub.data(), m_glu.ucol.data() ); 
  
  m_info = Success;
  m_factorizationIsOk = true;
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
    dec()._solve(rhs(),dst);
  }
};

} // end namespace internal



} // End namespace Eigen 
#endif