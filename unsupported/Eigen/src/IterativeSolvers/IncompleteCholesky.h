// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_INCOMPLETE_CHOlESKY_H
#define EIGEN_INCOMPLETE_CHOlESKY_H
#include "Eigen/src/IterativeLinearSolvers/IncompleteLUT.h" 
#include <Eigen/OrderingMethods>
#include <list>

namespace Eigen {  
/** 
 * \brief Modified Incomplete Cholesky with dual threshold
 * 
 * References : C-J. Lin and J. J. Moré, Incomplete Cholesky Factorizations with
 *              Limited memory, SIAM J. Sci. Comput.  21(1), pp. 24-45, 1999
 * 
 * \tparam _MatrixType The type of the sparse matrix. It should be a symmetric 
 *                     matrix. It is advised to give  a row-oriented sparse matrix 
 * \tparam _UpLo The triangular part of the matrix to reference. 
 * \tparam _OrderingType 
 */

template <typename Scalar, int _UpLo = Lower, typename _OrderingType = NaturalOrdering<int> >
class IncompleteCholesky : internal::noncopyable
{
  public:
    typedef SparseMatrix<Scalar,ColMajor> MatrixType;
    typedef _OrderingType OrderingType;
    typedef typename MatrixType::RealScalar RealScalar; 
    typedef typename MatrixType::Index Index; 
    typedef PermutationMatrix<Dynamic, Dynamic, Index> PermutationType;
    typedef Matrix<Scalar,Dynamic,1> VectorType; 
    typedef Matrix<Index,Dynamic, 1> IndexType; 

  public:
    IncompleteCholesky() {}
    IncompleteCholesky(const MatrixType& matrix)
    {
      compute(matrix);
    }
    
    Index rows() const { return m_L.rows(); }
    
    Index cols() const { return m_L.cols(); }
    

    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was succesful,
      *          \c NumericalIssue if the matrix appears to be negative.
      */
    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "IncompleteLLT is not initialized.");
      return m_info;
    }
    /**
    * \brief Computes the fill reducing permutation vector. 
    */
    template<typename MatrixType>
    void analyzePattern(const MatrixType& mat)
    {
      OrderingType ord; 
      ord(mat, m_perm); 
      m_analysisIsOk = true; 
    }
    
    template<typename MatrixType>
    void factorize(const MatrixType& amat);
    
    template<typename MatrixType>
    void compute (const MatrixType& matrix)
    {
      analyzePattern(matrix); 
      factorize(matrix);
    }
    
    template<typename Rhs, typename Dest>
    void _solve(const Rhs& b, Dest& x) const
    {
      eigen_assert(m_factorizationIsOk && "factorize() should be called first");
      if (m_perm.rows() == b.rows())
        x = m_perm.inverse() * b; 
      else 
        x = b; 
      x = m_L.template triangularView<UnitLower>().solve(x); 
      x = m_L.adjoint().template triangularView<Upper>().solve(x); 
      if (m_perm.rows() == b.rows())
        x = m_perm * x;
    }
    template<typename Rhs> inline const internal::solve_retval<IncompleteCholesky, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "IncompleteLLT is not initialized.");
      eigen_assert(cols()==b.rows()
                && "IncompleteLLT::solve(): invalid number of rows of the right hand side matrix b");
      return internal::solve_retval<IncompleteCholesky, Rhs>(*this, b.derived());
    }
  protected:
    SparseMatrix<Scalar,ColMajor> m_L;  // The lower part stored in CSC
    bool m_analysisIsOk; 
    bool m_factorizationIsOk; 
    bool m_isInitialized;
    ComputationInfo m_info;
    PermutationType m_perm; 
    
}; 

template<typename Scalar, int _UpLo, typename OrderingType>
template<typename _MatrixType>
void IncompleteCholesky<Scalar,_UpLo, OrderingType>::factorize(const _MatrixType& mat)
{
  eigen_assert(m_analysisIsOk && "analyzePattern() should be called first"); 
  
  // FIXME Stability: We should probably compute the scaling factors and the shifts that are needed to ensure an efficient LLT preconditioner. 
  
  // Dropping strategies : Keep only the p largest elements per column, where p is the number of elements in the column of the original matrix. Other strategies will be added
  
  // Apply the fill-reducing permutation computed in analyzePattern()
  if (m_perm.rows() == mat.rows() )
    m_L.template selfadjointView<Lower>() = mat.template selfadjointView<_UpLo>().twistedBy(m_perm);
  else
    m_L.template selfadjointView<Lower>() = mat.template selfadjointView<_UpLo>();
  
  int n = mat.cols(); 
  
  Scalar *vals = m_L.valuePtr(); //Values 
  Index *rowIdx = m_L.innerIndexPtr(); //Row indices 
  Index *colPtr = m_L.outerIndexPtr(); // Pointer to the beginning of each row
  VectorType firstElt(n-1); // for each j, points to the next entry in vals that will be used in the factorization
  // Initialize firstElt; 
  for (int j = 0; j < n-1; j++) firstElt(j) = colPtr[j]+1; 
  std::vector<std::list<Index> > listCol(n); // listCol(j) is a linked list of columns to update column j
  VectorType curCol(n); // Store a  nonzero values in each column
  VectorType irow(n); // Row indices of nonzero elements in each column
  // jki version of the Cholesky factorization 
  for (int j=0; j < n; j++)
  {
     //Left-looking factorize the column j 
     // First, load the jth column into curCol 
     Scalar diag = vals[colPtr[j]];  // Lower diagonal matrix with 
     curCol.setZero();
     irow.setLinSpaced(n,0,n-1); 
     for (int i = colPtr[j] + 1; i < colPtr[j+1]; i++)
     {
       curCol(rowIdx[i]) = vals[i]; 
       irow(rowIdx[i]) = rowIdx[i]; 
     }
     
     std::list<int>::iterator k; 
     // Browse all previous columns that will update column j
     for(k = listCol[j].begin(); k != listCol[j].end(); k++) 
     {
       int jk = firstElt(*k); // First element to use in the column 
       Scalar a_jk = vals[jk]; 
       diag -= a_jk * a_jk; 
       jk += 1; 
       for (int i = jk; i < colPtr[*k]; i++)
       {
         curCol(rowIdx[i]) -= vals[i] * a_jk ;
       }
       firstElt(*k) = jk; 
       if (jk < colPtr[*k+1]) 
       {
         // Add this column to the updating columns list for column *k+1
         listCol[rowIdx[jk]].push_back(*k); 
       }
     }
     
     // Select the largest p elements
     //  p is the original number of elements in the column (without the diagonal)
     int p = colPtr[j+1] - colPtr[j] - 2 ; 
     internal::QuickSplit(curCol, irow, p); 
     if(RealScalar(diag) <= 0)
     {
       m_info = NumericalIssue; 
       return; 
     }
     RealScalar rdiag = internal::sqrt(RealScalar(diag));
     Scalar scal = Scalar(1)/rdiag; 
     vals[colPtr[j]] = rdiag;
     // Insert the largest p elements in the matrix and scale them meanwhile  
     int cpt = 0; 
     for (int i = colPtr[j]+1; i < colPtr[j+1]; i++)
     {
       vals[i] = curCol(cpt) * scal; 
       rowIdx[i] = irow(cpt); 
       cpt ++; 
     }
  }
  m_factorizationIsOk = true; 
  m_isInitialized = true;
  m_info = Success; 
}

namespace internal {

template<typename _MatrixType, typename Rhs>
struct solve_retval<IncompleteCholesky<_MatrixType>, Rhs>
  : solve_retval_base<IncompleteCholesky<_MatrixType>, Rhs>
{
  typedef IncompleteCholesky<_MatrixType> Dec;
  EIGEN_MAKE_SOLVE_HELPERS(Dec,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dec()._solve(rhs(),dst);
  }
};

} // end namespace internal

} // end namespace Eigen 

#endif