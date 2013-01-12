#ifndef EIGEN_SPARSE_QR_H
#define EIGEN_SPARSE_QR_H
// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Desire Nuentsa <desire.nuentsa_wakam@inria.fr>
// Copyright (C) 2012 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


namespace Eigen {
#include "../SparseLU/SparseLU_Coletree.h"

template<typename MatrixType, typename OrderingType> class SparseQR;
template<typename SparseQRType> struct SparseQRMatrixQReturnType;
template<typename SparseQRType> struct SparseQRMatrixQTransposeReturnType;
template<typename SparseQRType, typename Derived> struct SparseQR_QProduct;
namespace internal {
  template <typename SparseQRType> struct traits<SparseQRMatrixQReturnType<SparseQRType> >
  {
    typedef typename SparseQRType::MatrixType ReturnType;
  };
  template <typename SparseQRType> struct traits<SparseQRMatrixQTransposeReturnType<SparseQRType> >
  {
    typedef typename SparseQRType::MatrixType ReturnType;
  };
  template <typename SparseQRType, typename Derived> struct traits<SparseQR_QProduct<SparseQRType, Derived> >
  {
    typedef typename Derived::PlainObject ReturnType;
  };
} // End namespace internal

/**
  * \ingroup SparseQR_Module
  * \class SparseQR
  * \brief Sparse left-looking QR factorization
  * 
  * This class is used to perform a left-looking QR decomposition 
  * of sparse matrices. The result is then used to solve linear leasts_square systems.
  * Clearly, a QR factorization is returned such that A*P = Q*R where :
  * 
  * P is the column permutation. Use colsPermutation() to get it.
  * 
  * Q is the orthogonal matrix represented as Householder reflectors. 
  * Use matrixQ() to get an expression and matrixQ().transpose() to get the transpose.
  * You can then apply it to a vector.
  * 
  * R is the sparse triangular factor. Use matrixR() to get it as SparseMatrix.
  * 
  * \note This is not a rank-revealing QR decomposition.
  * 
  * \tparam _MatrixType The type of the sparse matrix A, must be a column-major SparseMatrix<>
  * \tparam _OrderingType The fill-reducing ordering method. See the \link OrderingMethods_Module 
  *  OrderingMethods \endlink module for the list of built-in and external ordering methods.
  * 
  * 
  */
template<typename _MatrixType, typename _OrderingType>
class SparseQR
{
  public:
    typedef _MatrixType MatrixType;
    typedef _OrderingType OrderingType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::Index Index;
    typedef SparseMatrix<Scalar,ColMajor,Index> QRMatrixType;
    typedef Matrix<Index, Dynamic, 1> IndexVector;
    typedef Matrix<Scalar, Dynamic, 1> ScalarVector;
    typedef PermutationMatrix<Dynamic, Dynamic, Index> PermutationType;
  public:
    SparseQR () : m_isInitialized(false),m_analysisIsok(false)
    { }
    
    SparseQR(const MatrixType& mat) : m_isInitialized(false),m_analysisIsok(false)
    {
      compute(mat);
    }
    void compute(const MatrixType& mat)
    {
      analyzePattern(mat);
      factorize(mat);
    }
    void analyzePattern(const MatrixType& mat);
    void factorize(const MatrixType& mat);
    
    /** \returns the number of rows of the represented matrix. 
      */
    inline Index rows() const { return m_pmat.rows(); }
    
    /** \returns the number of columns of the represented matrix. 
      */
    inline Index cols() const { return m_pmat.cols();}
    
    /** \returns a const reference to the \b sparse upper triangular matrix R of the QR factorization.
      */
    const MatrixType& matrixR() const { return m_R; }
    
    /** \returns an expression of the matrix Q as products of sparse Householder reflectors.
      * You can do the following to get an actual SparseMatrix representation of Q:
      * \code
      * SparseMatrix<double> Q = SparseQR<SparseMatrix<double> >(A).matrixQ();
      * \endcode
      */
    SparseQRMatrixQReturnType<SparseQR> matrixQ() const 
    { return SparseQRMatrixQReturnType<SparseQR>(*this); }
    
    /** \returns a const reference to the fill-in reducing permutation that was applied to the columns of A
      */
    const PermutationType& colsPermutation() const
    { 
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_perm_c;
    }
    
    /** \internal */
    template<typename Rhs, typename Dest>
    bool _solve(const MatrixBase<Rhs> &B, MatrixBase<Dest> &dest) const
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
      Index rank = this->matrixR().cols();
      // Compute Q^T * b;
      dest = this->matrixQ().transpose() * B;      
      // Solve with the triangular matrix R
      Dest y;
      y = this->matrixR().template triangularView<Upper>().solve(dest.derived().topRows(rank));
      
      // Apply the column permutation
      if (m_perm_c.size())  dest.topRows(rank) =  colsPermutation().inverse() * y;
      else                  dest = y;
      
      m_info = Success;
      return true;
    }
    
    /** \returns the solution X of \f$ A X = B \f$ using the current decomposition of A.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const internal::solve_retval<SparseQR, Rhs> solve(const MatrixBase<Rhs>& B) const 
    {
      eigen_assert(m_isInitialized && "The factorization should be called first, use compute()");
      eigen_assert(this->rows() == B.rows() && "SparseQR::solve() : invalid number of rows in the right hand side matrix");
      return internal::solve_retval<SparseQR, Rhs>(*this, B.derived());
    }
    
    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was succesful,
      *          \c NumericalIssue if the QR factorization reports a numerical problem
      *          \c InvalidInput if the input matrix is invalid
      *
      * \sa iparm()          
      */
    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_info;
    }
    
  protected:
    bool m_isInitialized;
    bool m_analysisIsok;
    bool m_factorizationIsok;
    mutable ComputationInfo m_info;
    QRMatrixType m_pmat;            // Temporary matrix
    QRMatrixType m_R;               // The triangular factor matrix
    QRMatrixType m_Q;               // The orthogonal reflectors
    ScalarVector m_hcoeffs;         // The Householder coefficients
    PermutationType m_perm_c;       // Column  permutation 
    PermutationType m_perm_r;       // Column permutation 
    IndexVector m_etree;            // Column elimination tree
    IndexVector m_firstRowElt;      // First element in each row
    IndexVector m_found_diag_elem;  // Existence of diagonal elements
    template <typename, typename > friend struct SparseQR_QProduct;
    
};

/** \brief Preprocessing step of a QR factorization 
  * 
  * In this step, the fill-reducing permutation is computed and applied to the columns of A
  * and the column elimination tree is computed as well. Only the sparcity pattern of \a mat is exploited.
  * \note In this step it is assumed that there is no empty row in the matrix \a mat
  */
template <typename MatrixType, typename OrderingType>
void SparseQR<MatrixType,OrderingType>::analyzePattern(const MatrixType& mat)
{
  // Compute the column fill reducing ordering
  OrderingType ord; 
  ord(mat, m_perm_c); 
  Index n = mat.cols();
  Index m = mat.rows();
  // Permute the input matrix... only the column pointers are permuted
  // FIXME: directly send "m_perm.inverse() * mat" to coletree -> need an InnerIterator to the sparse-permutation-product expression.
  m_pmat = mat;
  m_pmat.uncompress();
  for (int i = 0; i < n; i++)
  {
    Index p = m_perm_c.size() ? m_perm_c.indices()(i) : i;
    m_pmat.outerIndexPtr()[p] = mat.outerIndexPtr()[i]; 
    m_pmat.innerNonZeroPtr()[p] = mat.outerIndexPtr()[i+1] - mat.outerIndexPtr()[i]; 
  }
  // Compute the column elimination tree of the permuted matrix
  internal::coletree(m_pmat, m_etree, m_firstRowElt);
  
  m_R.resize(n, n);
  m_Q.resize(m, m);
  // Allocate space for nonzero elements : rough estimation
  m_R.reserve(2*mat.nonZeros()); //FIXME Get a more accurate estimation through symbolic factorization with the etree
  m_Q.reserve(2*mat.nonZeros());
  m_hcoeffs.resize(n);
  m_analysisIsok = true;
}

/** \brief Perform the numerical QR factorization of the input matrix
  * 
  * The function SparseQR::analyzePattern(const MatrixType&) must have been called beforehand with
  * a matrix having the same sparcity pattern than \a mat.
  * 
  * \param mat The sparse column-major matrix
  */
template <typename MatrixType, typename OrderingType>
void SparseQR<MatrixType,OrderingType>::factorize(const MatrixType& mat)
{
  eigen_assert(m_analysisIsok && "analyzePattern() should be called before this step");
  Index m = mat.rows();
  Index n = mat.cols();
  IndexVector mark(m); mark.setConstant(-1);  // Record the visited nodes
  IndexVector Ridx(n), Qidx(m);               // Store temporarily the row indexes for the current column of R and Q
  Index nzcolR, nzcolQ;                       // Number of nonzero for the current column of R and Q
  Index pcol;
  ScalarVector tval(m); tval.setZero();       // Temporary vector
  IndexVector iperm(m);
  bool found_diag;
  if (m_perm_c.size())
    for(int i = 0; i < m; i++) iperm(m_perm_c.indices()(i)) = i;
  else
    iperm.setLinSpaced(m, 0, m-1);
      
  // Left looking QR factorization : Compute a column of R and Q at a time
  for (Index col = 0; col < n; col++)
  {
    m_R.startVec(col);
    m_Q.startVec(col);
    mark(col) = col;
    Qidx(0) = col; 
    nzcolR = 0; nzcolQ = 1;
    pcol = iperm(col);
    found_diag = false;
    // Find the nonzero locations of the column k of R, 
    // i.e All the nodes (with indexes lower than k) reachable through the col etree rooted at node k
    for (typename MatrixType::InnerIterator itp(mat, pcol); itp || !found_diag; ++itp)
    {
      Index curIdx = col;
      if (itp) curIdx = itp.row();
      if(curIdx == col) found_diag = true;
      // Get the nonzeros indexes  of the current column of R
      Index st = m_firstRowElt(curIdx); // The traversal of the etree starts here 
      if (st < 0 )
      {
        std::cerr << " Empty row found during Numerical factorization ... Abort \n";
        m_info = NumericalIssue;
        return;
      }
      // Traverse the etree 
      Index bi = nzcolR;
      for (; mark(st) != col; st = m_etree(st))
      {
        Ridx(nzcolR) = st; // Add this row to the list 
        mark(st) = col; // Mark this row as visited
        nzcolR++;
      }
      // Reverse the list to get the topological ordering
      Index nt = nzcolR-bi;
      for(int i = 0; i < nt/2; i++) std::swap(Ridx(bi+i), Ridx(nzcolR-i-1));
       
      // Copy the current row value of mat
      if (itp) tval(curIdx) = itp.value();
      else tval(curIdx) = Scalar(0.);
      
      // Compute the pattern of Q(:,k)
      if (curIdx > col && mark(curIdx) < col) 
      {
        Qidx(nzcolQ) = curIdx; // Add this row to the pattern of Q
        mark(curIdx) = col; // And mark it as visited
        nzcolQ++;
      }
    }
    
    // Browse all the indexes of R(:,col) in reverse order
    for (Index i = nzcolR-1; i >= 0; i--)
    {
      Index curIdx = Ridx(i);
      // Apply the <curIdx> householder vector  to tval
      Scalar tdot(0.);
      //First compute q'*tval
      for (typename QRMatrixType::InnerIterator itq(m_Q, curIdx); itq; ++itq)
      {
        tdot += internal::conj(itq.value()) * tval(itq.row());
      }
      tdot *= m_hcoeffs(curIdx);
      // Then compute tval = tval - q*tau
      for (typename QRMatrixType::InnerIterator itq(m_Q, curIdx); itq; ++itq)
      {
        tval(itq.row()) -= itq.value() * tdot;
      }
      //With the topological ordering, updates for curIdx are fully done at this point
      m_R.insertBackByOuterInnerUnordered(col, curIdx) = tval(curIdx);
      tval(curIdx) = Scalar(0.);
      
      // Detect fill-in for the current column of Q
      if(m_etree(curIdx) == col)
      {
        for (typename QRMatrixType::InnerIterator itq(m_Q, curIdx); itq; ++itq)
        {
          Index iQ = itq.row();
          if (mark(iQ) < col)
          {
            Qidx(nzcolQ++) = iQ; // Add this row to the pattern of Q
            mark(iQ) = col; //And mark it as visited
          }
        }
      }
    } // End update current column of R
    
    // Record the current (unscaled) column of V.
    for (Index itq = 0; itq < nzcolQ; ++itq)
    {
      Index iQ = Qidx(itq);      
      m_Q.insertBackByOuterInnerUnordered(col,iQ) = tval(iQ);
      tval(iQ) = Scalar(0.);
    }
    // Compute the new Householder reflection
    RealScalar sqrNorm =0.;
    Scalar tau; RealScalar beta;
    typename QRMatrixType::InnerIterator itq(m_Q, col);
    Scalar c0 = (itq) ? itq.value() : Scalar(0.);
    //First, the squared norm of Q((col+1):m, col)
    if(itq) ++itq;
    for (; itq; ++itq)
    {
      sqrNorm += internal::abs2(itq.value());
    }
    if(sqrNorm == RealScalar(0) && internal::imag(c0) == RealScalar(0))
    {
      tau = RealScalar(0);
      beta = internal::real(c0);
      typename QRMatrixType::InnerIterator it(m_Q,col);
      it.valueRef() = 1; //FIXME A row permutation should be performed at this point
    }
    else
    {
      beta = std::sqrt(internal::abs2(c0) + sqrNorm);
      if(internal::real(c0) >= RealScalar(0))
        beta = -beta;
      typename QRMatrixType::InnerIterator it(m_Q,col);
      it.valueRef() = 1;
      for (++it; it; ++it)
      {
        it.valueRef() /= (c0 - beta);
      }
      tau = internal::conj((beta-c0) / beta);
        
    }
    m_hcoeffs(col) = tau;
    m_R.insertBackByOuterInnerUnordered(col, col) = beta;
  }
  // Finalize the column pointers of the sparse matrices R and Q
  m_R.finalize(); m_R.makeCompressed();
  m_Q.finalize(); m_Q.makeCompressed();
  m_isInitialized = true; 
  m_factorizationIsok = true;
  m_info = Success;
  
}

namespace internal {
  
template<typename _MatrixType, typename OrderingType, typename Rhs>
struct solve_retval<SparseQR<_MatrixType,OrderingType>, Rhs>
  : solve_retval_base<SparseQR<_MatrixType,OrderingType>, Rhs>
{
  typedef SparseQR<_MatrixType,OrderingType> Dec;
  EIGEN_MAKE_SOLVE_HELPERS(Dec,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dec()._solve(rhs(),dst);
  }
};

} // end namespace internal

template <typename SparseQRType, typename Derived>
struct SparseQR_QProduct : ReturnByValue<SparseQR_QProduct<SparseQRType, Derived> >
{
  typedef typename SparseQRType::QRMatrixType MatrixType;
  typedef typename SparseQRType::Scalar Scalar;
  typedef typename SparseQRType::Index Index;
  // Get the references 
  SparseQR_QProduct(const SparseQRType& qr, const Derived& other, bool transpose) : 
  m_qr(qr),m_other(other),m_transpose(transpose) {}
  inline Index rows() const { return m_transpose ? m_qr.rowsQ() : m_qr.cols(); }
  inline Index cols() const { return m_other.cols(); }
  
  // Assign to a vector
  template<typename DesType>
  void evalTo(DesType& res) const
  {
    Index m = m_qr.rows();
    Index n = m_qr.cols(); 
    if (m_transpose)
    {
      eigen_assert(m_qr.m_Q.rows() == m_other.rows() && "Non conforming object sizes");
      // Compute res = Q' * other :
      res =  m_other;
      for (Index k = 0; k < n; k++)
      {
        Scalar tau; 
        // Or alternatively 
        tau = m_qr.m_Q.col(k).tail(m-k).dot(res.tail(m-k)); 
        tau = tau * m_qr.m_hcoeffs(k);
        res -= tau * m_qr.m_Q.col(k);
      }
    }
    else
    {
      eigen_assert(m_qr.m_Q.cols() == m_other.rows() && "Non conforming object sizes");
      // Compute res = Q * other :
      res = m_other;
      for (Index k = n-1; k >=0; k--)
      {
        Scalar tau;
        tau = m_qr.m_Q.col(k).tail(m-k).dot(res.tail(m-k));
        tau = tau * m_qr.m_hcoeffs(k);
        res -= tau * m_qr.m_Q.col(k);
      }
    }
  }
  
  const SparseQRType& m_qr;
  const Derived& m_other;
  bool m_transpose;
};

template<typename SparseQRType>
struct SparseQRMatrixQReturnType
{  
  SparseQRMatrixQReturnType(const SparseQRType& qr) : m_qr(qr) {}
  template<typename Derived>
  SparseQR_QProduct<SparseQRType, Derived> operator*(const MatrixBase<Derived>& other)
  {
    return SparseQR_QProduct<SparseQRType,Derived>(m_qr,other.derived(),false);
  }
  SparseQRMatrixQTransposeReturnType<SparseQRType> adjoint() const
  {
    return SparseQRMatrixQTransposeReturnType<SparseQRType>(m_qr);
  }
  // To use for operations with the transpose of Q
  SparseQRMatrixQTransposeReturnType<SparseQRType> transpose() const
  {
    return SparseQRMatrixQTransposeReturnType<SparseQRType>(m_qr);
  }
  const SparseQRType& m_qr;
};

template<typename SparseQRType>
struct SparseQRMatrixQTransposeReturnType
{
  SparseQRMatrixQTransposeReturnType(const SparseQRType& qr) : m_qr(qr) {}
  template<typename Derived>
  SparseQR_QProduct<SparseQRType,Derived> operator*(const MatrixBase<Derived>& other)
  {
    return SparseQR_QProduct<SparseQRType,Derived>(m_qr,other.derived(), true);
  }
  const SparseQRType& m_qr;
};

} // end namespace Eigen

#endif
