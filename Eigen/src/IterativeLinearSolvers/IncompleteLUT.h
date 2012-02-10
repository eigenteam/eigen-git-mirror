// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
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

#ifndef EIGEN_INCOMPLETE_LUT_H
#define EIGEN_INCOMPLETE_LUT_H
#include <bench/btl/generic_bench/utils/utilities.h>
#include <Eigen/src/OrderingMethods/Amd.h>

/**
 * \brief Incomplete LU factorization with dual-threshold strategy
 * During the numerical factorization, two dropping rules are used :
 *  1) any element whose magnitude is less than some tolerance  is dropped. 
 *    This tolerance is obtained by multiplying the input tolerance @p droptol 
 *    by the  average magnitude of all the original elements in the current row.
 *  2) After the elimination of the row, only the @p fill largest elements in 
 *    the L part and the @p fill largest elements in the U part are kept 
 *    (in addition to the diagonal element ). Note that @p fill is computed from 
 *    the input parameter @p fillfactor which is used the ratio to control the fill_in 
 *    relatively to the initial number of nonzero elements.
 * 
 * The two extreme cases are when @p droptol=0 (to keep all the @p fill*2 largest elements)
 * and when @p fill=n/2 with @p droptol being different to zero. 
 * 
 * References : Yousef Saad, ILUT: A dual threshold incomplete LU factorization, 
 *              Numerical Linear Algebra with Applications, 1(4), pp 387-402, 1994.
 * 
 * NOTE : The following implementation is derived from the ILUT implementation
 * in the SPARSKIT package, Copyright (C) 2005, the Regents of the University of Minnesota 
 *  released under the terms of the GNU LGPL; 
 * see http://www-users.cs.umn.edu/~saad/software/SPARSKIT/README for more details.
 */
template <typename _Scalar>
class IncompleteLUT
{
    typedef _Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef Matrix<Scalar,Dynamic,1> Vector;
    typedef SparseMatrix<Scalar,RowMajor> FactorType;
    typedef SparseMatrix<Scalar,ColMajor> PermutType;
    typedef typename FactorType::Index Index;

  public:
    typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
    
    IncompleteLUT() : m_droptol(NumTraits<Scalar>::dummy_precision()),m_fillfactor(10),m_isInitialized(false),m_analysisIsOk(false),m_factorizationIsOk(false) {}; 
    
    template<typename MatrixType>
    IncompleteLUT(const MatrixType& mat, RealScalar droptol, int fillfactor) 
    : m_droptol(droptol),m_fillfactor(fillfactor),m_isInitialized(false),m_analysisIsOk(false),m_factorizationIsOk(false)
    {
      eigen_assert(fillfactor != 0);
      compute(mat); 
    }
    
    Index rows() const { return m_lu.rows(); }
    
    Index cols() const { return m_lu.cols(); }
    
    template<typename MatrixType>
    void analyzePattern(const MatrixType& amat)
    {
      /* Compute the Fill-reducing permutation */
      SparseMatrix<Scalar,ColMajor, Index> mat1 = amat;
      SparseMatrix<Scalar,ColMajor, Index> mat2 = amat.transpose();
      SparseMatrix<Scalar,ColMajor, Index> AtA = mat2 * mat1; /* Symmetrize the pattern */
      AtA.prune(keep_diag());
      internal::minimum_degree_ordering<Scalar, Index>(AtA, m_P);  /* Then compute the AMD ordering... */
      
      m_Pinv  = m_P.inverse(); /* ... and the inverse permutation */
      m_analysisIsOk = true; 
    }
    
    template<typename MatrixType>
    void  factorize(const MatrixType& amat)
    {
      eigen_assert((amat.rows() == amat.cols()) && "The factorization should be done on a square matrix");
      int n = amat.cols();  /* Size of the matrix */
      m_lu.resize(n,n); 
      int fill_in; /* Number of largest elements to keep in each row */
      int nnzL, nnzU; /* Number of largest nonzero elements to keep in the L and the U part of the current row */
      /* Declare Working vectors and variables */
      int sizeu; /*  number of nonzero elements in the upper part of the current row */
      int sizel; /*  number of nonzero elements in the lower part of the current row */
      Vector u(n) ; /* real values of the row -- maximum size is n --  */
      VectorXi ju(n); /*column position of the values in u -- maximum size  is n*/
      VectorXi jr(n); /* Indicate the position of the nonzero elements in the vector u -- A zero location is indicated by -1*/
      int j, k, ii, jj, jpos, minrow, len;
      Scalar fact, prod;
      RealScalar rownorm;

      /* Apply the fill-reducing permutation */
      
      eigen_assert(m_analysisIsOk && "You must first call analyzePattern()");
      SparseMatrix<Scalar,RowMajor, Index> mat;
      mat = amat.twistedBy(m_Pinv);  
      
      /* Initialization */
      fact = 0;
      jr.fill(-1); 
      ju.fill(0);
      u.fill(0);
      fill_in =   static_cast<int> (amat.nonZeros()*m_fillfactor)/n+1;
      if (fill_in > n) fill_in = n;
      nnzL = fill_in/2; 
      nnzU = nnzL;
      m_lu.reserve(n * (nnzL + nnzU + 1)); 
      for (int ii = 0; ii < n; ii++) 
      { /* global loop over the rows of the sparse matrix */
      
        /* Copy the lower and the upper part of the row i of mat in the working vector u */
        sizeu = 1;
        sizel = 0;
        ju(ii) = ii;
        u(ii) = 0; 
        jr(ii) = ii;
        rownorm = 0; 
        
        typename FactorType::InnerIterator j_it(mat, ii); /* Iterate through the current row ii */
        for (; j_it; ++j_it)
        {
          k = j_it.index(); 
          if (k < ii) 
          { /* Copy the lower part */
            ju(sizel) = k; 
            u(sizel) = j_it.value();
            jr(k) = sizel; 
            ++sizel;
          }
          else if (k == ii)
          {
            u(ii) = j_it.value();
          } 
          else
          { /* Copy the upper part */
            jpos = ii + sizeu;
            ju(jpos) = k;
            u(jpos) = j_it.value();
            jr(k) = jpos; 
            ++sizeu; 
          }
          rownorm += internal::abs2(j_it.value());
        } /* end copy of the row */
        /* detect possible zero row */
        if (rownorm == 0) eigen_internal_assert(false); 
        rownorm = std::sqrt(rownorm); /* Take the 2-norm of the current row as a relative tolerance */
        
        /* Now, eliminate the previous nonzero rows */
        jj = 0; len = 0; 
        while (jj < sizel) 
        { /* In order to eliminate in the correct order, we must select first the smallest column index among  ju(jj:sizel) */
        
          minrow = ju.segment(jj,sizel-jj).minCoeff(&k); /* k est relatif au segment */
          k += jj;
          if (minrow != ju(jj)) { /* swap the two locations */ 
            j = ju(jj);
            std::swap(ju(jj), ju(k));  
            jr(minrow) = jj;   jr(j) = k; 
            std::swap(u(jj), u(k)); 
          }      
          /* Reset this location to zero */
          jr(minrow) = -1;
          
          /* Start elimination */
          typename FactorType::InnerIterator ki_it(m_lu, minrow);   
          while (ki_it && ki_it.index() < minrow) ++ki_it;  
          if(ki_it && ki_it.col()==minrow) fact = u(jj) / ki_it.value();
          else { eigen_internal_assert(false); }
          if( std::abs(fact) <= m_droptol ) {
            jj++;
            continue ; /* This element is been dropped */
          }
          /* linear combination of the current row ii and the row minrow */
          ++ki_it;
          for (; ki_it; ++ki_it) {
            prod = fact * ki_it.value();
            j = ki_it.index();
            jpos =  jr(j); 
            if (j >= ii) { /* Dealing with the upper part */
              if (jpos == -1) { /* Fill-in element */ 
                int newpos = ii + sizeu;
                ju(newpos) = j;
                u(newpos) = - prod;
                jr(j) = newpos;
                sizeu++;
                if (sizeu > n) { eigen_internal_assert(false);}
              }
              else { /* Not a fill_in element */
                u(jpos) -= prod;
              }
            }
            else { /* Dealing with the lower part */
              if (jpos == -1) { /* Fill-in element */
                ju(sizel) = j;
                jr(j) = sizel;
                u(sizel) = - prod;
                sizel++;
                if(sizel > n) { eigen_internal_assert(false);}
              }
              else {
                u(jpos) -= prod;
              }
            }
          }
          /* Store the pivot element */
          u(len) = fact;
          ju(len) = minrow;
          ++len; 
          
          jj++; 
        } /* End While loop -- end of the elimination on the row ii*/
        /* Reset the upper part of the pointer jr to zero */
        for (k = 0; k <sizeu; k++){
          jr(ju(ii+k)) = -1;
        }
        /* Sort the L-part of the row --use Quicksplit()*/
        sizel = len; 
        len = std::min(sizel, nnzL ); 
        typename Vector::SegmentReturnType ul(u.segment(0, len)); 
        typename VectorXi::SegmentReturnType jul(ju.segment(0, len));
        QuickSplit(ul, jul, len); 
        
        
        /* Store the  largest  m_fill elements of the L part  */
        m_lu.startVec(ii);
        for (k = 0; k < len; k++){
          m_lu.insertBackByOuterInnerUnordered(ii,ju(k)) = u(k);
        }
        
        /* Store the diagonal element */
        if (u(ii) == Scalar(0)) 
          u(ii) = std::sqrt(m_droptol ) * rownorm ;  /* NOTE This is used to avoid a zero pivot, because we are doing an incomplete factorization  */
        m_lu.insertBackByOuterInnerUnordered(ii, ii) = u(ii);
        /* Sort the U-part of the row -- Use Quicksplit() */
        len = 0; 
        for (k = 1; k < sizeu; k++) { /* First, drop any element that is below a relative tolerance */
          if ( std::abs(u(ii+k)) > m_droptol * rownorm ) {
            ++len; 
            u(ii + len) = u(ii + k); 
            ju(ii + len) = ju(ii + k); 
          }
        }
        sizeu = len + 1; /* To take into account the diagonal element */
        len = std::min(sizeu, nnzU);
        typename Vector::SegmentReturnType uu(u.segment(ii+1, sizeu-1)); 
        typename VectorXi::SegmentReturnType juu(ju.segment(ii+1, sizeu-1));
        QuickSplit(uu, juu, len); 
        /* Store the largest <fill> elements of the U part */
        for (k = ii + 1; k < ii + len; k++){
          m_lu.insertBackByOuterInnerUnordered(ii,ju(k)) = u(k);
        }
      } /* End global for-loop */
      m_lu.finalize();
      m_lu.makeCompressed(); /* NOTE To save the extra space */
      
      m_factorizationIsOk = true; 
    }
    
    /**
    * Compute an incomplete LU factorization with dual threshold on the matrix mat
    * No pivoting is done in this version
    * 
    **/
    template<typename MatrixType>
    IncompleteLUT<Scalar>& compute(const MatrixType& amat)
    {
      analyzePattern(amat); 
      factorize(amat);
      eigen_assert(m_factorizationIsOk == true); 
      m_isInitialized = true;
      return *this;
    }

    
    void setDroptol(RealScalar droptol); 
    void setFillfactor(int fillfactor); 
    
    
    
    
    template<typename Rhs, typename Dest>
    void _solve(const Rhs& b, Dest& x) const
    {
      x = m_Pinv * b;  
      x = m_lu.template triangularView<UnitLower>().solve(x);/* Compute L*x = P*b for x */
      x = m_lu.template triangularView<Upper>().solve(x); /* Compute U * z  = y for z */
      x = m_P * x; 
    }

    template<typename Rhs> inline const internal::solve_retval<IncompleteLUT, Rhs>
     solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "IncompleteLUT is not initialized.");
      eigen_assert(cols()==b.rows()
                && "IncompleteLUT::solve(): invalid number of rows of the right hand side matrix b");
      return internal::solve_retval<IncompleteLUT, Rhs>(*this, b.derived());
    }
    
protected:
    FactorType m_lu;
    RealScalar m_droptol;
    int m_fillfactor;   
    bool m_factorizationIsOk; 
    bool m_analysisIsOk; 
    bool m_isInitialized;
    template <typename VectorV, typename VectorI> 
    int QuickSplit(VectorV &row, VectorI &ind, int ncut); 
    PermutationMatrix<Dynamic,Dynamic,Index> m_P; /* Fill-reducing permutation */
    PermutationMatrix<Dynamic,Dynamic,Index> m_Pinv; /* Inverse permutation */ 
    
    /** keeps off-diagonal entries; drops diagonal entries */
    struct keep_diag {
      inline bool operator() (const Index& row, const Index& col, const Scalar&) const
      {
        return row!=col;
      }
    };
};

/**
 * Set control parameter droptol
 *  \param droptol   Drop any element whose magnitude is less than this tolerance 
 **/ 
template<typename Scalar>
void IncompleteLUT<Scalar>::setDroptol(RealScalar droptol)
{
  this->m_droptol = droptol;   
}

/**
 * Set control parameter fillfactor
 * \param fillfactor  This is used to compute the  number @p fill_in of largest elements to keep on each row. 
 **/ 
template<typename Scalar>
void IncompleteLUT<Scalar>::setFillfactor(int fillfactor)
{
  this->m_fillfactor = fillfactor;   
}


/**
 * Compute a quick-sort split of a vector 
 * On output, the vector row is permuted such that its elements satisfy
 * abs(row(i)) >= abs(row(ncut)) if i<ncut
 * abs(row(i)) <= abs(row(ncut)) if i>ncut 
 * \param row The vector of values
 * \param ind The array of index for the elements in @p row
 * \param ncut  The number of largest elements to keep
 **/ 
template <typename Scalar>
template <typename VectorV, typename VectorI>
int   IncompleteLUT<Scalar>::QuickSplit(VectorV &row, VectorI &ind, int ncut)
{
  int i,j,mid; 
  Scalar d; 
  int n = row.size(); /* lenght of the vector */
  int first, last ; 
  
  ncut--; /* to fit the zero-based indices */
  first = 0; 
  last = n-1; 
  if (ncut < first || ncut > last ) return 0;
  
  do {
    mid = first; 
    RealScalar abskey = std::abs(row(mid)); 
    for (j = first + 1; j <= last; j++) {
      if ( std::abs(row(j)) > abskey) {
        ++mid;
        std::swap(row(mid), row(j));
        std::swap(ind(mid), ind(j));
      }
    }
    /* Interchange for the pivot element */
    std::swap(row(mid), row(first));
    std::swap(ind(mid), ind(first));
    
    if (mid > ncut) last = mid - 1;
    else if (mid < ncut ) first = mid + 1; 
  } while (mid != ncut );
  
  
  return 0; /* mid is equal to ncut */
  
}


namespace internal {

template<typename _MatrixType, typename Rhs>
struct solve_retval<IncompleteLUT<_MatrixType>, Rhs>
  : solve_retval_base<IncompleteLUT<_MatrixType>, Rhs>
{
  typedef IncompleteLUT<_MatrixType> Dec;
  EIGEN_MAKE_SOLVE_HELPERS(Dec,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dec()._solve(rhs(),dst);
  }
};

}
#endif // EIGEN_INCOMPLETE_LUT_H

