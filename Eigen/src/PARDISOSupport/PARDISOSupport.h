/*
 Copyright (c) 2011, Intel Corporation. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************
 *   Content : Eigen bindings to Intel(R) MKL PARDISO
 ********************************************************************************
*/

#ifndef EIGEN_PARDISOSUPPORT_H
#define EIGEN_PARDISOSUPPORT_H

template<typename _MatrixType>
class PardisoLU;
template<typename _MatrixType>
class PardisoLLT;
template<typename _MatrixType>
class PardisoLDLT;

namespace internal
{
  template<typename Index>
  struct pardiso_run_selector
  {
    static Index run(_MKL_DSS_HANDLE_t pt, Index maxfct, Index mnum, Index type, Index phase, Index n, void *a,
        Index *ia, Index *ja, Index *perm, Index nrhs, Index *iparm, Index msglvl, void *b, void *x)
    {
      Index error = 0;
      ::pardiso(pt, &maxfct, &mnum, &type, &phase, &n, a, ia, ja, perm, &nrhs, iparm, &msglvl, b, x, &error);
      return error;
    }
  };
  template<>
  struct pardiso_run_selector<long long int>
  {
    typedef long long int Index;
    static Index run(_MKL_DSS_HANDLE_t pt, Index maxfct, Index mnum, Index type, Index phase, Index n, void *a,
        Index *ia, Index *ja, Index *perm, Index nrhs, Index *iparm, Index msglvl, void *b, void *x)
    {
      Index error = 0;
      ::pardiso_64(pt, &maxfct, &mnum, &type, &phase, &n, a, ia, ja, perm, &nrhs, iparm, &msglvl, b, x, &error);
      return error;
    }
  };

  template<class Pardiso>
  struct pardiso_traits;

  template<typename _MatrixType>
  struct pardiso_traits< PardisoLU<_MatrixType> >
  {
    typedef _MatrixType MatrixType;
    typedef typename _MatrixType::Scalar Scalar;
    typedef typename _MatrixType::RealScalar RealScalar;
    typedef typename _MatrixType::Index Index;
  };

  template<typename _MatrixType>
  struct pardiso_traits< PardisoLLT<_MatrixType> >
  {
    typedef _MatrixType MatrixType;
    typedef typename _MatrixType::Scalar Scalar;
    typedef typename _MatrixType::RealScalar RealScalar;
    typedef typename _MatrixType::Index Index;
  };

  template<typename _MatrixType>
  struct pardiso_traits< PardisoLDLT<_MatrixType> >
  {
    typedef _MatrixType MatrixType;
    typedef typename _MatrixType::Scalar Scalar;
    typedef typename _MatrixType::RealScalar RealScalar;
    typedef typename _MatrixType::Index Index;
  };

}

template<class Derived>
class PardisoImpl
{
  public:
    typedef typename internal::pardiso_traits<Derived>::MatrixType MatrixType;
    typedef typename internal::pardiso_traits<Derived>::Scalar Scalar;
    typedef typename internal::pardiso_traits<Derived>::RealScalar RealScalar;
    typedef typename internal::pardiso_traits<Derived>::Index Index;
    typedef Matrix<Scalar,Dynamic,1> VectorType;
    typedef Matrix<Index, 1, MatrixType::ColsAtCompileTime> IntRowVectorType;
    typedef Matrix<Index, MatrixType::RowsAtCompileTime, 1> IntColVectorType;
    enum {
      ScalarIsComplex = NumTraits<Scalar>::IsComplex
    };

    PardisoImpl(int flags) : m_flags(flags)
    {
      eigen_assert((sizeof(Index) >= sizeof(_INTEGER_t) && sizeof(Index) <= 8) && "Non-supported index type");
      memset(m_iparm, 0, sizeof(m_iparm));
      m_msglvl = 0; /* No output */
      m_initialized = false;
    }

    ~PardisoImpl()
    {
      pardisoRelease();
    }

    inline Index cols() const { return m_matrix.cols(); }
    inline Index rows() const { return m_matrix.rows(); }
  
    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was succesful,
      *          \c NumericalIssue if the matrix.appears to be negative.
      */
    ComputationInfo info() const
    {
      eigen_assert(m_initialized && "Decomposition is not initialized.");
      return m_info;
    }

    int orderingMethod() const
    {
      return m_flags&OrderingMask;
    }

    Derived& compute(const MatrixType& matrix);
    /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const internal::solve_retval<PardisoImpl, Rhs>
    solve(const MatrixBase<Rhs>& b, const int transposed = SvNoTrans) const
    {
      eigen_assert(m_initialized && "SimplicialCholesky is not initialized.");
      eigen_assert(rows()==b.rows()
                && "PardisoImpl::solve(): invalid number of rows of the right hand side matrix b");
      return internal::solve_retval<PardisoImpl, Rhs>(*this, b.derived(), transposed);
    }

    Derived& derived()
    {
      return *static_cast<Derived*>(this);
    }
    const Derived& derived() const
    {
      return *static_cast<const Derived*>(this);
    }

    template<typename BDerived, typename XDerived>
    bool _solve(const MatrixBase<BDerived> &b, MatrixBase<XDerived>& x, const int transposed = SvNoTrans) const;

  protected:
    void pardisoRelease()
    {
      if(m_initialized) // Factorization ran at least once
      {
        internal::pardiso_run_selector<Index>::run(m_pt, 1, 1, m_type, -1, m_matrix.rows(), NULL, NULL, NULL, m_perm.data(), 0,
            m_iparm, m_msglvl, NULL, NULL);
        memset(m_iparm, 0, sizeof(m_iparm));
      }
    }
  protected:
    // cached data to reduce reallocation, etc.

    ComputationInfo m_info;
    bool m_symmetric, m_initialized, m_succeeded;
    int m_flags;
    Index m_type, m_msglvl;
    mutable void *m_pt[64];
    mutable Index m_iparm[64];
    mutable SparseMatrix<Scalar, RowMajor> m_matrix;
    mutable IntColVectorType m_perm;
};

template<class Derived>
Derived& PardisoImpl<Derived>::compute(const MatrixType& a)
{
  Index n = a.rows(), i;
  eigen_assert(a.rows() == a.cols());

  pardisoRelease();
  memset(m_pt, 0, sizeof(m_pt));
  m_initialized = true;

  m_symmetric = abs(m_type) < 10;

  switch (orderingMethod())
  {
    case MinimumDegree_AT_PLUS_A  : m_iparm[1] = 0; break;
    case NaturalOrdering          : m_iparm[5] = 1; break;
    case Metis                    : m_iparm[1] = 3; break;
    default:
      //std::cerr << "Eigen: ordering method \"" << Base::orderingMethod() << "\" not supported by the PARDISO backend\n";
      m_iparm[1] = 0;
  };

  m_iparm[0] = 1; /* No solver default */
  /* Numbers of processors, value of OMP_NUM_THREADS */
  m_iparm[2] = 1;
  m_iparm[3] = 0; /* No iterative-direct algorithm */
  m_iparm[4] = 0; /* No user fill-in reducing permutation */
  m_iparm[5] = 0; /* Write solution into x */
  m_iparm[6] = 0; /* Not in use */
  m_iparm[7] = 2; /* Max numbers of iterative refinement steps */
  m_iparm[8] = 0; /* Not in use */
  m_iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
  m_iparm[10] = m_symmetric ? 0 : 1; /* Use nonsymmetric permutation and scaling MPS */
  m_iparm[11] = 0; /* Not in use */
  m_iparm[12] = m_symmetric ? 0 : 1; /* Maximum weighted matching algorithm is switched-off (default for symmetric). Try m_iparm[12] = 1 in case of inappropriate accuracy */
  m_iparm[13] = 0; /* Output: Number of perturbed pivots */
  m_iparm[14] = 0; /* Not in use */
  m_iparm[15] = 0; /* Not in use */
  m_iparm[16] = 0; /* Not in use */
  m_iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
  m_iparm[18] = -1; /* Output: Mflops for LU factorization */
  m_iparm[19] = 0; /* Output: Numbers of CG Iterations */
  m_iparm[20] = 0; /* 1x1 pivoting */
  m_iparm[26] = 0; /* No matrix checker */
  m_iparm[27] = (sizeof(RealScalar) == 4) ? 1 : 0;
  m_iparm[34] = 0; /* Fortran indexing */
  m_iparm[59] = 1; /* Automatic switch between In-Core and Out-of-Core modes */

  m_perm.resize(n);
  if(orderingMethod() == NaturalOrdering)
  {
    for(Index i = 0; i < n; i++)
      m_perm[i] = i;
  }

  m_matrix = a;

  /* Convert to Fortran-style indexing */
  for(i = 0; i <= m_matrix.rows(); ++i)
    ++m_matrix._outerIndexPtr()[i];
  for(i = 0; i < m_matrix.nonZeros(); ++i)
    ++m_matrix._innerIndexPtr()[i];

  Index error = internal::pardiso_run_selector<Index>::run(m_pt, 1, 1, m_type, 12, n, 
      m_matrix._valuePtr(), m_matrix._outerIndexPtr(), m_matrix._innerIndexPtr(),
      m_perm.data(), 0, m_iparm, m_msglvl, NULL, NULL);

  switch(error)
  {
    case 0:
      m_succeeded = true;
	  m_info = Success;
      return derived();
    case -4:
    case -7:
      m_info = NumericalIssue;
      break;
    default:
      m_info = InvalidInput;
  }
  m_succeeded = false;
  return derived();
}

template<class Base>
template<typename BDerived,typename XDerived>
bool PardisoImpl<Base>::_solve(const MatrixBase<BDerived> &b,
                        MatrixBase<XDerived>& x, const int transposed) const
{
  if(m_iparm[0] == 0) // Factorization was not computed
    return false;

  Index n = m_matrix.rows();
  Index nrhs = b.cols();
  eigen_assert(n==b.rows());
  eigen_assert(((MatrixBase<BDerived>::Flags & RowMajorBit) == 0 || nrhs == 1) && "Row-major right hand sides are not supported");
  eigen_assert(((MatrixBase<XDerived>::Flags & RowMajorBit) == 0 || nrhs == 1) && "Row-major matrices of unknowns are not supported");
  eigen_assert(((nrhs == 1) || b.outerStride() == b.rows()));

  x.derived().resizeLike(b);

  switch (transposed) {
    case SvNoTrans    : m_iparm[11] = 0 ; break;
    case SvTranspose  : m_iparm[11] = 2 ; break;
    case SvAdjoint    : m_iparm[11] = 1 ; break;
    default:
      //std::cerr << "Eigen: transposition  option \"" << transposed << "\" not supported by the PARDISO backend\n";
      m_iparm[11] = 0;
  }

  Index error = internal::pardiso_run_selector<Index>::run(m_pt, 1, 1, m_type, 33, n, 
      m_matrix._valuePtr(), m_matrix._outerIndexPtr(), m_matrix._innerIndexPtr(),
      m_perm.data(), nrhs, m_iparm, m_msglvl, const_cast<Scalar*>(&b(0, 0)), &x(0, 0));

  return error==0;
}


/** \ingroup PARDISOSupport_Module
  * \class PardisoLU
  * \brief A sparse direct LU factorization and solver based on the PARDISO library
  *
  * This class allows to solve for A.X = B sparse linear problems via a direct LU factorization
  * using the Intel MKL PARDISO library. The sparse matrix A must be squared and invertible.
  * The vectors or matrices X and B can be either dense or sparse.
  *
  * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
  *
  * \sa \ref TutorialSparseDirectSolvers
  */
template<typename MatrixType>
class PardisoLU : public PardisoImpl< PardisoLU<MatrixType> >
{
  protected:
    typedef PardisoImpl< PardisoLU<MatrixType> > Base;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;
    using Base::m_type;

  public:

    using Base::compute;
    using Base::solve;

    PardisoLU(int flags = Metis)
      : Base(flags)
    {
      m_type = Base::ScalarIsComplex ? 13 : 11;
    }

    PardisoLU(const MatrixType& matrix, int flags = Metis)
      : Base(flags)
    {
      m_type = Base::ScalarIsComplex ? 13 : 11;
      compute(matrix);
    }
};

/** \ingroup PARDISOSupport_Module
  * \class PardisoLLT
  * \brief A sparse direct Cholesky (LLT) factorization and solver based on the PARDISO library
  *
  * This class allows to solve for A.X = B sparse linear problems via a LL^T Cholesky factorization
  * using the Intel MKL PARDISO library. The sparse matrix A must be selfajoint and positive definite.
  * The vectors or matrices X and B can be either dense or sparse.
  *
  * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
  *
  * \sa \ref TutorialSparseDirectSolvers
  */
template<typename MatrixType>
class PardisoLLT : public PardisoImpl< PardisoLLT<MatrixType> >
{
  protected:
    typedef PardisoImpl< PardisoLLT<MatrixType> > Base;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;
    using Base::m_type;

  public:

    using Base::compute;
    using Base::solve;

    PardisoLLT(int flags = Metis)
      : Base(flags)
    {
      m_type = Base::ScalarIsComplex ? 4 : 2;
    }

    PardisoLLT(const MatrixType& matrix, int flags = Metis)
      : Base(flags)
    {
      m_type = Base::ScalarIsComplex ? 4 : 2;
      compute(matrix);
    }
};

/** \ingroup PARDISOSupport_Module
  * \class PardisoLDLT
  * \brief A sparse direct Cholesky (LLT) factorization and solver based on the PARDISO library
  *
  * This class allows to solve for A.X = B sparse linear problems via a LDL^T Cholesky factorization
  * using the Intel MKL PARDISO library. The sparse matrix A must be selfajoint and positive definite.
  * The vectors or matrices X and B can be either dense or sparse.
  *
  * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
  *
  * \sa \ref TutorialSparseDirectSolvers
  */
template<typename MatrixType>
class PardisoLDLT : public PardisoImpl< PardisoLDLT<MatrixType> >
{
  protected:
    typedef PardisoImpl< PardisoLDLT<MatrixType> > Base;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;
    using Base::m_type;

  public:

    using Base::compute;
    using Base::solve;

    PardisoLDLT(int flags = Metis)
      : Base(flags)
    {
      m_type = Base::ScalarIsComplex ? -4 : -2;
    }

    PardisoLDLT(const MatrixType& matrix, int flags = Metis, bool hermitian = true)
      : Base(flags)
    {
      compute(matrix, hermitian);
    }

    void compute(const MatrixType& matrix, bool hermitian = true)
    {
      m_type = Base::ScalarIsComplex ? (hermitian ? -4 : 6) : -2;
      Base::compute(matrix);
    }

};

namespace internal {
  
template<typename _Derived, typename Rhs>
struct solve_retval<PardisoImpl<_Derived>, Rhs>
  : solve_retval_base<PardisoImpl<_Derived>, Rhs>
{
  typedef PardisoImpl<_Derived> Dec;
  EIGEN_MAKE_SOLVE_HELPERS(Dec,Rhs)

  solve_retval(const PardisoImpl<_Derived>& dec, const Rhs& rhs, const int transposed)
    : Base(dec, rhs), m_transposed(transposed) {}

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dec()._solve(rhs(),dst,m_transposed);
  }

  int m_transposed;
};

}

#endif // EIGEN_PARDISOSUPPORT_H
