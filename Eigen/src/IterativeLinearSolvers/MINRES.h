// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Giacomo Po <gpo@ucla.edu>
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef EIGEN_MINRES_H_
#define EIGEN_MINRES_H_


namespace Eigen {
    
    namespace internal {
        
        /** \internal Low-level MINRES algorithm
         * \param mat The matrix A
         * \param rhs The right hand side vector b
         * \param x On input and initial solution, on output the computed solution.
         * \param precond A preconditioner being able to efficiently solve for an
         *                approximation of Ax=b (regardless of b)
         * \param iters On input the max number of iteration, on output the number of performed iterations.
         * \param tol_error On input the tolerance error, on output an estimation of the relative error.
         */
        template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
        EIGEN_DONT_INLINE
        void minres(const MatrixType& mat, const Rhs& rhs, Dest& x,
                    const Preconditioner& precond, int& iters,
                    typename Dest::RealScalar& tol_error)
        {
            typedef typename Dest::RealScalar RealScalar;
            typedef typename Dest::Scalar Scalar;
            typedef Matrix<Scalar,Dynamic,1> VectorType;
            // initialize
            const int maxIters(iters);  // initialize maxIters to iters
            const int N(mat.cols());    // the size of the matrix
            const RealScalar threshold(tol_error); // convergence threshold
            VectorType v(VectorType::Zero(N));
            VectorType v_hat(rhs-mat*x);
            RealScalar beta(v_hat.norm());
            RealScalar c(1.0); // the cosine of the Givens rotation
            RealScalar c_old(1.0);
            RealScalar s(0.0); // the sine of the Givens rotation
            RealScalar s_old(0.0); // the sine of the Givens rotation
            VectorType w(VectorType::Zero(N));
            VectorType w_old(w);
            RealScalar eta(beta);
            RealScalar norm_rMR=beta;
            const RealScalar norm_r0(beta);
            
            int n = 0;
            while ( n < maxIters ){
                
                
                // Lanczos
                VectorType v_old(v);
                v=v_hat/beta;
                VectorType Av(mat*v);
                RealScalar alpha(v.transpose()*Av);
                v_hat=Av-alpha*v-beta*v_old;
                RealScalar beta_old(beta);
                beta=v_hat.norm();
                
                // QR
                RealScalar c_oold(c_old);
                c_old=c;
                RealScalar s_oold(s_old);
                s_old=s;
                RealScalar r1_hat=c_old *alpha-c_oold*s_old *beta_old;
                RealScalar r1 =std::pow(std::pow(r1_hat,2)+std::pow(beta,2),0.5);
                RealScalar r2 =s_old *alpha+c_oold*c_old*beta_old;
                RealScalar r3 =s_oold*beta_old;
                
                // Givens rotation
                c=r1_hat/r1;
                s=beta/r1;
                
                // update
                VectorType w_oold(w_old);
                w_old=w;
                w=(v-r3*w_oold-r2*w_old) /r1;
                x += c*eta*w;
                norm_rMR *= std::fabs(s);
                eta=-s*eta;
                //if(norm_rMR/norm_r0 < threshold){
                    if ( (mat*x-rhs).norm()/rhs.norm() < threshold){
                    break;
                }
                n++;
            }
            tol_error = (mat*x-rhs).norm()/rhs.norm(); // return error DOES mat*x NEED TO BE RECOMPUTED???
            iters = n;  // return number of iterations
        }
        
    }
    
    template< typename _MatrixType, int _UpLo=Lower,
    typename _Preconditioner = DiagonalPreconditioner<typename _MatrixType::Scalar> >
    class MINRES;
    
    namespace internal {
        
        template< typename _MatrixType, int _UpLo, typename _Preconditioner>
        struct traits<MINRES<_MatrixType,_UpLo,_Preconditioner> >
        {
            typedef _MatrixType MatrixType;
            typedef _Preconditioner Preconditioner;
        };
        
    }
    
    /** \ingroup IterativeLinearSolvers_Module
     * \brief A minimal residual solver for sparse symmetric problems
     *
     * This class allows to solve for A.x = b sparse linear problems using the MINRES algorithm
     * of Paige and Saunders (1975). The sparse matrix A must be symmetric (possibly indefinite).
     * The vectors x and b can be either dense or sparse.
     *
     * \tparam _MatrixType the type of the sparse matrix A, can be a dense or a sparse matrix.
     * \tparam _UpLo the triangular part that will be used for the computations. It can be Lower
     *               or Upper. Default is Lower.
     * \tparam _Preconditioner the type of the preconditioner. Default is DiagonalPreconditioner
     *
     * The maximal number of iterations and tolerance value can be controlled via the setMaxIterations()
     * and setTolerance() methods. The defaults are the size of the problem for the maximal number of iterations
     * and NumTraits<Scalar>::epsilon() for the tolerance.
     *
     * This class can be used as the direct solver classes. Here is a typical usage example:
     * \code
     * int n = 10000;
     * VectorXd x(n), b(n);
     * SparseMatrix<double> A(n,n);
     * // fill A and b
     * MINRES<SparseMatrix<double> > mr;
     * mr.compute(A);
     * x = mr.solve(b);
     * std::cout << "#iterations:     " << mr.iterations() << std::endl;
     * std::cout << "estimated error: " << mr.error()      << std::endl;
     * // update b, and solve again
     * x = mr.solve(b);
     * \endcode
     *
     * By default the iterations start with x=0 as an initial guess of the solution.
     * One can control the start using the solveWithGuess() method. Here is a step by
     * step execution example starting with a random guess and printing the evolution
     * of the estimated error:
     * * \code
     * x = VectorXd::Random(n);
     * mr.setMaxIterations(1);
     * int i = 0;
     * do {
     *   x = mr.solveWithGuess(b,x);
     *   std::cout << i << " : " << mr.error() << std::endl;
     *   ++i;
     * } while (mr.info()!=Success && i<100);
     * \endcode
     * Note that such a step by step excution is slightly slower.
     *
     * \sa class ConjugateGradient, BiCGSTAB, SimplicialCholesky, DiagonalPreconditioner, IdentityPreconditioner
     */
    template< typename _MatrixType, int _UpLo, typename _Preconditioner>
    class MINRES : public IterativeSolverBase<MINRES<_MatrixType,_UpLo,_Preconditioner> >
    {
        
        typedef IterativeSolverBase<MINRES> Base;
        using Base::mp_matrix;
        using Base::m_error;
        using Base::m_iterations;
        using Base::m_info;
        using Base::m_isInitialized;
    public:
        typedef _MatrixType MatrixType;
        typedef typename MatrixType::Scalar Scalar;
        typedef typename MatrixType::Index Index;
        typedef typename MatrixType::RealScalar RealScalar;
        typedef _Preconditioner Preconditioner;
        
        enum {UpLo = _UpLo};
        
    public:
        
        /** Default constructor. */
        MINRES() : Base() {}
        
        /** Initialize the solver with matrix \a A for further \c Ax=b solving.
         *
         * This constructor is a shortcut for the default constructor followed
         * by a call to compute().
         *
         * \warning this class stores a reference to the matrix A as well as some
         * precomputed values that depend on it. Therefore, if \a A is changed
         * this class becomes invalid. Call compute() to update it with the new
         * matrix A, or modify a copy of A.
         */
        MINRES(const MatrixType& A) : Base(A) {}
        
        /** Destructor. */
        ~MINRES(){}
		
        /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A
         * \a x0 as an initial solution.
         *
         * \sa compute()
         */
        template<typename Rhs,typename Guess>
        inline const internal::solve_retval_with_guess<MINRES, Rhs, Guess>
        solveWithGuess(const MatrixBase<Rhs>& b, const Guess& x0) const
        {
            eigen_assert(m_isInitialized && "MINRES is not initialized.");
            eigen_assert(Base::rows()==b.rows()
                         && "MINRES::solve(): invalid number of rows of the right hand side matrix b");
            return internal::solve_retval_with_guess
            <MINRES, Rhs, Guess>(*this, b.derived(), x0);
        }
        
        /** \internal */
        template<typename Rhs,typename Dest>
        void _solveWithGuess(const Rhs& b, Dest& x) const
        {
            m_iterations = Base::maxIterations();
            m_error = Base::m_tolerance;
            
            for(int j=0; j<b.cols(); ++j)
            {
                m_iterations = Base::maxIterations();
                m_error = Base::m_tolerance;
                
                typename Dest::ColXpr xj(x,j);
                internal::minres(mp_matrix->template selfadjointView<UpLo>(), b.col(j), xj,
                                 Base::m_preconditioner, m_iterations, m_error);
            }
            
            m_isInitialized = true;
            m_info = m_error <= Base::m_tolerance ? Success : NoConvergence;
        }
        
        /** \internal */
        template<typename Rhs,typename Dest>
        void _solve(const Rhs& b, Dest& x) const
        {
            x.setOnes();
            _solveWithGuess(b,x);
        }
        
    protected:
        
    };
    
    namespace internal {
        
        template<typename _MatrixType, int _UpLo, typename _Preconditioner, typename Rhs>
        struct solve_retval<MINRES<_MatrixType,_UpLo,_Preconditioner>, Rhs>
        : solve_retval_base<MINRES<_MatrixType,_UpLo,_Preconditioner>, Rhs>
        {
            typedef MINRES<_MatrixType,_UpLo,_Preconditioner> Dec;
            EIGEN_MAKE_SOLVE_HELPERS(Dec,Rhs)
            
            template<typename Dest> void evalTo(Dest& dst) const
            {
                dec()._solve(rhs(),dst);
            }
        };
        
    } // end namespace internal
    
} // end namespace Eigen

#endif // EIGEN_MINRES_H

