// -*- coding: utf-8
// vim: set fileencoding=utf-8

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>
// Copyright (C) 2012 Desire Nuentsa <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_LEVENBERGMARQUARDT_H
#define EIGEN_LEVENBERGMARQUARDT_H


namespace Eigen {
namespace LevenbergMarquardtSpace {
    enum Status {
        NotStarted = -2,
        Running = -1,
        ImproperInputParameters = 0,
        RelativeReductionTooSmall = 1,
        RelativeErrorTooSmall = 2,
        RelativeErrorAndReductionTooSmall = 3,
        CosinusTooSmall = 4,
        TooManyFunctionEvaluation = 5,
        FtolTooSmall = 6,
        XtolTooSmall = 7,
        GtolTooSmall = 8,
        UserAsked = 9
    };
}

template <typename _Scalar, int NX=Dynamic, int NY=Dynamic>
struct DenseFunctor
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
  typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
  typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;
  typedef ColPivHouseholderQR<JacobianType> QRSolver;
  const int m_inputs, m_values;

  DenseFunctor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  DenseFunctor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  //int operator()(const InputType &x, ValueType& fvec) { }
  // should be defined in derived classes
  
  //int df(const InputType &x, JacobianType& fjac) { }
  // should be defined in derived classes
};

#ifdef EIGEN_SPQR_SUPPORT
template <typename _Scalar, typename _Index>
struct SparseFunctor
{
  typedef _Scalar Scalar;
  typedef _Index Index;
  typedef Matrix<Scalar,Dynamic,1> InputType;
  typedef Matrix<Scalar,Dynamic,1> ValueType;
  typedef SparseMatrix<Scalar, ColMajor, Index> JacobianType;
  typedef SPQR<JacobianType> QRSolver;
  
  SparseFunctor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }
  
  const int m_inputs, m_values;
  //int operator()(const InputType &x, ValueType& fvec) { }
  // to be defined in the functor
  
  //int df(const InputType &x, JacobianType& fjac) { }
  // to be defined in the functor if no automatic differentiation
  
};
#endif
namespace internal {
template <typename QRSolver, typename VectorType>
void lmpar2(const QRSolver &qr, const VectorType  &diag, const VectorType  &qtb,
	    typename VectorType::Scalar m_delta, typename VectorType::Scalar &par,
	    VectorType  &x);
    }
/**
  * \ingroup NonLinearOptimization_Module
  * \brief Performs non linear optimization over a non-linear function,
  * using a variant of the Levenberg Marquardt algorithm.
  *
  * Check wikipedia for more information.
  * http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
  */
template<typename _FunctorType>
class LevenbergMarquardt
{
  public:
    typedef _FunctorType FunctorType;
    typedef typename FunctorType::QRSolver QRSolver;
    typedef typename FunctorType::JacobianType JacobianType;
    typedef typename JacobianType::Scalar Scalar;
    typedef typename JacobianType::RealScalar RealScalar; 
    typedef typename JacobianType::Index Index;
    typedef typename QRSolver::Index PermIndex;
    typedef Matrix<Scalar,Dynamic,1> FVectorType;
    typedef PermutationMatrix<Dynamic,Dynamic> PermutationType;
  public:
    LevenbergMarquardt(FunctorType& functor) 
    : m_functor(functor),m_nfev(0),m_njev(0),m_fnorm(0.0),m_gnorm(0)
    {
      resetParameters();
      m_useExternalScaling=false; 
    }
    
    LevenbergMarquardtSpace::Status minimize(FVectorType &x);
    LevenbergMarquardtSpace::Status minimizeInit(FVectorType &x);
    LevenbergMarquardtSpace::Status minimizeOneStep(FVectorType &x);
    LevenbergMarquardtSpace::Status lmder1(
      FVectorType  &x, 
      const Scalar tol = std::sqrt(NumTraits<Scalar>::epsilon())
    );
    static LevenbergMarquardtSpace::Status lmdif1(
            FunctorType &functor,
            FVectorType  &x,
            Index *nfev,
            const Scalar tol = std::sqrt(NumTraits<Scalar>::epsilon())
            );
    
    /** Sets the default parameters */
    void resetParameters() 
    { 
      m_factor = 100.; 
      m_maxfev = 400; 
      m_ftol = std::sqrt(NumTraits<RealScalar>::epsilon());
      m_xtol = std::sqrt(NumTraits<RealScalar>::epsilon());
      m_gtol = 0. ; 
      m_epsfcn = 0. ;
    }
    
    /** Sets the tolerance for the norm of the solution vector*/
    void setXtol(RealScalar xtol) { m_xtol = xtol; }
    
    /** Sets the tolerance for the norm of the vector function*/
    void setFtol(RealScalar ftol) { m_ftol = ftol; }
    
    /** Sets the tolerance for the norm of the gradient of the error vector*/
    void setGtol(RealScalar gtol) { m_gtol = gtol; }
    
    /** Sets the step bound for the diagonal shift */
    void setFactor(RealScalar factor) { m_factor = factor; }    
    
    /** Sets the error precision  */
    void setEpsilon (RealScalar epsfcn) { m_epsfcn = epsfcn; }
    
    /** Sets the maximum number of function evaluation */
    void setMaxfev(Index maxfev) {m_maxfev = maxfev; }
    
    /** Use an external Scaling. If set to true, pass a nonzero diagonal to diag() */
    void setExternalScaling(bool value) {m_useExternalScaling  = value; }
    
    /** Get a reference to the diagonal of the jacobian */
    FVectorType& diag() {return m_diag; }
    
    /** Number of iterations performed */
    Index iterations() { return m_iter; }
    
    /** Number of functions evaluation */
    Index nfev() { return m_nfev; }
    
    /** Number of jacobian evaluation */
    Index njev() { return m_njev; }
    
    /** Norm of current vector function */
    RealScalar fnorm() {return m_fnorm; }
    
    /** Norm of the gradient of the error */
    RealScalar gnorm() {return m_gnorm; }
    
    /** the LevenbergMarquardt parameter */
    RealScalar lm_param(void) { return m_par; }
    
    /** reference to the  current vector function 
     */
    FVectorType& fvec() {return m_fvec; }
    
    /** reference to the matrix where the current Jacobian matrix is stored
     */
    JacobianType& fjac() {return m_fjac; }
    
    /** the permutation used
     */
    PermutationType permutation() {return m_permutation; }
    
  private:
    JacobianType m_fjac; 
    FunctorType &m_functor;
    FVectorType m_fvec, m_qtf, m_diag; 
    Index n;
    Index m; 
    Index m_nfev;
    Index m_njev; 
    RealScalar m_fnorm; // Norm of the current vector function
    RealScalar m_gnorm; //Norm of the gradient of the error 
    RealScalar m_factor; //
    Index m_maxfev; // Maximum number of function evaluation
    RealScalar m_ftol; //Tolerance in the norm of the vector function
    RealScalar m_xtol; // 
    RealScalar m_gtol; //tolerance of the norm of the error gradient
    RealScalar m_epsfcn; //
    Index m_iter; // Number of iterations performed
    RealScalar m_delta;
    bool m_useExternalScaling;
    PermutationType m_permutation;
    FVectorType m_wa1, m_wa2, m_wa3, m_wa4; //Temporary vectors
    RealScalar m_par;
};

template<typename FunctorType>
LevenbergMarquardtSpace::Status
LevenbergMarquardt<FunctorType>::minimize(FVectorType  &x)
{
    LevenbergMarquardtSpace::Status status = minimizeInit(x);
    if (status==LevenbergMarquardtSpace::ImproperInputParameters)
        return status;
    do {
//       std::cout << " uv " << x.transpose() << "\n";
        status = minimizeOneStep(x);
    } while (status==LevenbergMarquardtSpace::Running);
    return status;
}

template<typename FunctorType>
LevenbergMarquardtSpace::Status
LevenbergMarquardt<FunctorType>::minimizeInit(FVectorType  &x)
{
    n = x.size();
    m = m_functor.values();

    m_wa1.resize(n); m_wa2.resize(n); m_wa3.resize(n);
    m_wa4.resize(m);
    m_fvec.resize(m);
    //FIXME Sparse Case : Allocate space for the jacobian
    m_fjac.resize(m, n);
//     m_fjac.reserve(VectorXi::Constant(n,5)); // FIXME Find a better alternative
    if (!m_useExternalScaling)
        m_diag.resize(n);
    assert( (!m_useExternalScaling || m_diag.size()==n) || "When m_useExternalScaling is set, the caller must provide a valid 'm_diag'");
    m_qtf.resize(n);

    /* Function Body */
    m_nfev = 0;
    m_njev = 0;

    /*     check the input parameters for errors. */
    if (n <= 0 || m < n || m_ftol < 0. || m_xtol < 0. || m_gtol < 0. || m_maxfev <= 0 || m_factor <= 0.)
        return LevenbergMarquardtSpace::ImproperInputParameters;

    if (m_useExternalScaling)
        for (Index j = 0; j < n; ++j)
            if (m_diag[j] <= 0.)
                return LevenbergMarquardtSpace::ImproperInputParameters;

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */
    m_nfev = 1;
    if ( m_functor(x, m_fvec) < 0)
        return LevenbergMarquardtSpace::UserAsked;
    m_fnorm = m_fvec.stableNorm();

    /*     initialize levenberg-marquardt parameter and iteration counter. */
    m_par = 0.;
    m_iter = 1;

    return LevenbergMarquardtSpace::NotStarted;
}

template<typename FunctorType>
LevenbergMarquardtSpace::Status
LevenbergMarquardt<FunctorType>::minimizeOneStep(FVectorType  &x)
{
  typedef typename FunctorType::JacobianType JacobianType; 
  using std::abs;
  using std::sqrt;
  RealScalar temp, temp1,temp2; 
  RealScalar ratio; 
  RealScalar pnorm, xnorm, fnorm1, actred, dirder, prered;
  assert(x.size()==n); // check the caller is not cheating us

  temp = 0.0; xnorm = 0.0;
  /* calculate the jacobian matrix. */
  Index df_ret = m_functor.df(x, m_fjac);
  if (df_ret<0)
      return LevenbergMarquardtSpace::UserAsked;
  if (df_ret>0)
      // numerical diff, we evaluated the function df_ret times
      m_nfev += df_ret;
  else m_njev++;

  /* compute the qr factorization of the jacobian. */
  for (int j = 0; j < x.size(); ++j)
    m_wa2(j) = m_fjac.col(j).norm(); 
  //FIXME Implement bluenorm for sparse vectors
//     m_wa2 = m_fjac.colwise().blueNorm();
  QRSolver qrfac(m_fjac); //FIXME Check if the QR decomposition succeed
  // Make a copy of the first factor with the associated permutation
  JacobianType rfactor;
  rfactor = qrfac.matrixQR();
  m_permutation = (qrfac.colsPermutation());

  /* on the first iteration and if external scaling is not used, scale according */
  /* to the norms of the columns of the initial jacobian. */
  if (m_iter == 1) {
      if (!m_useExternalScaling)
          for (Index j = 0; j < n; ++j)
              m_diag[j] = (m_wa2[j]==0.)? 1. : m_wa2[j];

      /* on the first iteration, calculate the norm of the scaled x */
      /* and initialize the step bound m_delta. */
      xnorm = m_diag.cwiseProduct(x).stableNorm();
      m_delta = m_factor * xnorm;
      if (m_delta == 0.)
          m_delta = m_factor;
  }

  /* form (q transpose)*m_fvec and store the first n components in */
  /* m_qtf. */
  m_wa4 = m_fvec;
  m_wa4 = qrfac.matrixQ().adjoint() * m_fvec; 
  m_qtf = m_wa4.head(n);

  /* compute the norm of the scaled gradient. */
  m_gnorm = 0.;
  if (m_fnorm != 0.)
      for (Index j = 0; j < n; ++j)
          if (m_wa2[m_permutation.indices()[j]] != 0.)
              m_gnorm = (std::max)(m_gnorm, abs( rfactor.col(j).head(j+1).dot(m_qtf.head(j+1)/m_fnorm) / m_wa2[m_permutation.indices()[j]]));

  /* test for convergence of the gradient norm. */
  if (m_gnorm <= m_gtol)
      return LevenbergMarquardtSpace::CosinusTooSmall;

  /* rescale if necessary. */
  if (!m_useExternalScaling)
      m_diag = m_diag.cwiseMax(m_wa2);

  do {
    /* determine the levenberg-marquardt parameter. */
    internal::lmpar2(qrfac, m_diag, m_qtf, m_delta, m_par, m_wa1);

    /* store the direction p and x + p. calculate the norm of p. */
    m_wa1 = -m_wa1;
    m_wa2 = x + m_wa1;
    pnorm = m_diag.cwiseProduct(m_wa1).stableNorm();

    /* on the first iteration, adjust the initial step bound. */
    if (m_iter == 1)
        m_delta = (std::min)(m_delta,pnorm);

    /* evaluate the function at x + p and calculate its norm. */
    if ( m_functor(m_wa2, m_wa4) < 0)
        return LevenbergMarquardtSpace::UserAsked;
    ++m_nfev;
    fnorm1 = m_wa4.stableNorm();

    /* compute the scaled actual reduction. */
    actred = -1.;
    if (Scalar(.1) * fnorm1 < m_fnorm)
        actred = 1. - internal::abs2(fnorm1 / m_fnorm);

    /* compute the scaled predicted reduction and */
    /* the scaled directional derivative. */
    m_wa3 = rfactor.template triangularView<Upper>() * (m_permutation.inverse() *m_wa1);
    temp1 = internal::abs2(m_wa3.stableNorm() / m_fnorm);
    temp2 = internal::abs2(sqrt(m_par) * pnorm / m_fnorm);
    prered = temp1 + temp2 / Scalar(.5);
    dirder = -(temp1 + temp2);

    /* compute the ratio of the actual to the predicted */
    /* reduction. */
    ratio = 0.;
    if (prered != 0.)
        ratio = actred / prered;

    /* update the step bound. */
    if (ratio <= Scalar(.25)) {
        if (actred >= 0.)
            temp = RealScalar(.5);
        if (actred < 0.)
            temp = RealScalar(.5) * dirder / (dirder + RealScalar(.5) * actred);
        if (RealScalar(.1) * fnorm1 >= m_fnorm || temp < RealScalar(.1))
            temp = Scalar(.1);
        /* Computing MIN */
        m_delta = temp * (std::min)(m_delta, pnorm / RealScalar(.1));
        m_par /= temp;
    } else if (!(m_par != 0. && ratio < RealScalar(.75))) {
        m_delta = pnorm / RealScalar(.5);
        m_par = RealScalar(.5) * m_par;
    }

    /* test for successful iteration. */
    if (ratio >= RealScalar(1e-4)) {
        /* successful iteration. update x, m_fvec, and their norms. */
        x = m_wa2;
        m_wa2 = m_diag.cwiseProduct(x);
        m_fvec = m_wa4;
        xnorm = m_wa2.stableNorm();
        m_fnorm = fnorm1;
        ++m_iter;
    }

    /* tests for convergence. */
    if (abs(actred) <= m_ftol && prered <= m_ftol && Scalar(.5) * ratio <= 1. && m_delta <= m_xtol * xnorm)
        return LevenbergMarquardtSpace::RelativeErrorAndReductionTooSmall;
    if (abs(actred) <= m_ftol && prered <= m_ftol && Scalar(.5) * ratio <= 1.)
        return LevenbergMarquardtSpace::RelativeReductionTooSmall;
    if (m_delta <= m_xtol * xnorm)
        return LevenbergMarquardtSpace::RelativeErrorTooSmall;

    /* tests for termination and stringent tolerances. */
    if (m_nfev >= m_maxfev)
        return LevenbergMarquardtSpace::TooManyFunctionEvaluation;
    if (abs(actred) <= NumTraits<Scalar>::epsilon() && prered <= NumTraits<Scalar>::epsilon() && Scalar(.5) * ratio <= 1.)
        return LevenbergMarquardtSpace::FtolTooSmall;
    if (m_delta <= NumTraits<Scalar>::epsilon() * xnorm)
        return LevenbergMarquardtSpace::XtolTooSmall;
    if (m_gnorm <= NumTraits<Scalar>::epsilon())
        return LevenbergMarquardtSpace::GtolTooSmall;

  } while (ratio < Scalar(1e-4));

  return LevenbergMarquardtSpace::Running;
}

template<typename FunctorType>
LevenbergMarquardtSpace::Status
LevenbergMarquardt<FunctorType>::lmder1(
        FVectorType  &x,
        const Scalar tol
        )
{
    n = x.size();
    m = m_functor.values();

    /* check the input parameters for errors. */
    if (n <= 0 || m < n || tol < 0.)
        return LevenbergMarquardtSpace::ImproperInputParameters;

    resetParameters();
    m_ftol = tol;
    m_xtol = tol;
    m_maxfev = 100*(n+1);

    return minimize(x);
}


template<typename FunctorType>
LevenbergMarquardtSpace::Status
LevenbergMarquardt<FunctorType>::lmdif1(
        FunctorType &functor,
        FVectorType  &x,
        Index *nfev,
        const Scalar tol
        )
{
    Index n = x.size();
    Index m = functor.values();

    /* check the input parameters for errors. */
    if (n <= 0 || m < n || tol < 0.)
        return LevenbergMarquardtSpace::ImproperInputParameters;

    NumericalDiff<FunctorType> numDiff(functor);
    // embedded LevenbergMarquardt
    LevenbergMarquardt<NumericalDiff<FunctorType> > lm(numDiff);
    lm.setFtol(tol);
    lm.setXtol(tol);
    lm.setMaxfev(200*(n+1));

    LevenbergMarquardtSpace::Status info = LevenbergMarquardtSpace::Status(lm.minimize(x));
    if (nfev)
        * nfev = lm.nfev();
    return info;
}

namespace internal {
  
  template <typename QRSolver, typename VectorType>
    void lmpar2(
		const QRSolver &qr,
		const VectorType  &diag,
		const VectorType  &qtb,
		typename VectorType::Scalar m_delta,
		typename VectorType::Scalar &par,
		VectorType  &x)

  {
    using std::sqrt;
    using std::abs;
    typedef typename QRSolver::MatrixType MatrixType;
    typedef typename QRSolver::Scalar Scalar;
    typedef typename QRSolver::Index Index;

    /* Local variables */
    Index j;
    Scalar fp;
    Scalar parc, parl;
    Index iter;
    Scalar temp, paru;
    Scalar gnorm;
    Scalar dxnorm;


    /* Function Body */
    const Scalar dwarf = (std::numeric_limits<Scalar>::min)();
    const Index n = qr.matrixQR().cols();
    assert(n==diag.size());
    assert(n==qtb.size());

    VectorType  wa1, wa2;

    /* compute and store in x the gauss-newton direction. if the */
    /* jacobian is rank-deficient, obtain a least squares solution. */

    //    const Index rank = qr.nonzeroPivots(); // exactly double(0.)
    const Index rank = qr.rank(); // use a threshold
    wa1 = qtb;
    wa1.tail(n-rank).setZero();
    //FIXME There is no solve in place for sparse triangularView
    //qr.matrixQR().topLeftCorner(rank, rank).template triangularView<Upper>().solveInPlace(wa1.head(rank));
    wa1.head(rank) = qr.matrixQR().topLeftCorner(rank, rank).template triangularView<Upper>().solve(qtb.head(rank));

    x = qr.colsPermutation()*wa1;

    /* initialize the iteration counter. */
    /* evaluate the function at the origin, and test */
    /* for acceptance of the gauss-newton direction. */
    iter = 0;
    wa2 = diag.cwiseProduct(x);
    dxnorm = wa2.blueNorm();
    fp = dxnorm - m_delta;
    if (fp <= Scalar(0.1) * m_delta) {
      par = 0;
      return;
    }

    /* if the jacobian is not rank deficient, the newton */
    /* step provides a lower bound, parl, for the zero of */
    /* the function. otherwise set this bound to zero. */
    parl = 0.;
    if (rank==n) {
      wa1 = qr.colsPermutation().inverse() *  diag.cwiseProduct(wa2)/dxnorm;
      qr.matrixQR().topLeftCorner(n, n).transpose().template triangularView<Lower>().solveInPlace(wa1);
      temp = wa1.blueNorm();
      parl = fp / m_delta / temp / temp;
    }

    /* calculate an upper bound, paru, for the zero of the function. */
    for (j = 0; j < n; ++j)
      wa1[j] = qr.matrixQR().col(j).head(j+1).dot(qtb.head(j+1)) / diag[qr.colsPermutation().indices()(j)];

    gnorm = wa1.stableNorm();
    paru = gnorm / m_delta;
    if (paru == 0.)
      paru = dwarf / (std::min)(m_delta,Scalar(0.1));

    /* if the input par lies outside of the interval (parl,paru), */
    /* set par to the closer endpoint. */
    par = (std::max)(par,parl);
    par = (std::min)(par,paru);
    if (par == 0.)
      par = gnorm / dxnorm;

    /* beginning of an iteration. */
    MatrixType s;
    s = qr.matrixQR();
    while (true) {
      ++iter;

      /* evaluate the function at the current value of par. */
      if (par == 0.)
	par = (std::max)(dwarf,Scalar(.001) * paru); /* Computing MAX */
      wa1 = sqrt(par)* diag;

      VectorType sdiag(n);
      lmqrsolv(s, qr.colsPermutation(), wa1, qtb, x, sdiag);

      wa2 = diag.cwiseProduct(x);
      dxnorm = wa2.blueNorm();
      temp = fp;
      fp = dxnorm - m_delta;

      /* if the function is small enough, accept the current value */
      /* of par. also test for the exceptional cases where parl */
      /* is zero or the number of iterations has reached 10. */
      if (abs(fp) <= Scalar(0.1) * m_delta || (parl == 0. && fp <= temp && temp < 0.) || iter == 10)
	break;

      /* compute the newton correction. */
      wa1 = qr.colsPermutation().inverse() * diag.cwiseProduct(wa2/dxnorm);
      // we could almost use this here, but the diagonal is outside qr, in sdiag[]
      // qr.matrixQR().topLeftCorner(n, n).transpose().template triangularView<Lower>().solveInPlace(wa1);
      for (j = 0; j < n; ++j) {
	wa1[j] /= sdiag[j];
	temp = wa1[j];
	for (Index i = j+1; i < n; ++i)
	  wa1[i] -= s.coeff(i,j) * temp;
      }
      temp = wa1.blueNorm();
      parc = fp / m_delta / temp / temp;

      /* depending on the sign of the function, update parl or paru. */
      if (fp > 0.)
	parl = (std::max)(parl,par);
      if (fp < 0.)
	paru = (std::min)(paru,par);

      /* compute an improved estimate for par. */
      par = (std::max)(parl,par+parc);
    }
    if (iter == 0)
      par = 0.;
    return;
  }
} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_LEVENBERGMARQUARDT_H
