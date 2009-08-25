
template<typename FunctorType, typename Scalar=double>
class HybridNonLinearSolver
{
public:
    HybridNonLinearSolver(const FunctorType &_functor)
        : functor(_functor) {}

    enum Status {
        Running = -1,
        ImproperInputParameters = 0,
        RelativeErrorTooSmall = 1,
        TooManyFunctionEvaluation = 2, 
        TolTooSmall = 3,
        NotMakingProgressJacobian = 4,
        NotMakingProgressIterations = 5,
        UserAksed = 6
    };

    struct Parameters {
        Parameters()
            : factor(Scalar(100.))
            , maxfev(1000)
            , xtol(ei_sqrt(epsilon<Scalar>()))
            , nb_of_subdiagonals(-1)
            , nb_of_superdiagonals(-1)
            , epsfcn (Scalar(0.)) {}
        Scalar factor;
        int maxfev;   // maximum number of function evaluation
        Scalar xtol;
        int nb_of_subdiagonals;
        int nb_of_superdiagonals;
        Scalar epsfcn;
    };

    Status solve(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const Scalar tol = ei_sqrt(epsilon<Scalar>())
            );

    Status solveInit(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const Parameters &parameters,
            const int mode=1
            );
    Status solveOneStep(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const Parameters &parameters,
            const int mode=1
            );
    Status solve(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const Parameters &parameters,
            const int mode=1
            );

    Status solveNumericalDiff(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const Scalar tol = ei_sqrt(epsilon<Scalar>())
            );

    Status solveNumericalDiffInit(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const Parameters &parameters,
            const int mode=1
            );
    Status solveNumericalDiffOneStep(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const Parameters &parameters,
            const int mode=1
            );
    Status solveNumericalDiff(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const Parameters &parameters,
            const int mode=1
            );

    Matrix< Scalar, Dynamic, 1 >  fvec;
    Matrix< Scalar, Dynamic, Dynamic > fjac;
    Matrix< Scalar, Dynamic, 1 >  R;
    Matrix< Scalar, Dynamic, 1 >  qtf;
    Matrix< Scalar, Dynamic, 1 >  diag;
    int nfev;
    int njev;
private:
    const FunctorType &functor;
    int n;
    Scalar sum;
    bool sing;
    int iter;
    Scalar temp;
    Scalar delta;
    bool jeval;
    int ncsuc;
    Scalar ratio;
    Scalar fnorm;
    Scalar pnorm, xnorm, fnorm1;
    int nslow1, nslow2;
    int ncfail;
    Scalar actred, prered;
    Matrix< Scalar, Dynamic, 1 > wa1, wa2, wa3, wa4;
};



template<typename FunctorType, typename Scalar>
typename HybridNonLinearSolver<FunctorType,Scalar>::Status
HybridNonLinearSolver<FunctorType,Scalar>::solve(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const Scalar tol
        )
{
    n = x.size();
    Parameters parameters;

    /* check the input parameters for errors. */
    if (n <= 0 || tol < 0.) {
        printf("HybridNonLinearSolver::solve() bad args : n,tol,...");
        return ImproperInputParameters;
    }

    parameters.maxfev = 100*(n+1);
    parameters.xtol = tol;
    diag.setConstant(n, 1.);
    return solve(
        x, 
        parameters,
        2
    );
}

template<typename FunctorType, typename Scalar>
typename HybridNonLinearSolver<FunctorType,Scalar>::Status
HybridNonLinearSolver<FunctorType,Scalar>::solveInit(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const Parameters &parameters,
        const int mode
        )
{
    n = x.size();

    wa1.resize(n); wa2.resize(n); wa3.resize(n); wa4.resize(n);
    fvec.resize(n);
    qtf.resize(n);
    R.resize( (n*(n+1))/2);
    fjac.resize(n, n);
    if (mode != 2)
        diag.resize(n);
    assert( (mode!=2 || diag.size()==n) || "When using mode==2, the caller must provide a valid 'diag'");

    /* Function Body */
    nfev = 0;
    njev = 0;

    /*     check the input parameters for errors. */

    if (n <= 0 || parameters.xtol < 0. || parameters.maxfev <= 0 || parameters.factor <= 0. )
        return ImproperInputParameters;
    if (mode == 2)
        for (int j = 0; j < n; ++j)
            if (diag[j] <= 0.)
                return ImproperInputParameters;

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    nfev = 1;
    if ( functor.f(x, fvec) < 0)
        return UserAksed;
    fnorm = fvec.stableNorm();

    /*     initialize iteration counter and monitors. */

    iter = 1;
    ncsuc = 0;
    ncfail = 0;
    nslow1 = 0;
    nslow2 = 0;

    return Running;
}

template<typename FunctorType, typename Scalar>
typename HybridNonLinearSolver<FunctorType,Scalar>::Status
HybridNonLinearSolver<FunctorType,Scalar>::solveOneStep(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const Parameters &parameters,
        const int mode
        )
{
    int i, j, l, iwa[1];
    jeval = true;

    /* calculate the jacobian matrix. */

    if ( functor.df(x, fjac) < 0)
        return UserAksed;
    ++njev;

    /* compute the qr factorization of the jacobian. */

    ei_qrfac<Scalar>(n, n, fjac.data(), fjac.rows(), false, iwa, 1, wa1.data(), wa2.data());

    /* on the first iteration and if mode is 1, scale according */
    /* to the norms of the columns of the initial jacobian. */

    if (iter == 1) {
        if (mode != 2)
            for (j = 0; j < n; ++j) {
                diag[j] = wa2[j];
                if (wa2[j] == 0.)
                    diag[j] = 1.;
            }

        /* on the first iteration, calculate the norm of the scaled x */
        /* and initialize the step bound delta. */

        wa3 = diag.cwise() * x;
        xnorm = wa3.stableNorm();
        delta = parameters.factor * xnorm;
        if (delta == 0.)
            delta = parameters.factor;
    }

    /* form (q transpose)*fvec and store in qtf. */

    qtf = fvec;
    for (j = 0; j < n; ++j)
        if (fjac(j,j) != 0.) {
            sum = 0.;
            for (i = j; i < n; ++i)
                sum += fjac(i,j) * qtf[i];
            temp = -sum / fjac(j,j);
            for (i = j; i < n; ++i)
                qtf[i] += fjac(i,j) * temp;
        }

    /* copy the triangular factor of the qr factorization into r. */

    sing = false;
    for (j = 0; j < n; ++j) {
        l = j;
        if (j)
            for (i = 0; i < j; ++i) {
                R[l] = fjac(i,j);
                l = l + n - i -1;
            }
        R[l] = wa1[j];
        if (wa1[j] == 0.)
            sing = true;
    }

    /* accumulate the orthogonal factor in fjac. */

    ei_qform<Scalar>(n, n, fjac.data(), fjac.rows(), wa1.data());

    /* rescale if necessary. */

    /* Computing MAX */
    if (mode != 2)
        diag = diag.cwise().max(wa2);

    /* beginning of the inner loop. */

    while (true) {

        /* determine the direction p. */

        ei_dogleg<Scalar>(R, diag, qtf, delta, wa1);

        /* store the direction p and x + p. calculate the norm of p. */

        wa1 = -wa1;
        wa2 = x + wa1;
        wa3 = diag.cwise() * wa1;
        pnorm = wa3.stableNorm();

        /* on the first iteration, adjust the initial step bound. */

        if (iter == 1)
            delta = std::min(delta,pnorm);

        /* evaluate the function at x + p and calculate its norm. */

        if ( functor.f(wa2, wa4) < 0)
            return UserAksed;
        ++nfev;
        fnorm1 = wa4.stableNorm();

        /* compute the scaled actual reduction. */

        actred = -1.;
        if (fnorm1 < fnorm) /* Computing 2nd power */
            actred = 1. - ei_abs2(fnorm1 / fnorm);

        /* compute the scaled predicted reduction. */

        l = 0;
        for (i = 0; i < n; ++i) {
            sum = 0.;
            for (j = i; j < n; ++j) {
                sum += R[l] * wa1[j];
                ++l;
            }
            wa3[i] = qtf[i] + sum;
        }
        temp = wa3.stableNorm();
        prered = 0.;
        if (temp < fnorm) /* Computing 2nd power */
            prered = 1. - ei_abs2(temp / fnorm);

        /* compute the ratio of the actual to the predicted */
        /* reduction. */

        ratio = 0.;
        if (prered > 0.)
            ratio = actred / prered;

        /* update the step bound. */

        if (ratio < Scalar(.1)) {
            ncsuc = 0;
            ++ncfail;
            delta = Scalar(.5) * delta;
        } else {
            ncfail = 0;
            ++ncsuc;
            if (ratio >= Scalar(.5) || ncsuc > 1) /* Computing MAX */
                delta = std::max(delta, pnorm / Scalar(.5));
            if (ei_abs(ratio - 1.) <= Scalar(.1)) {
                delta = pnorm / Scalar(.5);
            }
        }

        /* test for successful iteration. */

        if (ratio >= Scalar(1e-4)) {
            /* successful iteration. update x, fvec, and their norms. */
            x = wa2;
            wa2 = diag.cwise() * x;
            fvec = wa4;
            xnorm = wa2.stableNorm();
            fnorm = fnorm1;
            ++iter;
        }

        /* determine the progress of the iteration. */

        ++nslow1;
        if (actred >= Scalar(.001))
            nslow1 = 0;
        if (jeval)
            ++nslow2;
        if (actred >= Scalar(.1))
            nslow2 = 0;

        /* test for convergence. */

        if (delta <= parameters.xtol * xnorm || fnorm == 0.)
            return RelativeErrorTooSmall;

        /* tests for termination and stringent tolerances. */

        if (nfev >= parameters.maxfev)
            return TooManyFunctionEvaluation;
        if (Scalar(.1) * std::max(Scalar(.1) * delta, pnorm) <= epsilon<Scalar>() * xnorm)
            return TolTooSmall;
        if (nslow2 == 5)
            return NotMakingProgressJacobian;
        if (nslow1 == 10)
            return NotMakingProgressIterations;

        /* criterion for recalculating jacobian. */

        if (ncfail == 2)
            break; // leave inner loop and go for the next outer loop iteration

        /* calculate the rank one modification to the jacobian */
        /* and update qtf if necessary. */

        for (j = 0; j < n; ++j) {
            sum = wa4.dot(fjac.col(j));
            wa2[j] = (sum - wa3[j]) / pnorm;
            wa1[j] = diag[j] * (diag[j] * wa1[j] / pnorm);
            if (ratio >= Scalar(1e-4))
                qtf[j] = sum;
        }

        /* compute the qr factorization of the updated jacobian. */

        ei_r1updt<Scalar>(n, n, R.data(), R.size(), wa1.data(), wa2.data(), wa3.data(), &sing);
        ei_r1mpyq<Scalar>(n, n, fjac.data(), fjac.rows(), wa2.data(), wa3.data());
        ei_r1mpyq<Scalar>(1, n, qtf.data(), 1, wa2.data(), wa3.data());

        /* end of the inner loop. */

        jeval = false;
    }
    /* end of the outer loop. */

    return Running;
}


template<typename FunctorType, typename Scalar>
typename HybridNonLinearSolver<FunctorType,Scalar>::Status
HybridNonLinearSolver<FunctorType,Scalar>::solve(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const Parameters &parameters,
        const int mode
        )
{
    Status status = solveInit(x, parameters, mode);
    while (status==Running)
        status = solveOneStep(x, parameters, mode);
    return status;
}



template<typename FunctorType, typename Scalar>
typename HybridNonLinearSolver<FunctorType,Scalar>::Status
HybridNonLinearSolver<FunctorType,Scalar>::solveNumericalDiff(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const Scalar tol
        )
{
    n = x.size();
    Parameters parameters;

    /* check the input parameters for errors. */
    if (n <= 0 || tol < 0.) {
        printf("HybridNonLinearSolver::solve() bad args : n,tol,...");
        return ImproperInputParameters;
    }

    parameters.maxfev = 200*(n+1);
    parameters.xtol = tol;

    diag.setConstant(n, 1.);
    return solveNumericalDiff(
        x,
        parameters,
        2
    );
}

template<typename FunctorType, typename Scalar>
typename HybridNonLinearSolver<FunctorType,Scalar>::Status
HybridNonLinearSolver<FunctorType,Scalar>::solveNumericalDiffInit(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const Parameters &parameters,
        const int mode
        )
{
    n = x.size();

    int nsub = parameters.nb_of_subdiagonals;
    int nsup = parameters.nb_of_superdiagonals;
    if (nsub<0) nsub= n-1;
    if (nsup<0) nsup= n-1;

    wa1.resize(n); wa2.resize(n); wa3.resize(n); wa4.resize(n);
    qtf.resize(n);
    R.resize( (n*(n+1))/2);
    fjac.resize(n, n);
    fvec.resize(n);
    if (mode != 2)
        diag.resize(n);
    assert( (mode!=2 || diag.size()==n) || "When using mode==2, the caller must provide a valid 'diag'");


    /* Function Body */

    nfev = 0;

    /*     check the input parameters for errors. */

    if (n <= 0 || parameters.xtol < 0. || parameters.maxfev <= 0 || nsub< 0 || nsup< 0 || parameters.factor <= 0. )
        return ImproperInputParameters;
    if (mode == 2)
        for (int j = 0; j < n; ++j)
            if (diag[j] <= 0.)
                return ImproperInputParameters;

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    nfev = 1;
    if ( functor.f(x, fvec) < 0)
        return UserAksed;
    fnorm = fvec.stableNorm();

    /*     initialize iteration counter and monitors. */

    iter = 1;
    ncsuc = 0;
    ncfail = 0;
    nslow1 = 0;
    nslow2 = 0;

    return Running;
}

template<typename FunctorType, typename Scalar>
typename HybridNonLinearSolver<FunctorType,Scalar>::Status
HybridNonLinearSolver<FunctorType,Scalar>::solveNumericalDiffOneStep(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const Parameters &parameters,
        const int mode
        )
{
    int i, j, l, iwa[1];
    jeval = true;
    int nsub = parameters.nb_of_subdiagonals;
    int nsup = parameters.nb_of_superdiagonals;
    if (nsub<0) nsub= n-1;
    if (nsup<0) nsup= n-1;

    /* calculate the jacobian matrix. */

    if (ei_fdjac1(functor, x, fvec, fjac, nsub, nsup, parameters.epsfcn) <0)
        return UserAksed;
    nfev += std::min(nsub+ nsup+ 1, n);

    /* compute the qr factorization of the jacobian. */

    ei_qrfac<Scalar>(n, n, fjac.data(), fjac.rows(), false, iwa, 1, wa1.data(), wa2.data());

    /* on the first iteration and if mode is 1, scale according */
    /* to the norms of the columns of the initial jacobian. */

    if (iter == 1) {
        if (mode != 2)
            for (j = 0; j < n; ++j) {
                diag[j] = wa2[j];
                if (wa2[j] == 0.)
                    diag[j] = 1.;
            }

        /* on the first iteration, calculate the norm of the scaled x */
        /* and initialize the step bound delta. */

        wa3 = diag.cwise() * x;
        xnorm = wa3.stableNorm();
        delta = parameters.factor * xnorm;
        if (delta == 0.)
            delta = parameters.factor;
    }

    /* form (q transpose)*fvec and store in qtf. */

    qtf = fvec;
    for (j = 0; j < n; ++j)
        if (fjac(j,j) != 0.) {
            sum = 0.;
            for (i = j; i < n; ++i)
                sum += fjac(i,j) * qtf[i];
            temp = -sum / fjac(j,j);
            for (i = j; i < n; ++i)
                qtf[i] += fjac(i,j) * temp;
        }

    /* copy the triangular factor of the qr factorization into r. */

    sing = false;
    for (j = 0; j < n; ++j) {
        l = j;
        if (j)
            for (i = 0; i < j; ++i) {
                R[l] = fjac(i,j);
                l = l + n - i -1;
            }
        R[l] = wa1[j];
        if (wa1[j] == 0.)
            sing = true;
    }

    /* accumulate the orthogonal factor in fjac. */

    ei_qform<Scalar>(n, n, fjac.data(), fjac.rows(), wa1.data());

    /* rescale if necessary. */

    /* Computing MAX */
    if (mode != 2)
        diag = diag.cwise().max(wa2);

    /* beginning of the inner loop. */

    while (true) {

        /* determine the direction p. */

        ei_dogleg<Scalar>(R, diag, qtf, delta, wa1);

        /* store the direction p and x + p. calculate the norm of p. */

        wa1 = -wa1;
        wa2 = x + wa1;
        wa3 = diag.cwise() * wa1;
        pnorm = wa3.stableNorm();

        /* on the first iteration, adjust the initial step bound. */

        if (iter == 1)
            delta = std::min(delta,pnorm);

        /* evaluate the function at x + p and calculate its norm. */

        if ( functor.f(wa2, wa4) < 0)
            return UserAksed;
        ++nfev;
        fnorm1 = wa4.stableNorm();

        /* compute the scaled actual reduction. */

        actred = -1.;
        if (fnorm1 < fnorm) /* Computing 2nd power */
            actred = 1. - ei_abs2(fnorm1 / fnorm);

        /* compute the scaled predicted reduction. */

        l = 0;
        for (i = 0; i < n; ++i) {
            sum = 0.;
            for (j = i; j < n; ++j) {
                sum += R[l] * wa1[j];
                ++l;
            }
            wa3[i] = qtf[i] + sum;
        }
        temp = wa3.stableNorm();
        prered = 0.;
        if (temp < fnorm) /* Computing 2nd power */
            prered = 1. - ei_abs2(temp / fnorm);

        /* compute the ratio of the actual to the predicted */
        /* reduction. */

        ratio = 0.;
        if (prered > 0.)
            ratio = actred / prered;

        /* update the step bound. */

        if (ratio < Scalar(.1)) {
            ncsuc = 0;
            ++ncfail;
            delta = Scalar(.5) * delta;
        } else {
            ncfail = 0;
            ++ncsuc;
            if (ratio >= Scalar(.5) || ncsuc > 1) /* Computing MAX */
                delta = std::max(delta, pnorm / Scalar(.5));
            if (ei_abs(ratio - 1.) <= Scalar(.1)) {
                delta = pnorm / Scalar(.5);
            }
        }

        /* test for successful iteration. */

        if (ratio >= Scalar(1e-4)) {
            /* successful iteration. update x, fvec, and their norms. */
            x = wa2;
            wa2 = diag.cwise() * x;
            fvec = wa4;
            xnorm = wa2.stableNorm();
            fnorm = fnorm1;
            ++iter;
        }

        /* determine the progress of the iteration. */

        ++nslow1;
        if (actred >= Scalar(.001))
            nslow1 = 0;
        if (jeval)
            ++nslow2;
        if (actred >= Scalar(.1))
            nslow2 = 0;

        /* test for convergence. */

        if (delta <= parameters.xtol * xnorm || fnorm == 0.)
            return RelativeErrorTooSmall;

        /* tests for termination and stringent tolerances. */

        if (nfev >= parameters.maxfev)
            return TooManyFunctionEvaluation;
        if (Scalar(.1) * std::max(Scalar(.1) * delta, pnorm) <= epsilon<Scalar>() * xnorm)
            return TolTooSmall;
        if (nslow2 == 5)
            return NotMakingProgressJacobian;
        if (nslow1 == 10)
            return NotMakingProgressIterations;

        /* criterion for recalculating jacobian approximation */
        /* by forward differences. */

        if (ncfail == 2)
            break; // leave inner loop and go for the next outer loop iteration

        /* calculate the rank one modification to the jacobian */
        /* and update qtf if necessary. */

        for (j = 0; j < n; ++j) {
            sum = wa4.dot(fjac.col(j));
            wa2[j] = (sum - wa3[j]) / pnorm;
            wa1[j] = diag[j] * (diag[j] * wa1[j] / pnorm);
            if (ratio >= Scalar(1e-4))
                qtf[j] = sum;
        }

        /* compute the qr factorization of the updated jacobian. */

        ei_r1updt<Scalar>(n, n, R.data(), R.size(), wa1.data(), wa2.data(), wa3.data(), &sing);
        ei_r1mpyq<Scalar>(n, n, fjac.data(), fjac.rows(), wa2.data(), wa3.data());
        ei_r1mpyq<Scalar>(1, n, qtf.data(), 1, wa2.data(), wa3.data());

        /* end of the inner loop. */

        jeval = false;
    }
    /* end of the outer loop. */

    return Running;
}

template<typename FunctorType, typename Scalar>
typename HybridNonLinearSolver<FunctorType,Scalar>::Status
HybridNonLinearSolver<FunctorType,Scalar>::solveNumericalDiff(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const Parameters &parameters,
        const int mode
        )
{
    Status status = solveNumericalDiffInit(x, parameters, mode);
    while (status==Running)
        status = solveNumericalDiffOneStep(x, parameters, mode);
    return status;
}

