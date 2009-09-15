
template<typename FunctorType, typename Scalar=double>
class LevenbergMarquardt 
{
public:
    LevenbergMarquardt(FunctorType &_functor)
        : functor(_functor) { nfev = njev = iter = 0;  fnorm=gnorm = 0.; }

    enum Status {
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

    struct Parameters {
        Parameters()
            : factor(Scalar(100.))
            , maxfev(400)
            , ftol(ei_sqrt(epsilon<Scalar>()))
            , xtol(ei_sqrt(epsilon<Scalar>()))
            , gtol(Scalar(0.))
            , epsfcn(Scalar(0.)) {}
        Scalar factor;
        int maxfev;   // maximum number of function evaluation
        Scalar ftol;
        Scalar xtol;
        Scalar gtol;
        Scalar epsfcn;
    };

    Status lmder1(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const Scalar tol = ei_sqrt(epsilon<Scalar>())
            );

    Status minimize(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const int mode=1
            );
    Status minimizeInit(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const int mode=1
            );
    Status minimizeOneStep(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const int mode=1
            );

    Status lmdif1(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const Scalar tol = ei_sqrt(epsilon<Scalar>())
            );

    Status minimizeNumericalDiff(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const int mode=1
            );
    Status minimizeNumericalDiffInit(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const int mode=1
            );
    Status minimizeNumericalDiffOneStep(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const int mode=1
            );

    Status lmstr1(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const Scalar tol = ei_sqrt(epsilon<Scalar>())
            );

    Status minimizeOptimumStorage(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const int mode=1
            );
    Status minimizeOptimumStorageInit(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const int mode=1
            );
    Status minimizeOptimumStorageOneStep(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const int mode=1
            );

    void resetParameters(void) { parameters = Parameters(); }
    Parameters parameters;
    Matrix< Scalar, Dynamic, 1 >  fvec;
    Matrix< Scalar, Dynamic, Dynamic > fjac;
    VectorXi ipvt;
    Matrix< Scalar, Dynamic, 1 >  qtf;
    Matrix< Scalar, Dynamic, 1 >  diag;
    int nfev;
    int njev;
    int iter;
    Scalar fnorm, gnorm;
private:
    FunctorType &functor;
    int n;
    int m;
    Matrix< Scalar, Dynamic, 1 > wa1, wa2, wa3, wa4;

    Scalar par, sum;
    Scalar temp, temp1, temp2;
    Scalar delta;
    Scalar ratio;
    Scalar pnorm, xnorm, fnorm1, actred, dirder, prered;
};

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::lmder1(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const Scalar tol
        )
{
    n = x.size();
    m = functor.nbOfFunctions();

    /* check the input parameters for errors. */
    if (n <= 0 || m < n || tol < 0.)
        return ImproperInputParameters;

    resetParameters();
    parameters.ftol = tol;
    parameters.xtol = tol;
    parameters.maxfev = 100*(n+1);

    return minimize(x);
}


template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimize(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const int mode
        )
{
    Status status = minimizeInit(x, mode);
    while (status==Running)
        status = minimizeOneStep(x, mode);
    return status;
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeInit(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const int mode
        )
{
    n = x.size();
    m = functor.nbOfFunctions();

    wa1.resize(n); wa2.resize(n); wa3.resize(n);
    wa4.resize(m);
    fvec.resize(m);
    ipvt.resize(n);
    fjac.resize(m, n);
    if (mode != 2)
        diag.resize(n);
    assert( (mode!=2 || diag.size()==n) || "When using mode==2, the caller must provide a valid 'diag'");
    qtf.resize(n);

    /* Function Body */
    nfev = 0;
    njev = 0;

    /*     check the input parameters for errors. */

    if (n <= 0 || m < n || parameters.ftol < 0. || parameters.xtol < 0. || parameters.gtol < 0. || parameters.maxfev <= 0 || parameters.factor <= 0.)
        return ImproperInputParameters;

    if (mode == 2)
        for (int j = 0; j < n; ++j)
            if (diag[j] <= 0.)
                return ImproperInputParameters;

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    nfev = 1;
    if ( functor.f(x, fvec) < 0)
        return UserAsked;
    fnorm = fvec.stableNorm();

    /*     initialize levenberg-marquardt parameter and iteration counter. */

    par = 0.;
    iter = 1;

    return Running;
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeOneStep(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const int mode
        )
{
    int i, j, l;

    /* calculate the jacobian matrix. */

    if (functor.df(x, fjac) < 0)
        return UserAsked;
    ++njev;

    /* compute the qr factorization of the jacobian. */

    ei_qrfac<Scalar>(m, n, fjac.data(), fjac.rows(), true, ipvt.data(), wa1.data(), wa2.data());
    ipvt.cwise()-=1; // qrfac() creates ipvt with fortran convetion (1->n), convert it to c (0->n-1)

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

    /* form (q transpose)*fvec and store the first n components in */
    /* qtf. */

    wa4 = fvec;
    for (j = 0; j < n; ++j) {
        if (fjac(j,j) != 0.) {
            sum = 0.;
            for (i = j; i < m; ++i)
                sum += fjac(i,j) * wa4[i];
            temp = -sum / fjac(j,j);
            for (i = j; i < m; ++i)
                wa4[i] += fjac(i,j) * temp;
        }
        fjac(j,j) = wa1[j];
        qtf[j] = wa4[j];
    }

    /* compute the norm of the scaled gradient. */

    gnorm = 0.;
    if (fnorm != 0.)
        for (j = 0; j < n; ++j) {
            l = ipvt[j];
            if (wa2[l] != 0.) {
                sum = 0.;
                for (i = 0; i <= j; ++i)
                    sum += fjac(i,j) * (qtf[i] / fnorm);
                /* Computing MAX */
                gnorm = std::max(gnorm, ei_abs(sum / wa2[l]));
            }
        }

    /* test for convergence of the gradient norm. */

    if (gnorm <= parameters.gtol)
        return CosinusTooSmall;

    /* rescale if necessary. */

    if (mode != 2) /* Computing MAX */
        diag = diag.cwise().max(wa2);

    /* beginning of the inner loop. */
    do {

        /* determine the levenberg-marquardt parameter. */

        ei_lmpar<Scalar>(fjac, ipvt, diag, qtf, delta, par, wa1, wa2);

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
            return UserAsked;
        ++nfev;
        fnorm1 = wa4.stableNorm();

        /* compute the scaled actual reduction. */

        actred = -1.;
        if (Scalar(.1) * fnorm1 < fnorm) /* Computing 2nd power */
            actred = 1. - ei_abs2(fnorm1 / fnorm);

        /* compute the scaled predicted reduction and */
        /* the scaled directional derivative. */

        wa3.fill(0.);
        for (j = 0; j < n; ++j) {
            l = ipvt[j];
            temp = wa1[l];
            for (i = 0; i <= j; ++i)
                wa3[i] += fjac(i,j) * temp;
        }
        temp1 = ei_abs2(wa3.stableNorm() / fnorm);
        temp2 = ei_abs2(ei_sqrt(par) * pnorm / fnorm);
        /* Computing 2nd power */
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
                temp = Scalar(.5);
            if (actred < 0.)
                temp = Scalar(.5) * dirder / (dirder + Scalar(.5) * actred);
            if (Scalar(.1) * fnorm1 >= fnorm || temp < Scalar(.1))
                temp = Scalar(.1);
            /* Computing MIN */
            delta = temp * std::min(delta, pnorm / Scalar(.1));
            par /= temp;
        } else if (!(par != 0. && ratio < Scalar(.75))) {
            delta = pnorm / Scalar(.5);
            par = Scalar(.5) * par;
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

        /* tests for convergence. */

        if (ei_abs(actred) <= parameters.ftol && prered <= parameters.ftol && Scalar(.5) * ratio <= 1. && delta <= parameters.xtol * xnorm)
            return RelativeErrorAndReductionTooSmall;
        if (ei_abs(actred) <= parameters.ftol && prered <= parameters.ftol && Scalar(.5) * ratio <= 1.)
            return RelativeReductionTooSmall;
        if (delta <= parameters.xtol * xnorm)
            return RelativeErrorTooSmall;

        /* tests for termination and stringent tolerances. */

        if (nfev >= parameters.maxfev)
            return TooManyFunctionEvaluation;
        if (ei_abs(actred) <= epsilon<Scalar>() && prered <= epsilon<Scalar>() && Scalar(.5) * ratio <= 1.)
            return FtolTooSmall;
        if (delta <= epsilon<Scalar>() * xnorm)
            return XtolTooSmall;
        if (gnorm <= epsilon<Scalar>())
            return GtolTooSmall;
        /* end of the inner loop. repeat if iteration unsuccessful. */
    } while (ratio < Scalar(1e-4));
    /* end of the outer loop. */
    return Running;
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::lmdif1(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const Scalar tol
        )
{
    n = x.size();
    m = functor.nbOfFunctions();

    /* check the input parameters for errors. */
    if (n <= 0 || m < n || tol < 0.)
        return ImproperInputParameters;

    resetParameters();
    parameters.ftol = tol;
    parameters.xtol = tol;
    parameters.maxfev = 200*(n+1);

    return minimizeNumericalDiff(x);
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeNumericalDiffInit(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const int mode
        )
{
    n = x.size();
    m = functor.nbOfFunctions();

    wa1.resize(n); wa2.resize(n); wa3.resize(n);
    wa4.resize(m);
    fvec.resize(m);
    ipvt.resize(n);
    fjac.resize(m, n);
    if (mode != 2 )
        diag.resize(n);
    assert( (mode!=2 || diag.size()==n) || "When using mode==2, the caller must provide a valid 'diag'");
    qtf.resize(n);

    /* Function Body */
    nfev = 0;
    njev = 0;

    /*     check the input parameters for errors. */

    if (n <= 0 || m < n || parameters.ftol < 0. || parameters.xtol < 0. || parameters.gtol < 0. || parameters.maxfev <= 0 || parameters.factor <= 0.)
        return ImproperInputParameters;
    if (mode == 2)
        for (int j = 0; j < n; ++j)
            if (diag[j] <= 0.)
                return ImproperInputParameters;

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    nfev = 1;
    if ( functor.f(x, fvec) < 0)
        return UserAsked;
    fnorm = fvec.stableNorm();

    /*     initialize levenberg-marquardt parameter and iteration counter. */

    par = 0.;
    iter = 1;

    return Running;
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeNumericalDiffOneStep(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const int mode
        )
{
    int i, j, l;

    /* calculate the jacobian matrix. */

    if ( ei_fdjac2(functor, x, fvec, fjac, parameters.epsfcn) < 0)
        return UserAsked;
    nfev += n;

    /* compute the qr factorization of the jacobian. */

    ei_qrfac<Scalar>(m, n, fjac.data(), fjac.rows(), true, ipvt.data(), wa1.data(), wa2.data());
    ipvt.cwise()-=1; // qrfac() creates ipvt with fortran convetion (1->n), convert it to c (0->n-1)

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

    /* form (q transpose)*fvec and store the first n components in */
    /* qtf. */

    wa4 = fvec;
    for (j = 0; j < n; ++j) {
        if (fjac(j,j) != 0.) {
            sum = 0.;
            for (i = j; i < m; ++i)
                sum += fjac(i,j) * wa4[i];
            temp = -sum / fjac(j,j);
            for (i = j; i < m; ++i)
                wa4[i] += fjac(i,j) * temp;
        }
        fjac(j,j) = wa1[j];
        qtf[j] = wa4[j];
    }

    /* compute the norm of the scaled gradient. */

    gnorm = 0.;
    if (fnorm != 0.)
        for (j = 0; j < n; ++j) {
            l = ipvt[j];
            if (wa2[l] != 0.) {
                sum = 0.;
                for (i = 0; i <= j; ++i)
                    sum += fjac(i,j) * (qtf[i] / fnorm);
                /* Computing MAX */
                gnorm = std::max(gnorm, ei_abs(sum / wa2[l]));
            }
        }

    /* test for convergence of the gradient norm. */

    if (gnorm <= parameters.gtol)
        return CosinusTooSmall;

    /* rescale if necessary. */

    if (mode != 2) /* Computing MAX */
        diag = diag.cwise().max(wa2);

    /* beginning of the inner loop. */
    do {

        /* determine the levenberg-marquardt parameter. */

        ei_lmpar<Scalar>(fjac, ipvt, diag, qtf, delta, par, wa1, wa2);

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
            return UserAsked;
        ++nfev;
        fnorm1 = wa4.stableNorm();

        /* compute the scaled actual reduction. */

        actred = -1.;
        if (Scalar(.1) * fnorm1 < fnorm) /* Computing 2nd power */
            actred = 1. - ei_abs2(fnorm1 / fnorm);

        /* compute the scaled predicted reduction and */
        /* the scaled directional derivative. */

        wa3.fill(0.);
        for (j = 0; j < n; ++j) {
            l = ipvt[j];
            temp = wa1[l];
            for (i = 0; i <= j; ++i)
                wa3[i] += fjac(i,j) * temp;
        }
        temp1 = ei_abs2(wa3.stableNorm() / fnorm);
        temp2 = ei_abs2(ei_sqrt(par) * pnorm / fnorm);
        /* Computing 2nd power */
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
                temp = Scalar(.5);
            if (actred < 0.)
                temp = Scalar(.5) * dirder / (dirder + Scalar(.5) * actred);
            if (Scalar(.1) * fnorm1 >= fnorm || temp < Scalar(.1))
                temp = Scalar(.1);
            /* Computing MIN */
            delta = temp * std::min(delta, pnorm / Scalar(.1));
            par /= temp;
        } else if (!(par != 0. && ratio < Scalar(.75))) {
            delta = pnorm / Scalar(.5);
            par = Scalar(.5) * par;
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

        /* tests for convergence. */

        if (ei_abs(actred) <= parameters.ftol && prered <= parameters.ftol && Scalar(.5) * ratio <= 1. && delta <= parameters.xtol * xnorm)
            return RelativeErrorAndReductionTooSmall;
        if (ei_abs(actred) <= parameters.ftol && prered <= parameters.ftol && Scalar(.5) * ratio <= 1.)
            return RelativeReductionTooSmall;
        if (delta <= parameters.xtol * xnorm)
            return RelativeErrorTooSmall;

        /* tests for termination and stringent tolerances. */

        if (nfev >= parameters.maxfev)
            return TooManyFunctionEvaluation;
        if (ei_abs(actred) <= epsilon<Scalar>() && prered <= epsilon<Scalar>() && Scalar(.5) * ratio <= 1.)
            return FtolTooSmall;
        if (delta <= epsilon<Scalar>() * xnorm)
            return XtolTooSmall;
        if (gnorm <= epsilon<Scalar>())
            return GtolTooSmall;
        /* end of the inner loop. repeat if iteration unsuccessful. */
    } while (ratio < Scalar(1e-4));
    /* end of the outer loop. */
    return Running;
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeNumericalDiff(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const int mode
        )
{
    Status status = minimizeNumericalDiffInit(x, mode);
    while (status==Running)
        status = minimizeNumericalDiffOneStep(x, mode);
    return status;
}


template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::lmstr1(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const Scalar tol
        )
{
    n = x.size();
    m = functor.nbOfFunctions();
    Matrix< Scalar, Dynamic, Dynamic > fjac(m, n);
    VectorXi ipvt;

    /* check the input parameters for errors. */
    if (n <= 0 || m < n || tol < 0.)
        return ImproperInputParameters;

    resetParameters();
    parameters.ftol = tol;
    parameters.xtol = tol;
    parameters.maxfev = 100*(n+1);

    return minimizeOptimumStorage(x);
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeOptimumStorageInit(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const int mode
        )
{
    n = x.size();
    m = functor.nbOfFunctions();

    wa1.resize(n); wa2.resize(n); wa3.resize(n);
    wa4.resize(m);
    fvec.resize(m);
    ipvt.resize(n);
    fjac.resize(m, n);
    if (mode != 2)
        diag.resize(n);
    assert( (mode!=2 || diag.size()==n) || "When using mode==2, the caller must provide a valid 'diag'");
    qtf.resize(n);

    /* Function Body */
    nfev = 0;
    njev = 0;

    /*     check the input parameters for errors. */

    if (n <= 0 || m < n || parameters.ftol < 0. || parameters.xtol < 0. || parameters.gtol < 0. || parameters.maxfev <= 0 || parameters.factor <= 0.)
        return ImproperInputParameters;

    if (mode == 2)
        for (int j = 0; j < n; ++j)
            if (diag[j] <= 0.)
                return ImproperInputParameters;

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    nfev = 1;
    if ( functor.f(x, fvec) < 0)
        return UserAsked;
    fnorm = fvec.stableNorm();

    /*     initialize levenberg-marquardt parameter and iteration counter. */

    par = 0.;
    iter = 1;

    return Running;
}


template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeOptimumStorageOneStep(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const int mode
        )
{
    int i, j, l;
    bool sing;

    /* compute the qr factorization of the jacobian matrix */
    /* calculated one row at a time, while simultaneously */
    /* forming (q transpose)*fvec and storing the first */
    /* n components in qtf. */

    qtf.fill(0.);
    fjac.fill(0.);
    int rownb = 2;
    for (i = 0; i < m; ++i) {
        if (functor.df(x, wa3, rownb) < 0) return UserAsked;
        temp = fvec[i];
        ei_rwupdt<Scalar>(n, fjac.data(), fjac.rows(), wa3.data(), qtf.data(), &temp, wa1.data(), wa2.data());
        ++rownb;
    }
    ++njev;

    /* if the jacobian is rank deficient, call qrfac to */
    /* reorder its columns and update the components of qtf. */

    sing = false;
    for (j = 0; j < n; ++j) {
        if (fjac(j,j) == 0.) {
            sing = true;
        }
        ipvt[j] = j;
        wa2[j] = fjac.col(j).start(j).stableNorm();
    }
    if (sing) {
        ipvt.cwise()+=1;
        ei_qrfac<Scalar>(n, n, fjac.data(), fjac.rows(), true, ipvt.data(), wa1.data(), wa2.data());
        ipvt.cwise()-=1; // qrfac() creates ipvt with fortran convetion (1->n), convert it to c (0->n-1)
        for (j = 0; j < n; ++j) {
            if (fjac(j,j) != 0.) {
                sum = 0.;
                for (i = j; i < n; ++i)
                    sum += fjac(i,j) * qtf[i];
                temp = -sum / fjac(j,j);
                for (i = j; i < n; ++i)
                    qtf[i] += fjac(i,j) * temp;
            }
            fjac(j,j) = wa1[j];
        }
    }

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

    /* compute the norm of the scaled gradient. */

    gnorm = 0.;
    if (fnorm != 0.)
        for (j = 0; j < n; ++j) {
            l = ipvt[j];
            if (wa2[l] != 0.) {
                sum = 0.;
                for (i = 0; i <= j; ++i)
                    sum += fjac(i,j) * (qtf[i] / fnorm);
                /* Computing MAX */
                gnorm = std::max(gnorm, ei_abs(sum / wa2[l]));
            }
        }

    /* test for convergence of the gradient norm. */

    if (gnorm <= parameters.gtol)
        return CosinusTooSmall;

    /* rescale if necessary. */

    if (mode != 2) /* Computing MAX */
        diag = diag.cwise().max(wa2);

    /* beginning of the inner loop. */
    do {

        /* determine the levenberg-marquardt parameter. */

        ei_lmpar<Scalar>(fjac, ipvt, diag, qtf, delta, par, wa1, wa2);

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
            return UserAsked;
        ++nfev;
        fnorm1 = wa4.stableNorm();

        /* compute the scaled actual reduction. */

        actred = -1.;
        if (Scalar(.1) * fnorm1 < fnorm) /* Computing 2nd power */
            actred = 1. - ei_abs2(fnorm1 / fnorm);

        /* compute the scaled predicted reduction and */
        /* the scaled directional derivative. */

        wa3.fill(0.);
        for (j = 0; j < n; ++j) {
            l = ipvt[j];
            temp = wa1[l];
            for (i = 0; i <= j; ++i)
                wa3[i] += fjac(i,j) * temp;
        }
        temp1 = ei_abs2(wa3.stableNorm() / fnorm);
        temp2 = ei_abs2(ei_sqrt(par) * pnorm / fnorm);
        /* Computing 2nd power */
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
                temp = Scalar(.5);
            if (actred < 0.)
                temp = Scalar(.5) * dirder / (dirder + Scalar(.5) * actred);
            if (Scalar(.1) * fnorm1 >= fnorm || temp < Scalar(.1))
                temp = Scalar(.1);
            /* Computing MIN */
            delta = temp * std::min(delta, pnorm / Scalar(.1));
            par /= temp;
        } else if (!(par != 0. && ratio < Scalar(.75))) {
            delta = pnorm / Scalar(.5);
            par = Scalar(.5) * par;
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

        /* tests for convergence. */

        if (ei_abs(actred) <= parameters.ftol && prered <= parameters.ftol && Scalar(.5) * ratio <= 1. && delta <= parameters.xtol * xnorm)
            return RelativeErrorAndReductionTooSmall;
        if (ei_abs(actred) <= parameters.ftol && prered <= parameters.ftol && Scalar(.5) * ratio <= 1.)
            return RelativeReductionTooSmall;
        if (delta <= parameters.xtol * xnorm)
            return RelativeErrorTooSmall;

        /* tests for termination and stringent tolerances. */

        if (nfev >= parameters.maxfev)
            return TooManyFunctionEvaluation;
        if (ei_abs(actred) <= epsilon<Scalar>() && prered <= epsilon<Scalar>() && Scalar(.5) * ratio <= 1.)
            return FtolTooSmall;
        if (delta <= epsilon<Scalar>() * xnorm)
            return XtolTooSmall;
        if (gnorm <= epsilon<Scalar>())
            return GtolTooSmall;
        /* end of the inner loop. repeat if iteration unsuccessful. */
    } while (ratio < Scalar(1e-4));
    /* end of the outer loop. */
    return Running;
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeOptimumStorage(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const int mode
        )
{
    Status status = minimizeOptimumStorageInit(x, mode);
    while (status==Running)
        status = minimizeOptimumStorageOneStep(x, mode);
    return status;
}


