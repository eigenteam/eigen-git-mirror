

template<typename FunctorType, typename Scalar>
class LevenbergMarquardtNumericalDiff 
{
public:
    LevenbergMarquardtNumericalDiff(const FunctorType &_functor)
        : functor(_functor) {}

    int minimize(
            Matrix< Scalar, Dynamic, 1 >  &x,
            Matrix< Scalar, Dynamic, 1 >  &fvec,
            const Scalar tol = ei_sqrt(epsilon<Scalar>())
            );

    int minimize(
            Matrix< Scalar, Dynamic, 1 >  &x,
            Matrix< Scalar, Dynamic, 1 >  &fvec,
            int &nfev,
            Matrix< Scalar, Dynamic, Dynamic > &fjac,
            VectorXi &ipvt,
            Matrix< Scalar, Dynamic, 1 >  &qtf,
            Matrix< Scalar, Dynamic, 1 >  &diag,
            int mode=1,
            Scalar factor = 100.,
            int maxfev = 400,
            Scalar ftol = ei_sqrt(epsilon<Scalar>()),
            Scalar xtol = ei_sqrt(epsilon<Scalar>()),
            Scalar gtol = Scalar(0.),
            Scalar epsfcn = Scalar(0.),
            int nprint=0
            );

private:
    const FunctorType &functor;
};


template<typename FunctorType, typename Scalar>
int LevenbergMarquardtNumericalDiff<FunctorType,Scalar>::minimize(
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        Scalar tol
        )
{
    const int n = x.size(), m=fvec.size();
    int info, nfev=0;
    Matrix< Scalar, Dynamic, Dynamic > fjac(m, n);
    Matrix< Scalar, Dynamic, 1> diag, qtf;
    VectorXi ipvt;

    /* check the input parameters for errors. */
    if (n <= 0 || m < n || tol < 0.) {
        printf("ei_lmder1 bad args : m,n,tol,...");
        return 0;
    }

    info = minimize(
        x, fvec,
        nfev,
        fjac, ipvt, qtf, diag,
        1,
        100.,
        (n+1)*200,
        tol, tol, Scalar(0.), Scalar(0.)
    );
    return (info==8)?4:info;
}

template<typename FunctorType, typename Scalar>
int LevenbergMarquardtNumericalDiff<FunctorType,Scalar>::minimize(
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        int &nfev,
        Matrix< Scalar, Dynamic, Dynamic > &fjac,
        VectorXi &ipvt,
        Matrix< Scalar, Dynamic, 1 >  &qtf,
        Matrix< Scalar, Dynamic, 1 >  &diag,
        int mode,
        Scalar factor,
        int maxfev,
        Scalar ftol,
        Scalar xtol,
        Scalar gtol,
        Scalar epsfcn,
        int nprint
        )
{
    const int m = fvec.size(), n = x.size();
    Matrix< Scalar, Dynamic, 1 > wa1(n), wa2(n), wa3(n), wa4(m);

    ipvt.resize(n);
    fjac.resize(m, n);
    diag.resize(n);
    qtf.resize(n);

    /* Local variables */
    int i, j, l;
    Scalar par, sum;
    int iter;
    Scalar temp, temp1, temp2;
    int iflag;
    Scalar delta;
    Scalar ratio;
    Scalar fnorm, gnorm;
    Scalar pnorm, xnorm, fnorm1, actred, dirder, prered;
    int info;

    /* Function Body */
    info = 0;
    iflag = 0;
    nfev = 0;

    /*     check the input parameters for errors. */

    if (n <= 0 || m < n || ftol < 0. || xtol < 0. || gtol < 0. || maxfev <= 0 || factor <= 0.)
        goto algo_end;
    if (mode == 2)
        for (j = 0; j < n; ++j)
            if (diag[j] <= 0.) goto algo_end;

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    iflag = functor.f(x, fvec);
    nfev = 1;
    if (iflag < 0)
        goto algo_end;
    fnorm = fvec.stableNorm();

    /*     initialize levenberg-marquardt parameter and iteration counter. */

    par = 0.;
    iter = 1;

    /*     beginning of the outer loop. */

    while (true) {

        /* calculate the jacobian matrix. */

        iflag = ei_fdjac2(functor, x, fvec, fjac, epsfcn);
        nfev += n;
        if (iflag < 0)
            break;

        /* if requested, call functor.f to enable printing of iterates. */

        if (nprint > 0) {
            iflag = 0;
            if ((iter - 1) % nprint == 0)
                iflag = functor.debug(x, fvec);
            if (iflag < 0)
                break;
        }

        /* compute the qr factorization of the jacobian. */

        ei_qrfac<Scalar>(m, n, fjac.data(), fjac.rows(), true, ipvt.data(), n, wa1.data(), wa2.data());
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
            delta = factor * xnorm;
            if (delta == 0.)
                delta = factor;
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

        if (gnorm <= gtol)
            info = 4;
        if (info != 0)
            break;

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

            iflag = functor.f(wa2, wa4);
            ++nfev;
            if (iflag < 0)
                goto algo_end;
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

            if (ei_abs(actred) <= ftol && prered <= ftol && Scalar(.5) * ratio <= 1.)
                info = 1;
            if (delta <= xtol * xnorm)
                info = 2;
            if (ei_abs(actred) <= ftol && prered <= ftol && Scalar(.5) * ratio <= 1. && info == 2)
                info = 3;
            if (info != 0)
                goto algo_end;

            /* tests for termination and stringent tolerances. */

            if (nfev >= maxfev)
                info = 5;
            if (ei_abs(actred) <= epsilon<Scalar>() && prered <= epsilon<Scalar>() && Scalar(.5) * ratio <= 1.)
                info = 6;
            if (delta <= epsilon<Scalar>() * xnorm)
                info = 7;
            if (gnorm <= epsilon<Scalar>())
                info = 8;
            if (info != 0)
                goto algo_end;
            /* end of the inner loop. repeat if iteration unsuccessful. */
        } while (ratio < Scalar(1e-4));
        /* end of the outer loop. */
    }
algo_end:

    /*     termination, either normal or user imposed. */
    if (iflag < 0)
        info = iflag;
    if (nprint > 0)
        iflag = functor.debug(x, fvec);
    return info;
}

