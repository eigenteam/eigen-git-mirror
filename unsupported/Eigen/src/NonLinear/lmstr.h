
template<typename Functor, typename Scalar>
int ei_lmstr(
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        int &nfev,
        int &njev,
        Matrix< Scalar, Dynamic, Dynamic > &fjac,
        VectorXi &ipvt,
        Matrix< Scalar, Dynamic, 1 >  &diag,
        int mode=1,
        Scalar factor = 100.,
        int maxfev = 400,
        Scalar ftol = ei_sqrt(epsilon<Scalar>()),
        Scalar xtol = ei_sqrt(epsilon<Scalar>()),
        Scalar gtol = Scalar(0.),
        int nprint=0
        )
{
    const int m = fvec.size(), n = x.size();
    Matrix< Scalar, Dynamic, 1 >
        qtf(n),
        wa1(n), wa2(n), wa3(n),
        wa4(m);
    int ldfjac = m;

    ipvt.resize(n);
    fjac.resize(ldfjac, n);
    diag.resize(n);

    /* Local variables */
    int i, j, l;
    Scalar par, sum;
    int sing;
    int iter;
    Scalar temp, temp1, temp2;
    int iflag;
    Scalar delta;
    Scalar ratio;
    Scalar fnorm, gnorm, pnorm, xnorm, fnorm1, actred, dirder, prered;
    int info;

    /* Function Body */
    info = 0;
    iflag = 0;
    nfev = 0;
    njev = 0;

    /*     check the input parameters for errors. */

    if (n <= 0 || m < n || ldfjac < n || ftol < 0. || xtol < 0. || 
            gtol < 0. || maxfev <= 0 || factor <= 0.) {
        goto L340;
    }
    if (mode != 2) {
        goto L20;
    }
    for (j = 0; j < n; ++j) {
        if (diag[j] <= 0.) {
            goto L340;
        }
        /* L10: */
    }
L20:

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    iflag = Functor::f(0, m, n, x.data(), fvec.data(), wa3.data(), 1);
    nfev = 1;
    if (iflag < 0) {
        goto L340;
    }
    fnorm = fvec.stableNorm();;

    /*     initialize levenberg-marquardt parameter and iteration counter. */

    par = 0.;
    iter = 1;

    /*     beginning of the outer loop. */

L30:

    /*        if requested, call Functor::f to enable printing of iterates. */

    if (nprint <= 0) {
        goto L40;
    }
    iflag = 0;
    if ((iter - 1) % nprint == 0) {
        iflag = Functor::f(0, m, n, x.data(), fvec.data(), wa3.data(), 0);
    }
    if (iflag < 0) {
        goto L340;
    }
L40:

    /*        compute the qr factorization of the jacobian matrix */
    /*        calculated one row at a time, while simultaneously */
    /*        forming (q transpose)*fvec and storing the first */
    /*        n components in qtf. */

    for (j = 0; j < n; ++j) {
        qtf[j] = 0.;
        for (i = 0; i < n; ++i) {
            fjac(i,j) = 0.;
            /* L50: */
        }
        /* L60: */
    }
    iflag = 2;
    for (i = 0; i < m; ++i) {
        if (Functor::f(0, m, n, x.data(), fvec.data(), wa3.data(), iflag) < 0) {
            goto L340;
        }
        temp = fvec[i];
        rwupdt(n, fjac.data(), ldfjac, wa3.data(), qtf.data(), &temp, wa1.data(), wa2.data());
        ++iflag;
        /* L70: */
    }
    ++njev;

    /*        if the jacobian is rank deficient, call qrfac to */
    /*        reorder its columns and update the components of qtf. */

    sing = FALSE_;
    for (j = 0; j < n; ++j) {
        if (fjac(j,j) == 0.) {
            sing = TRUE_;
        }
        ipvt[j] = j+1;
        wa2[j] = fjac.col(j).start(j).stableNorm();
//        wa2[j] = ei_enorm<Scalar>(j, &fjac[j * ldfjac + 1]);
//            sum += fjac[i + j * ldfjac] * (qtf[i] / fnorm);
        /* L80: */
    }
    if (! sing) {
        goto L130;
    }
    qrfac(n, n, fjac.data(), ldfjac, TRUE_, ipvt.data(), n, wa1.data(), wa2.data(), wa3.data());
    for (j = 0; j < n; ++j) {
        if (fjac(j,j) == 0.) {
            goto L110;
        }
        sum = 0.;
        for (i = j; i < n; ++i) {
            sum += fjac(i,j) * qtf[i];
            /* L90: */
        }
        temp = -sum / fjac(j,j);
        for (i = j; i < n; ++i) {
            qtf[i] += fjac(i,j) * temp;
            /* L100: */
        }
L110:
        fjac(j,j) = wa1[j];
        /* L120: */
    }
L130:

    /*        on the first iteration and if mode is 1, scale according */
    /*        to the norms of the columns of the initial jacobian. */

    if (iter != 1) {
        goto L170;
    }
    if (mode == 2) {
        goto L150;
    }
    for (j = 0; j < n; ++j) {
        diag[j] = wa2[j];
        if (wa2[j] == 0.) {
            diag[j] = 1.;
        }
        /* L140: */
    }
L150:

    /*        on the first iteration, calculate the norm of the scaled x */
    /*        and initialize the step bound delta. */

    for (j = 0; j < n; ++j) {
        wa3[j] = diag[j] * x[j];
        /* L160: */
    }
    xnorm = wa3.stableNorm();
    delta = factor * xnorm;
    if (delta == 0.) {
        delta = factor;
    }
L170:

    /*        compute the norm of the scaled gradient. */

    gnorm = 0.;
    if (fnorm == 0.) {
        goto L210;
    }
    for (j = 0; j < n; ++j) {
        l = ipvt[j]-1;
        if (wa2[l] == 0.) {
            goto L190;
        }
        sum = 0.;
        for (i = 0; i < j; ++i) {
            sum += fjac(i,j) * (qtf[i] / fnorm);
            /* L180: */
        }
        /* Computing MAX */
        gnorm = max(gnorm, ei_abs(sum/wa2[l]));
L190:
        /* L200: */
        ;
    }
L210:

    /*        test for convergence of the gradient norm. */

    if (gnorm <= gtol) {
        info = 4;
    }
    if (info != 0) {
        goto L340;
    }

    /*        rescale if necessary. */

    if (mode == 2) {
        goto L230;
    }
    for (j = 0; j < n; ++j) /* Computing MAX */
        diag[j] = max(diag[j], wa2[j]);
L230:

    /*        beginning of the inner loop. */

L240:

    /*           determine the levenberg-marquardt parameter. */

    lmpar(n, fjac.data(), ldfjac, ipvt.data(), diag.data(), qtf.data(), delta, &par,
            wa1.data(), wa2.data(), wa3.data(), wa4.data());

    /*           store the direction p and x + p. calculate the norm of p. */

    for (j = 0; j < n; ++j) {
        wa1[j] = -wa1[j];
        wa2[j] = x[j] + wa1[j];
        wa3[j] = diag[j] * wa1[j];
        /* L250: */
    }
    pnorm = wa3.stableNorm();

    /*           on the first iteration, adjust the initial step bound. */

    if (iter == 1) {
        delta = min(delta,pnorm);
    }

    /*           evaluate the function at x + p and calculate its norm. */

    iflag = Functor::f(0, m, n, wa2.data(), wa4.data(), wa3.data(), 1);
    ++nfev;
    if (iflag < 0) {
        goto L340;
    }
    fnorm1 = wa4.stableNorm();

    /*           compute the scaled actual reduction. */

    actred = -1.;
    if (p1 * fnorm1 < fnorm) /* Computing 2nd power */
        actred = 1. - ei_abs2(fnorm1 / fnorm);

    /*           compute the scaled predicted reduction and */
    /*           the scaled directional derivative. */

    for (j = 0; j < n; ++j) {
        wa3[j] = 0.;
        l = ipvt[j]-1;
        temp = wa1[l];
        for (i = 0; i <= j; ++i) {
            wa3[i] += fjac(i,j) * temp;
            /* L260: */
        }
        /* L270: */
    }
    temp1 = ei_abs2(wa3.stableNorm() / fnorm);
    temp2 = ei_abs2( ei_sqrt(par) * pnorm / fnorm);
    /* Computing 2nd power */
    prered = temp1 + temp2 / p5;
    dirder = -(temp1 + temp2);

    /*           compute the ratio of the actual to the predicted */
    /*           reduction. */

    ratio = 0.;
    if (prered != 0.) {
        ratio = actred / prered;
    }

    /*           update the step bound. */

    if (ratio > p25) {
        goto L280;
    }
    if (actred >= 0.) {
        temp = p5;
    }
    if (actred < 0.) {
        temp = p5 * dirder / (dirder + p5 * actred);
    }
    if (p1 * fnorm1 >= fnorm || temp < p1) {
        temp = p1;
    }
    /* Computing MIN */
    delta = temp * min(delta, pnorm / p1);

    par /= temp;
    goto L300;
L280:
    if (par != 0. && ratio < p75) {
        goto L290;
    }
    delta = pnorm / p5;
    par = p5 * par;
L290:
L300:

    /*           test for successful iteration. */

    if (ratio < p0001) {
        goto L330;
    }

    /*           successful iteration. update x, fvec, and their norms. */

    for (j = 0; j < n; ++j) {
        x[j] = wa2[j];
        wa2[j] = diag[j] * x[j];
        /* L310: */
    }
    for (i = 0; i < m; ++i) {
        fvec[i] = wa4[i];
        /* L320: */
    }
    xnorm = wa2.stableNorm();
    fnorm = fnorm1;
    ++iter;
L330:

    /*           tests for convergence. */

    if (ei_abs(actred) <= ftol && prered <= ftol && p5 * ratio <= 1.) {
        info = 1;
    }
    if (delta <= xtol * xnorm) {
        info = 2;
    }
    if (ei_abs(actred) <= ftol && prered <= ftol && p5 * ratio <= 1. && info 
            == 2) {
        info = 3;
    }
    if (info != 0) {
        goto L340;
    }

    /*           tests for termination and stringent tolerances. */

    if (nfev >= maxfev) {
        info = 5;
    }
    if (ei_abs(actred) <= epsilon<Scalar>() && prered <= epsilon<Scalar>() && p5 * ratio <= 1.) {
        info = 6;
    }
    if (delta <= epsilon<Scalar>() * xnorm) {
        info = 7;
    }
    if (gnorm <= epsilon<Scalar>()) {
        info = 8;
    }
    if (info != 0) {
        goto L340;
    }

    /*           end of the inner loop. repeat if iteration unsuccessful. */

    if (ratio < p0001) {
        goto L240;
    }

    /*        end of the outer loop. */

    goto L30;
L340:

    /*     termination, either normal or user imposed. */

    if (iflag < 0) {
        info = iflag;
    }
    iflag = 0;
    if (nprint > 0) {
        iflag = Functor::f(0, m, n, x.data(), fvec.data(), wa3.data(), 0);
    }
    return info;

    /*     last card of subroutine lmstr. */

} /* lmstr_ */

