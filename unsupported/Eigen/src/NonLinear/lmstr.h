
template<typename Scalar>
int lmstr_template(minpack_funcderstr_mn fcn, void *p, int m, int n, Scalar *x, 
        Scalar *fvec, Scalar *fjac, int ldfjac, Scalar ftol,
        Scalar xtol, Scalar gtol, int maxfev, Scalar *
        diag, int mode, Scalar factor, int nprint,
        int &nfev, int &njev, int *ipvt, Scalar *qtf, 
        Scalar *wa1, Scalar *wa2, Scalar *wa3, Scalar *wa4)
{
    /* Initialized data */

    /* System generated locals */
    int fjac_offset;

    /* Local variables */
    int i__, j, l;
    Scalar par, sum;
    int sing;
    int iter;
    Scalar temp, temp1, temp2;
    int iflag;
    Scalar delta;
    Scalar ratio;
    Scalar fnorm, gnorm, pnorm, xnorm, fnorm1, actred, dirder, prered;
    int info;

    /* Parameter adjustments */
    --wa4;
    --fvec;
    --wa3;
    --wa2;
    --wa1;
    --qtf;
    --ipvt;
    --diag;
    --x;
    fjac_offset = 1 + ldfjac;
    fjac -= fjac_offset;

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
    for (j = 1; j <= n; ++j) {
        if (diag[j] <= 0.) {
            goto L340;
        }
        /* L10: */
    }
L20:

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    iflag = (*fcn)(p, m, n, &x[1], &fvec[1], &wa3[1], 1);
    nfev = 1;
    if (iflag < 0) {
        goto L340;
    }
    fnorm = ei_enorm<Scalar>(m, &fvec[1]);

    /*     initialize levenberg-marquardt parameter and iteration counter. */

    par = 0.;
    iter = 1;

    /*     beginning of the outer loop. */

L30:

    /*        if requested, call fcn to enable printing of iterates. */

    if (nprint <= 0) {
        goto L40;
    }
    iflag = 0;
    if ((iter - 1) % nprint == 0) {
        iflag = (*fcn)(p, m, n, &x[1], &fvec[1], &wa3[1], 0);
    }
    if (iflag < 0) {
        goto L340;
    }
L40:

    /*        compute the qr factorization of the jacobian matrix */
    /*        calculated one row at a time, while simultaneously */
    /*        forming (q transpose)*fvec and storing the first */
    /*        n components in qtf. */

    for (j = 1; j <= n; ++j) {
        qtf[j] = 0.;
        for (i__ = 1; i__ <= n; ++i__) {
            fjac[i__ + j * ldfjac] = 0.;
            /* L50: */
        }
        /* L60: */
    }
    iflag = 2;
    for (i__ = 1; i__ <= m; ++i__) {
        if ((*fcn)(p, m, n, &x[1], &fvec[1], &wa3[1], iflag) < 0) {
            goto L340;
        }
        temp = fvec[i__];
        rwupdt(n, &fjac[fjac_offset], ldfjac, &wa3[1], &qtf[1], &temp, &wa1[
                1], &wa2[1]);
        ++iflag;
        /* L70: */
    }
    ++njev;

    /*        if the jacobian is rank deficient, call qrfac to */
    /*        reorder its columns and update the components of qtf. */

    sing = FALSE_;
    for (j = 1; j <= n; ++j) {
        if (fjac[j + j * ldfjac] == 0.) {
            sing = TRUE_;
        }
        ipvt[j] = j;
        wa2[j] = ei_enorm<Scalar>(j, &fjac[j * ldfjac + 1]);
        /* L80: */
    }
    if (! sing) {
        goto L130;
    }
    qrfac(n, n, &fjac[fjac_offset], ldfjac, TRUE_, &ipvt[1], n, &wa1[1], &
            wa2[1], &wa3[1]);
    for (j = 1; j <= n; ++j) {
        if (fjac[j + j * ldfjac] == 0.) {
            goto L110;
        }
        sum = 0.;
        for (i__ = j; i__ <= n; ++i__) {
            sum += fjac[i__ + j * ldfjac] * qtf[i__];
            /* L90: */
        }
        temp = -sum / fjac[j + j * ldfjac];
        for (i__ = j; i__ <= n; ++i__) {
            qtf[i__] += fjac[i__ + j * ldfjac] * temp;
            /* L100: */
        }
L110:
        fjac[j + j * ldfjac] = wa1[j];
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
    for (j = 1; j <= n; ++j) {
        diag[j] = wa2[j];
        if (wa2[j] == 0.) {
            diag[j] = 1.;
        }
        /* L140: */
    }
L150:

    /*        on the first iteration, calculate the norm of the scaled x */
    /*        and initialize the step bound delta. */

    for (j = 1; j <= n; ++j) {
        wa3[j] = diag[j] * x[j];
        /* L160: */
    }
    xnorm = ei_enorm<Scalar>(n, &wa3[1]);
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
    for (j = 1; j <= n; ++j) {
        l = ipvt[j];
        if (wa2[l] == 0.) {
            goto L190;
        }
        sum = 0.;
        for (i__ = 1; i__ <= j; ++i__) {
            sum += fjac[i__ + j * ldfjac] * (qtf[i__] / fnorm);
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
    for (j = 1; j <= n; ++j) /* Computing MAX */
        diag[j] = max(diag[j], wa2[j]);
L230:

    /*        beginning of the inner loop. */

L240:

    /*           determine the levenberg-marquardt parameter. */

    lmpar(n, &fjac[fjac_offset], ldfjac, &ipvt[1], &diag[1], &qtf[1], delta,
            &par, &wa1[1], &wa2[1], &wa3[1], &wa4[1]);

    /*           store the direction p and x + p. calculate the norm of p. */

    for (j = 1; j <= n; ++j) {
        wa1[j] = -wa1[j];
        wa2[j] = x[j] + wa1[j];
        wa3[j] = diag[j] * wa1[j];
        /* L250: */
    }
    pnorm = ei_enorm<Scalar>(n, &wa3[1]);

    /*           on the first iteration, adjust the initial step bound. */

    if (iter == 1) {
        delta = min(delta,pnorm);
    }

    /*           evaluate the function at x + p and calculate its norm. */

    iflag = (*fcn)(p, m, n, &wa2[1], &wa4[1], &wa3[1], 1);
    ++nfev;
    if (iflag < 0) {
        goto L340;
    }
    fnorm1 = ei_enorm<Scalar>(m, &wa4[1]);

    /*           compute the scaled actual reduction. */

    actred = -1.;
    if (p1 * fnorm1 < fnorm) /* Computing 2nd power */
        actred = 1. - ei_abs2(fnorm1 / fnorm);

    /*           compute the scaled predicted reduction and */
    /*           the scaled directional derivative. */

    for (j = 1; j <= n; ++j) {
        wa3[j] = 0.;
        l = ipvt[j];
        temp = wa1[l];
        for (i__ = 1; i__ <= j; ++i__) {
            wa3[i__] += fjac[i__ + j * ldfjac] * temp;
            /* L260: */
        }
        /* L270: */
    }
    temp1 = ei_enorm<Scalar>(n, &wa3[1]) / fnorm;
    temp2 = sqrt(par) * pnorm / fnorm;
    /* Computing 2nd power */
    prered = temp1 * temp1 + temp2 * temp2 / p5;
    dirder = -(temp1 * temp1 + temp2 * temp2);

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

    for (j = 1; j <= n; ++j) {
        x[j] = wa2[j];
        wa2[j] = diag[j] * x[j];
        /* L310: */
    }
    for (i__ = 1; i__ <= m; ++i__) {
        fvec[i__] = wa4[i__];
        /* L320: */
    }
    xnorm = ei_enorm<Scalar>(n, &wa2[1]);
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
        iflag = (*fcn)(p, m, n, &x[1], &fvec[1], &wa3[1], 0);
    }
    return info;

    /*     last card of subroutine lmstr. */

} /* lmstr_ */

