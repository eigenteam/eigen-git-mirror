
template<typename Scalar>
int hybrj_template(minpack_funcder_nn fcn, void *p, int n, Scalar *x, Scalar *
        fvec, Scalar *fjac, int ldfjac, Scalar xtol, int
        maxfev, Scalar *diag, int mode, Scalar factor, int
        nprint, int &nfev, int &njev, Scalar *r__, 
        int lr, Scalar *qtf, Scalar *wa1, Scalar *wa2, 
        Scalar *wa3, Scalar *wa4)
{
    /* Initialized data */

    /* System generated locals */
    int fjac_offset;

    /* Local variables */
    int i, j, l, jm1, iwa[1];
    Scalar sum;
    int sing;
    int iter;
    Scalar temp;
    int iflag;
    Scalar delta;
    int jeval;
    int ncsuc;
    Scalar ratio;
    Scalar fnorm;
    Scalar pnorm, xnorm, fnorm1;
    int nslow1, nslow2;
    int ncfail;
    Scalar actred, prered;
    int info;

    /* Parameter adjustments */
    --wa4;
    --wa3;
    --wa2;
    --wa1;
    --qtf;
    --diag;
    --fvec;
    --x;
    fjac_offset = 1 + ldfjac;
    fjac -= fjac_offset;
    --r__;

    /* Function Body */

    info = 0;
    iflag = 0;
    nfev = 0;
    njev = 0;

    /*     check the input parameters for errors. */

    if (n <= 0 || ldfjac < n || xtol < 0. || maxfev <= 0 || factor <= 
            0. || lr < n * (n + 1) / 2) {
        goto L300;
    }
    if (mode != 2) {
        goto L20;
    }
    for (j = 1; j <= n; ++j) {
        if (diag[j] <= 0.) {
            goto L300;
        }
        /* L10: */
    }
L20:

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    iflag = (*fcn)(p, n, &x[1], &fvec[1], &fjac[fjac_offset], ldfjac, 1);
    nfev = 1;
    if (iflag < 0) {
        goto L300;
    }
    fnorm = ei_enorm<Scalar>(n, &fvec[1]);

    /*     initialize iteration counter and monitors. */

    iter = 1;
    ncsuc = 0;
    ncfail = 0;
    nslow1 = 0;
    nslow2 = 0;

    /*     beginning of the outer loop. */

L30:
    jeval = TRUE_;

    /*        calculate the jacobian matrix. */

    iflag = (*fcn)(p, n, &x[1], &fvec[1], &fjac[fjac_offset], ldfjac, 2);
    ++njev;
    if (iflag < 0) {
        goto L300;
    }

    /*        compute the qr factorization of the jacobian. */

    qrfac(n, n, &fjac[fjac_offset], ldfjac, FALSE_, iwa, 1, &wa1[1], &
            wa2[1], &wa3[1]);

    /*        on the first iteration and if mode is 1, scale according */
    /*        to the norms of the columns of the initial jacobian. */

    if (iter != 1) {
        goto L70;
    }
    if (mode == 2) {
        goto L50;
    }
    for (j = 1; j <= n; ++j) {
        diag[j] = wa2[j];
        if (wa2[j] == 0.) {
            diag[j] = 1.;
        }
        /* L40: */
    }
L50:

    /*        on the first iteration, calculate the norm of the scaled x */
    /*        and initialize the step bound delta. */

    for (j = 1; j <= n; ++j) {
        wa3[j] = diag[j] * x[j];
        /* L60: */
    }
    xnorm = ei_enorm<Scalar>(n, &wa3[1]);
    delta = factor * xnorm;
    if (delta == 0.) {
        delta = factor;
    }
L70:

    /*        form (q transpose)*fvec and store in qtf. */

    for (i = 1; i <= n; ++i) {
        qtf[i] = fvec[i];
        /* L80: */
    }
    for (j = 1; j <= n; ++j) {
        if (fjac[j + j * ldfjac] == 0.) {
            goto L110;
        }
        sum = 0.;
        for (i = j; i <= n; ++i) {
            sum += fjac[i + j * ldfjac] * qtf[i];
            /* L90: */
        }
        temp = -sum / fjac[j + j * ldfjac];
        for (i = j; i <= n; ++i) {
            qtf[i] += fjac[i + j * ldfjac] * temp;
            /* L100: */
        }
L110:
        /* L120: */
        ;
    }

    /*        copy the triangular factor of the qr factorization into r. */

    sing = FALSE_;
    for (j = 1; j <= n; ++j) {
        l = j;
        jm1 = j - 1;
        if (jm1 < 1) {
            goto L140;
        }
        for (i = 1; i <= jm1; ++i) {
            r__[l] = fjac[i + j * ldfjac];
            l = l + n - i;
            /* L130: */
        }
L140:
        r__[l] = wa1[j];
        if (wa1[j] == 0.) {
            sing = TRUE_;
        }
        /* L150: */
    }

    /*        accumulate the orthogonal factor in fjac. */

    qform(n, n, &fjac[fjac_offset], ldfjac, &wa1[1]);

    /*        rescale if necessary. */

    if (mode == 2) {
        goto L170;
    }
    for (j = 1; j <= n; ++j) /* Computing MAX */
        diag[j] = max(diag[j], wa2[j]);
L170:

    /*        beginning of the inner loop. */

L180:

    /*           if requested, call fcn to enable printing of iterates. */

    if (nprint <= 0) {
        goto L190;
    }
    iflag = 0;
    if ((iter - 1) % nprint == 0) {
        iflag = (*fcn)(p, n, &x[1], &fvec[1], &fjac[fjac_offset], ldfjac, 0);
    }
    if (iflag < 0) {
        goto L300;
    }
L190:

    /*           determine the direction p. */

    dogleg(n, &r__[1], lr, &diag[1], &qtf[1], delta, &wa1[1], &wa2[1], &wa3[
            1]);

    /*           store the direction p and x + p. calculate the norm of p. */

    for (j = 1; j <= n; ++j) {
        wa1[j] = -wa1[j];
        wa2[j] = x[j] + wa1[j];
        wa3[j] = diag[j] * wa1[j];
        /* L200: */
    }
    pnorm = ei_enorm<Scalar>(n, &wa3[1]);

    /*           on the first iteration, adjust the initial step bound. */

    if (iter == 1) {
        delta = min(delta,pnorm);
    }

    /*           evaluate the function at x + p and calculate its norm. */

    iflag = (*fcn)(p, n, &wa2[1], &wa4[1], &fjac[fjac_offset], ldfjac, 1);
    ++nfev;
    if (iflag < 0) {
        goto L300;
    }
    fnorm1 = ei_enorm<Scalar>(n, &wa4[1]);

    /*           compute the scaled actual reduction. */

    actred = -1.;
    if (fnorm1 < fnorm) /* Computing 2nd power */
        actred = 1. - ei_abs2(fnorm1 / fnorm);

    /*           compute the scaled predicted reduction. */

    l = 1;
    for (i = 1; i <= n; ++i) {
        sum = 0.;
        for (j = i; j <= n; ++j) {
            sum += r__[l] * wa1[j];
            ++l;
            /* L210: */
        }
        wa3[i] = qtf[i] + sum;
        /* L220: */
    }
    temp = ei_enorm<Scalar>(n, &wa3[1]);
    prered = 0.;
    if (temp < fnorm) /* Computing 2nd power */
        prered = 1. - ei_abs2(temp / fnorm);

    /*           compute the ratio of the actual to the predicted */
    /*           reduction. */

    ratio = 0.;
    if (prered > 0.) {
        ratio = actred / prered;
    }

    /*           update the step bound. */

    if (ratio >= p1) {
        goto L230;
    }
    ncsuc = 0;
    ++ncfail;
    delta = p5 * delta;
    goto L240;
L230:
    ncfail = 0;
    ++ncsuc;
    if (ratio >= p5 || ncsuc > 1) /* Computing MAX */
        delta = max(delta, pnorm / p5);
    if (ei_abs(ratio - 1.) <= p1) {
        delta = pnorm / p5;
    }
L240:

    /*           test for successful iteration. */

    if (ratio < p0001) {
        goto L260;
    }

    /*           successful iteration. update x, fvec, and their norms. */

    for (j = 1; j <= n; ++j) {
        x[j] = wa2[j];
        wa2[j] = diag[j] * x[j];
        fvec[j] = wa4[j];
        /* L250: */
    }
    xnorm = ei_enorm<Scalar>(n, &wa2[1]);
    fnorm = fnorm1;
    ++iter;
L260:

    /*           determine the progress of the iteration. */

    ++nslow1;
    if (actred >= p001) {
        nslow1 = 0;
    }
    if (jeval) {
        ++nslow2;
    }
    if (actred >= p1) {
        nslow2 = 0;
    }

    /*           test for convergence. */

    if (delta <= xtol * xnorm || fnorm == 0.) {
        info = 1;
    }
    if (info != 0) {
        goto L300;
    }

    /*           tests for termination and stringent tolerances. */

    if (nfev >= maxfev) {
        info = 2;
    }
    /* Computing MAX */
    if (p1 * max(p1 * delta, pnorm) <= epsilon<Scalar>() * xnorm) {
        info = 3;
    }
    if (nslow2 == 5) {
        info = 4;
    }
    if (nslow1 == 10) {
        info = 5;
    }
    if (info != 0) {
        goto L300;
    }

    /*           criterion for recalculating jacobian. */

    if (ncfail == 2) {
        goto L290;
    }

    /*           calculate the rank one modification to the jacobian */
    /*           and update qtf if necessary. */

    for (j = 1; j <= n; ++j) {
        sum = 0.;
        for (i = 1; i <= n; ++i) {
            sum += fjac[i + j * ldfjac] * wa4[i];
            /* L270: */
        }
        wa2[j] = (sum - wa3[j]) / pnorm;
        wa1[j] = diag[j] * (diag[j] * wa1[j] / pnorm);
        if (ratio >= p0001) {
            qtf[j] = sum;
        }
        /* L280: */
    }

    /*           compute the qr factorization of the updated jacobian. */

    r1updt(n, n, &r__[1], lr, &wa1[1], &wa2[1], &wa3[1], &sing);
    r1mpyq(n, n, &fjac[fjac_offset], ldfjac, &wa2[1], &wa3[1]);
    r1mpyq(1, n, &qtf[1], 1, &wa2[1], &wa3[1]);

    /*           end of the inner loop. */

    jeval = FALSE_;
    goto L180;
L290:

    /*        end of the outer loop. */

    goto L30;
L300:

    /*     termination, either normal or user imposed. */

    if (iflag < 0) {
        info = iflag;
    }
    if (nprint > 0) {
        iflag = (*fcn)(p, n, &x[1], &fvec[1], &fjac[fjac_offset], ldfjac, 0);
    }
    return info;

    /*     last card of subroutine hybrj. */

} /* hybrj_ */

