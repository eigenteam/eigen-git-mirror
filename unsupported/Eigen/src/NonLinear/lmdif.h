
template<typename Functor, typename Scalar>
int ei_lmdif(
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
        )
{
    const int m = fvec.size(), n = x.size();
    Matrix< Scalar, Dynamic, 1 >
        wa1(n), wa2(n), wa3(n),
        wa4(m);
    int ldfjac = m;

    ipvt.resize(n);
    fjac.resize(ldfjac, n);
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

    if (n <= 0 || m < n || ldfjac < m || ftol < 0. || xtol < 0. || 
            gtol < 0. || maxfev <= 0 || factor <= 0.) {
        goto L300;
    }
    if (mode != 2) {
        goto L20;
    }
    for (j = 0; j < n; ++j) {
        if (diag[j] <= 0.) {
            goto L300;
        }
        /* L10: */
    }
L20:

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    iflag = Functor::f(0, m, n, x.data(), fvec.data(), 1);
    nfev = 1;
    if (iflag < 0) {
        goto L300;
    }
    fnorm = fvec.stableNorm();

    /*     initialize levenberg-marquardt parameter and iteration counter. */

    par = 0.;
    iter = 1;

    /*     beginning of the outer loop. */

L30:

    /*        calculate the jacobian matrix. */

    iflag = ei_fdjac2<Scalar>(Functor::f, 0, m, n, x.data(), fvec.data(), fjac.data(), ldfjac,
            epsfcn, wa4.data());
    nfev += n;
    if (iflag < 0) {
        goto L300;
    }

    /*        if requested, call Functor::f to enable printing of iterates. */

    if (nprint <= 0) {
        goto L40;
    }
    iflag = 0;
    if ((iter - 1) % nprint == 0) {
        iflag = Functor::f(0, m, n, x.data(), fvec.data(), 0);
    }
    if (iflag < 0) {
        goto L300;
    }
L40:

    /*        compute the qr factorization of the jacobian. */

    ei_qrfac<Scalar>(m, n, fjac.data(), ldfjac, true, ipvt.data(), n, wa1.data(), wa2.data(), wa3.data());
    ipvt.cwise()-=1; // qrfac() creates ipvt with fortran convetion (1->n), convert it to c (0->n-1)

    /*        on the first iteration and if mode is 1, scale according */
    /*        to the norms of the columns of the initial jacobian. */

    if (iter != 1) {
        goto L80;
    }
    if (mode == 2) {
        goto L60;
    }
    for (j = 0; j < n; ++j) {
        diag[j] = wa2[j];
        if (wa2[j] == 0.) {
            diag[j] = 1.;
        }
        /* L50: */
    }
L60:

    /*        on the first iteration, calculate the norm of the scaled x */
    /*        and initialize the step bound delta. */

    for (j = 0; j < n; ++j) {
        wa3[j] = diag[j] * x[j];
        /* L70: */
    }
    xnorm = wa3.stableNorm();;
    delta = factor * xnorm;
    if (delta == 0.) {
        delta = factor;
    }
L80:

    /*        form (q transpose)*fvec and store the first n components in */
    /*        qtf. */

    for (i = 0; i < m; ++i) {
        wa4[i] = fvec[i];
        /* L90: */
    }
    for (j = 0; j < n; ++j) {
        if (fjac[j + j * ldfjac] == 0.) {
            goto L120;
        }
        sum = 0.;
        for (i = j; i < m; ++i) {
            sum += fjac[i + j * ldfjac] * wa4[i];
            /* L100: */
        }
        temp = -sum / fjac[j + j * ldfjac];
        for (i = j; i < m; ++i) {
            wa4[i] += fjac[i + j * ldfjac] * temp;
            /* L110: */
        }
L120:
        fjac[j + j * ldfjac] = wa1[j];
        qtf[j] = wa4[j];
        /* L130: */
    }

    /*        compute the norm of the scaled gradient. */

    gnorm = 0.;
    if (fnorm == 0.) {
        goto L170;
    }
    for (j = 0; j < n; ++j) {
        l = ipvt[j];
        if (wa2[l] == 0.) {
            goto L150;
        }
        sum = 0.;
        for (i = 0; i <= j; ++i) {
            sum += fjac[i + j * ldfjac] * (qtf[i] / fnorm);
            /* L140: */
        }
        /* Computing MAX */
        gnorm = std::max(gnorm, ei_abs(sum / wa2[l]));
L150:
        /* L160: */
        ;
    }
L170:

    /*        test for convergence of the gradient norm. */

    if (gnorm <= gtol) {
        info = 4;
    }
    if (info != 0) {
        goto L300;
    }

    /*        rescale if necessary. */

    if (mode == 2) {
        goto L190;
    }
    for (j = 0; j < n; ++j) /* Computing MAX */
        diag[j] = std::max(diag[j], wa2[j]);
L190:

    /*        beginning of the inner loop. */

L200:

    /*           determine the levenberg-marquardt parameter. */

    ipvt.cwise()+=1; // lmpar() expects the fortran convention (as qrfac provides)
    ei_lmpar<Scalar>(n, fjac.data(), ldfjac, ipvt.data(), diag.data(), qtf.data(), delta,
            &par, wa1.data(), wa2.data(), wa3.data(), wa4.data());
    ipvt.cwise()-=1;

    /*           store the direction p and x + p. calculate the norm of p. */

    for (j = 0; j < n; ++j) {
        wa1[j] = -wa1[j];
        wa2[j] = x[j] + wa1[j];
        wa3[j] = diag[j] * wa1[j];
        /* L210: */
    }
    pnorm = wa3.stableNorm();

    /*           on the first iteration, adjust the initial step bound. */

    if (iter == 1) {
        delta = std::min(delta,pnorm);
    }

    /*           evaluate the function at x + p and calculate its norm. */

    iflag = Functor::f(0, m, n, wa2.data(), wa4.data(), 1);
    ++nfev;
    if (iflag < 0) {
        goto L300;
    }
    fnorm1 = wa4.stableNorm();

    /*           compute the scaled actual reduction. */

    actred = -1.;
    if (Scalar(.1) * fnorm1 < fnorm) /* Computing 2nd power */
        actred = 1. - ei_abs2(fnorm1 / fnorm);

    /*           compute the scaled predicted reduction and */
    /*           the scaled directional derivative. */

    for (j = 0; j < n; ++j) {
        wa3[j] = 0.;
        l = ipvt[j];
        temp = wa1[l];
        for (i = 0; i <= j; ++i) {
            wa3[i] += fjac[i + j * ldfjac] * temp;
            /* L220: */
        }
        /* L230: */
    }
    temp1 = ei_abs2(wa3.stableNorm() / fnorm);
    temp2 = ei_abs2(ei_sqrt(par) * pnorm / fnorm);
    /* Computing 2nd power */
    prered = temp1 + temp2 / Scalar(.5);
    dirder = -(temp1 + temp2);

    /*           compute the ratio of the actual to the predicted */
    /*           reduction. */

    ratio = 0.;
    if (prered != 0.) {
        ratio = actred / prered;
    }

    /*           update the step bound. */

    if (ratio > Scalar(.25)) {
        goto L240;
    }
    if (actred >= 0.) {
        temp = Scalar(.5);
    }
    if (actred < 0.) {
        temp = Scalar(.5) * dirder / (dirder + Scalar(.5) * actred);
    }
    if (Scalar(.1) * fnorm1 >= fnorm || temp < Scalar(.1)) {
        temp = Scalar(.1);
    }
    /* Computing MIN */
    delta = temp * std::min(delta, pnorm / Scalar(.1));
    par /= temp;
    goto L260;
L240:
    if (par != 0. && ratio < Scalar(.75)) {
        goto L250;
    }
    delta = pnorm / Scalar(.5);
    par = Scalar(.5) * par;
L250:
L260:

    /*           test for successful iteration. */

    if (ratio < Scalar(1e-4)) {
        goto L290;
    }

    /*           successful iteration. update x, fvec, and their norms. */

    for (j = 0; j < n; ++j) {
        x[j] = wa2[j];
        wa2[j] = diag[j] * x[j];
        /* L270: */
    }
    for (i = 0; i < m; ++i) {
        fvec[i] = wa4[i];
        /* L280: */
    }
    xnorm = wa2.stableNorm();
    fnorm = fnorm1;
    ++iter;
L290:

    /*           tests for convergence. */

    if (ei_abs(actred) <= ftol && prered <= ftol && Scalar(.5) * ratio <= 1.) {
        info = 1;
    }
    if (delta <= xtol * xnorm) {
        info = 2;
    }
    if (ei_abs(actred) <= ftol && prered <= ftol && Scalar(.5) * ratio <= 1. && info 
            == 2) {
        info = 3;
    }
    if (info != 0) {
        goto L300;
    }

    /*           tests for termination and stringent tolerances. */

    if (nfev >= maxfev) {
        info = 5;
    }
    if (ei_abs(actred) <= epsilon<Scalar>() && prered <= epsilon<Scalar>() && Scalar(.5) * ratio <= 1.) {
        info = 6;
    }
    if (delta <= epsilon<Scalar>() * xnorm) {
        info = 7;
    }
    if (gnorm <= epsilon<Scalar>()) {
        info = 8;
    }
    if (info != 0) {
        goto L300;
    }

    /*           end of the inner loop. repeat if iteration unsuccessful. */

    if (ratio < Scalar(1e-4)) {
        goto L200;
    }

    /*        end of the outer loop. */

    goto L30;
L300:

    /*     termination, either normal or user imposed. */

    if (iflag < 0) {
        info = iflag;
    }
    iflag = 0;
    if (nprint > 0) {
        iflag = Functor::f(0, m, n, x.data(), fvec.data(), 0);
    }
    return info;

    /*     last card of subroutine lmdif. */

} /* lmdif_ */

