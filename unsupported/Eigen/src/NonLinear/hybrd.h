
template<typename Functor, typename Scalar>
int ei_hybrd(
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        int &nfev,
        Matrix< Scalar, Dynamic, Dynamic > &fjac,
        Matrix< Scalar, Dynamic, 1 >  &R,
        Matrix< Scalar, Dynamic, 1 >  &qtf,
        Matrix< Scalar, Dynamic, 1 >  &diag,
        int mode=1,
        int nb_of_subdiagonals = -1,
        int nb_of_superdiagonals = -1,
        int maxfev = 2000,
        Scalar factor = Scalar(100.),
        Scalar xtol = ei_sqrt(epsilon<Scalar>()),
        Scalar epsfcn = Scalar(0.),
        int nprint=0
        )
{
    const int n = x.size();
    int lr = (n*(n+1))/2;
    Matrix< Scalar, Dynamic, 1 > wa1(n), wa2(n), wa3(n), wa4(n);


    if (nb_of_subdiagonals<0) nb_of_subdiagonals = n-1;
    if (nb_of_superdiagonals<0) nb_of_superdiagonals = n-1;
    fvec.resize(n);
    qtf.resize(n);
    R.resize(lr);
    int ldfjac = n;
    fjac.resize(ldfjac, n);

    /* Local variables */
    int i, j, l, iwa[1];
    Scalar sum;
    int sing;
    int iter;
    Scalar temp;
    int msum, iflag;
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

    /* Function Body */

    info = 0;
    iflag = 0;
    nfev = 0;

    /*     check the input parameters for errors. */

    if (n <= 0 || xtol < 0. || maxfev <= 0 || nb_of_subdiagonals < 0 || nb_of_superdiagonals < 0 ||
            factor <= 0. || ldfjac < n || lr < n * (n + 1) / 2) {
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

    iflag = Functor::f(0, n, x.data(), fvec.data(), 1);
    nfev = 1;
    if (iflag < 0) {
        goto L300;
    }
    fnorm = fvec.stableNorm();

    /*     determine the number of calls to fcn needed to compute */
    /*     the jacobian matrix. */

    /* Computing MIN */
    msum = std::min(nb_of_subdiagonals + nb_of_superdiagonals + 1, n);

    /*     initialize iteration counter and monitors. */

    iter = 1;
    ncsuc = 0;
    ncfail = 0;
    nslow1 = 0;
    nslow2 = 0;

    /*     beginning of the outer loop. */

L30:
    jeval = true;

    /*        calculate the jacobian matrix. */

    iflag = ei_fdjac1<Scalar>(Functor::f, 0, n, x.data(), fvec.data(), fjac.data(), ldfjac,
            nb_of_subdiagonals, nb_of_superdiagonals, epsfcn, wa1.data(), wa2.data());
    nfev += msum;
    if (iflag < 0) {
        goto L300;
    }

    /*        compute the qr factorization of the jacobian. */

    ei_qrfac<Scalar>(n, n, fjac.data(), ldfjac, false, iwa, 1, wa1.data(), wa2.data(), wa3.data());

    /*        on the first iteration and if mode is 1, scale according */
    /*        to the norms of the columns of the initial jacobian. */

    if (iter != 1) {
        goto L70;
    }
    if (mode == 2) {
        goto L50;
    }
    for (j = 0; j < n; ++j) {
        diag[j] = wa2[j];
        if (wa2[j] == 0.) {
            diag[j] = 1.;
        }
        /* L40: */
    }
L50:

    /*        on the first iteration, calculate the norm of the scaled x */
    /*        and initialize the step bound delta. */

    for (j = 0; j < n; ++j) {
        wa3[j] = diag[j] * x[j];
        /* L60: */
    }
    xnorm = wa3.stableNorm();
    delta = factor * xnorm;
    if (delta == 0.) {
        delta = factor;
    }
L70:

    /*        form (q transpose)*fvec and store in qtf. */

    for (i = 0; i < n; ++i) {
        qtf[i] = fvec[i];
        /* L80: */
    }
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
        /* L120: */
        ;
    }

    /*        copy the triangular factor of the qr factorization into r. */

    sing = false;
    for (j = 0; j < n; ++j) {
        l = j;
        if (j) {
            for (i = 0; i < j; ++i) {
                R[l] = fjac(i,j);
                l = l + n - i -1;
                /* L130: */
            }
        }
        R[l] = wa1[j];
        if (wa1[j] == 0.) {
            sing = true;
        }
        /* L150: */
    }

    /*        accumulate the orthogonal factor in fjac. */

    ei_qform<Scalar>(n, n, fjac.data(), ldfjac, wa1.data());

    /*        rescale if necessary. */

    if (mode == 2) {
        goto L170;
    }
    /* Computing MAX */
    for (j = 0; j < n; ++j)
        diag[j] = std::max(diag[j], wa2[j]);
L170:

    /*        beginning of the inner loop. */

L180:

    /*           if requested, call fcn to enable printing of iterates. */

    if (nprint <= 0) {
        goto L190;
    }
    iflag = 0;
    if ((iter - 1) % nprint == 0) {
        iflag = Functor::f(0, n, x.data(), fvec.data(), 0);
    }
    if (iflag < 0) {
        goto L300;
    }
L190:

    /*           determine the direction p. */

    ei_dogleg<Scalar>(n, R.data(), lr, diag.data(), qtf.data(), delta, wa1.data(), wa2.data(), wa3.data());

    /*           store the direction p and x + p. calculate the norm of p. */

    for (j = 0; j < n; ++j) {
        wa1[j] = -wa1[j];
        wa2[j] = x[j] + wa1[j];
        wa3[j] = diag[j] * wa1[j];
        /* L200: */
    }
    pnorm = wa3.stableNorm();

    /*           on the first iteration, adjust the initial step bound. */

    if (iter == 1) {
        delta = std::min(delta,pnorm);
    }

    /*           evaluate the function at x + p and calculate its norm. */

    iflag = Functor::f(0, n, wa2.data(), wa4.data(), 1);
    ++nfev;
    if (iflag < 0) {
        goto L300;
    }
    fnorm1 = wa4.stableNorm();

    /*           compute the scaled actual reduction. */

    actred = -1.;
    if (fnorm1 < fnorm) /* Computing 2nd power */
        actred = 1. - ei_abs2(fnorm1 / fnorm);

    /*           compute the scaled predicted reduction. */

    l = 0;
    for (i = 0; i < n; ++i) {
        sum = 0.;
        for (j = i; j < n; ++j) {
            sum += R[l] * wa1[j];
            ++l;
            /* L210: */
        }
        wa3[i] = qtf[i] + sum;
        /* L220: */
    }
    temp = wa3.stableNorm();
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

    if (ratio >= Scalar(.1)) {
        goto L230;
    }
    ncsuc = 0;
    ++ncfail;
    delta = Scalar(.5) * delta;
    goto L240;
L230:
    ncfail = 0;
    ++ncsuc;
    if (ratio >= Scalar(.5) || ncsuc > 1) /* Computing MAX */
        delta = std::max(delta, pnorm / Scalar(.5));
    if (ei_abs(ratio - 1.) <= Scalar(.1)) {
        delta = pnorm / Scalar(.5);
    }
L240:

    /*           test for successful iteration. */

    if (ratio < Scalar(1e-4)) {
        goto L260;
    }

    /*           successful iteration. update x, fvec, and their norms. */

    for (j = 0; j < n; ++j) {
        x[j] = wa2[j];
        wa2[j] = diag[j] * x[j];
        fvec[j] = wa4[j];
        /* L250: */
    }
    temp = wa2.stableNorm();
    fnorm = fnorm1;
    ++iter;
L260:

    /*           determine the progress of the iteration. */

    ++nslow1;
    if (actred >= Scalar(.001)) {
        nslow1 = 0;
    }
    if (jeval) {
        ++nslow2;
    }
    if (actred >= Scalar(.1)) {
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
    if (Scalar(.1) * std::max(Scalar(.1) * delta, pnorm) <= epsilon<Scalar>() * xnorm)
        info = 3;
    if (nslow2 == 5)
        info = 4;
    if (nslow1 == 10)
        info = 5;
    if (info != 0)
        goto L300;

    /*           criterion for recalculating jacobian approximation */
    /*           by forward differences. */

    if (ncfail == 2)
        goto L290;

    /*           calculate the rank one modification to the jacobian */
    /*           and update qtf if necessary. */

    for (j = 0; j < n; ++j) {
        sum = 0.;
        for (i = 0; i < n; ++i) {
            sum += fjac(i,j) * wa4[i];
            /* L270: */
        }
        wa2[j] = (sum - wa3[j]) / pnorm;
        wa1[j] = diag[j] * (diag[j] * wa1[j] / pnorm);
        if (ratio >= Scalar(1e-4)) {
            qtf[j] = sum;
        }
        /* L280: */
    }

    /*           compute the qr factorization of the updated jacobian. */

    ei_r1updt<Scalar>(n, n, R.data(), lr, wa1.data(), wa2.data(), wa3.data(), &sing);
    ei_r1mpyq<Scalar>(n, n, fjac.data(), ldfjac, wa2.data(), wa3.data());
    ei_r1mpyq<Scalar>(1, n, qtf.data(), 1, wa2.data(), wa3.data());

    /*           end of the inner loop. */

    jeval = false;
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
        Functor::f(0, n, x.data(), fvec.data(), 0);
    }
    return info;

    /*     last card of subroutine hybrd. */

} /* hybrd_ */

