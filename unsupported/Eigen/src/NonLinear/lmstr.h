
template<typename T>
int lmstr_template(minpack_funcderstr_mn fcn, void *p, int m, int n, T *x, 
	T *fvec, T *fjac, int ldfjac, T ftol,
	T xtol, T gtol, int maxfev, T *
	diag, int mode, T factor, int nprint,
	int *nfev, int *njev, int *ipvt, T *qtf, 
	T *wa1, T *wa2, T *wa3, T *wa4)
{
    /* Initialized data */

    /* System generated locals */
    int fjac_dim1, fjac_offset, i__1, i__2;
    T d__1, d__2, d__3;

    /* Local variables */
    int i__, j, l;
    T par, sum;
    int sing;
    int iter;
    T temp, temp1, temp2;
    int iflag;
    T delta;
    T ratio;
    T fnorm, gnorm, pnorm, xnorm, fnorm1, actred, dirder, 
	    epsmch, prered;
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
    fjac_dim1 = ldfjac;
    fjac_offset = 1 + fjac_dim1 * 1;
    fjac -= fjac_offset;

    /* Function Body */

/*     epsmch is the machine precision. */

    epsmch = dpmpar(1);

    info = 0;
    iflag = 0;
    *nfev = 0;
    *njev = 0;

/*     check the input parameters for errors. */

    if (n <= 0 || m < n || ldfjac < n || ftol < 0. || xtol < 0. || 
	    gtol < 0. || maxfev <= 0 || factor <= 0.) {
	goto L340;
    }
    if (mode != 2) {
	goto L20;
    }
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	if (diag[j] <= 0.) {
	    goto L340;
	}
/* L10: */
    }
L20:

/*     evaluate the function at the starting point */
/*     and calculate its norm. */

    iflag = (*fcn)(p, m, n, &x[1], &fvec[1], &wa3[1], 1);
    *nfev = 1;
    if (iflag < 0) {
	goto L340;
    }
    fnorm = enorm(m, &fvec[1]);

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

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	qtf[j] = 0.;
	i__2 = n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    fjac[i__ + j * fjac_dim1] = 0.;
/* L50: */
	}
/* L60: */
    }
    iflag = 2;
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((*fcn)(p, m, n, &x[1], &fvec[1], &wa3[1], iflag) < 0) {
	    goto L340;
	}
	temp = fvec[i__];
	rwupdt(n, &fjac[fjac_offset], ldfjac, &wa3[1], &qtf[1], &temp, &wa1[
		1], &wa2[1]);
	++iflag;
/* L70: */
    }
    ++(*njev);

/*        if the jacobian is rank deficient, call qrfac to */
/*        reorder its columns and update the components of qtf. */

    sing = FALSE_;
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	if (fjac[j + j * fjac_dim1] == 0.) {
	    sing = TRUE_;
	}
	ipvt[j] = j;
	wa2[j] = enorm(j, &fjac[j * fjac_dim1 + 1]);
/* L80: */
    }
    if (! sing) {
	goto L130;
    }
    qrfac(n, n, &fjac[fjac_offset], ldfjac, TRUE_, &ipvt[1], n, &wa1[1], &
	    wa2[1], &wa3[1]);
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	if (fjac[j + j * fjac_dim1] == 0.) {
	    goto L110;
	}
	sum = 0.;
	i__2 = n;
	for (i__ = j; i__ <= i__2; ++i__) {
	    sum += fjac[i__ + j * fjac_dim1] * qtf[i__];
/* L90: */
	}
	temp = -sum / fjac[j + j * fjac_dim1];
	i__2 = n;
	for (i__ = j; i__ <= i__2; ++i__) {
	    qtf[i__] += fjac[i__ + j * fjac_dim1] * temp;
/* L100: */
	}
L110:
	fjac[j + j * fjac_dim1] = wa1[j];
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
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	diag[j] = wa2[j];
	if (wa2[j] == 0.) {
	    diag[j] = 1.;
	}
/* L140: */
    }
L150:

/*        on the first iteration, calculate the norm of the scaled x */
/*        and initialize the step bound delta. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	wa3[j] = diag[j] * x[j];
/* L160: */
    }
    xnorm = enorm(n, &wa3[1]);
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
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	l = ipvt[j];
	if (wa2[l] == 0.) {
	    goto L190;
	}
	sum = 0.;
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    sum += fjac[i__ + j * fjac_dim1] * (qtf[i__] / fnorm);
/* L180: */
	}
/* Computing MAX */
	d__2 = gnorm, d__3 = (d__1 = sum / wa2[l], abs(d__1));
	gnorm = max(d__2,d__3);
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
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	d__1 = diag[j], d__2 = wa2[j];
	diag[j] = max(d__1,d__2);
/* L220: */
    }
L230:

/*        beginning of the inner loop. */

L240:

/*           determine the levenberg-marquardt parameter. */

    lmpar(n, &fjac[fjac_offset], ldfjac, &ipvt[1], &diag[1], &qtf[1], delta,
	     &par, &wa1[1], &wa2[1], &wa3[1], &wa4[1]);

/*           store the direction p and x + p. calculate the norm of p. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	wa1[j] = -wa1[j];
	wa2[j] = x[j] + wa1[j];
	wa3[j] = diag[j] * wa1[j];
/* L250: */
    }
    pnorm = enorm(n, &wa3[1]);

/*           on the first iteration, adjust the initial step bound. */

    if (iter == 1) {
	delta = min(delta,pnorm);
    }

/*           evaluate the function at x + p and calculate its norm. */

    iflag = (*fcn)(p, m, n, &wa2[1], &wa4[1], &wa3[1], 1);
    ++(*nfev);
    if (iflag < 0) {
	goto L340;
    }
    fnorm1 = enorm(m, &wa4[1]);

/*           compute the scaled actual reduction. */

    actred = -1.;
    if (p1 * fnorm1 < fnorm) {
/* Computing 2nd power */
	d__1 = fnorm1 / fnorm;
	actred = 1. - d__1 * d__1;
    }

/*           compute the scaled predicted reduction and */
/*           the scaled directional derivative. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	wa3[j] = 0.;
	l = ipvt[j];
	temp = wa1[l];
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    wa3[i__] += fjac[i__ + j * fjac_dim1] * temp;
/* L260: */
	}
/* L270: */
    }
    temp1 = enorm(n, &wa3[1]) / fnorm;
    temp2 = sqrt(par) * pnorm / fnorm;
/* Computing 2nd power */
    d__1 = temp1;
/* Computing 2nd power */
    d__2 = temp2;
    prered = d__1 * d__1 + d__2 * d__2 / p5;
/* Computing 2nd power */
    d__1 = temp1;
/* Computing 2nd power */
    d__2 = temp2;
    dirder = -(d__1 * d__1 + d__2 * d__2);

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
    d__1 = delta, d__2 = pnorm / p1;
    delta = temp * min(d__1,d__2);
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

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	x[j] = wa2[j];
	wa2[j] = diag[j] * x[j];
/* L310: */
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	fvec[i__] = wa4[i__];
/* L320: */
    }
    xnorm = enorm(n, &wa2[1]);
    fnorm = fnorm1;
    ++iter;
L330:

/*           tests for convergence. */

    if (abs(actred) <= ftol && prered <= ftol && p5 * ratio <= 1.) {
	info = 1;
    }
    if (delta <= xtol * xnorm) {
	info = 2;
    }
    if (abs(actred) <= ftol && prered <= ftol && p5 * ratio <= 1. && info 
	    == 2) {
	info = 3;
    }
    if (info != 0) {
	goto L340;
    }

/*           tests for termination and stringent tolerances. */

    if (*nfev >= maxfev) {
	info = 5;
    }
    if (abs(actred) <= epsmch && prered <= epsmch && p5 * ratio <= 1.) {
	info = 6;
    }
    if (delta <= epsmch * xnorm) {
	info = 7;
    }
    if (gnorm <= epsmch) {
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

