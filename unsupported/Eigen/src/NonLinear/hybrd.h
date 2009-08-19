
template<typename T>
int hybrd_template(minpack_func_nn fcn, void *p, int n, T *x, T *
	fvec, T xtol, int maxfev, int ml, int mu, 
	T epsfcn, T *diag, int mode, T
	factor, int nprint, int *nfev, T *
	fjac, int ldfjac, T *r__, int lr, T *qtf, 
	T *wa1, T *wa2, T *wa3, T *wa4)
{
    /* Initialized data */

    /* System generated locals */
    int fjac_dim1, fjac_offset, i__1, i__2;
    T d__1, d__2;

    /* Local variables */
    int i__, j, l, jm1, iwa[1];
    T sum;
    int sing;
    int iter;
    T temp;
    int msum, iflag;
    T delta;
    int jeval;
    int ncsuc;
    T ratio;
    T fnorm;
    T pnorm, xnorm, fnorm1;
    int nslow1, nslow2;
    int ncfail;
    T actred, prered;
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
    fjac_dim1 = ldfjac;
    fjac_offset = 1 + fjac_dim1 * 1;
    fjac -= fjac_offset;
    --r__;

    /* Function Body */

    info = 0;
    iflag = 0;
    *nfev = 0;

/*     check the input parameters for errors. */

    if (n <= 0 || xtol < 0. || maxfev <= 0 || ml < 0 || mu < 0 ||
	    factor <= 0. || ldfjac < n || lr < n * (n + 1) / 2) {
	goto L300;
    }
    if (mode != 2) {
	goto L20;
    }
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	if (diag[j] <= 0.) {
	    goto L300;
	}
/* L10: */
    }
L20:

/*     evaluate the function at the starting point */
/*     and calculate its norm. */

    iflag = (*fcn)(p, n, &x[1], &fvec[1], 1);
    *nfev = 1;
    if (iflag < 0) {
	goto L300;
    }
    fnorm = enorm(n, &fvec[1]);

/*     determine the number of calls to fcn needed to compute */
/*     the jacobian matrix. */

/* Computing MIN */
    i__1 = ml + mu + 1;
    msum = min(i__1,n);

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

    iflag = fdjac1(fcn, p, n, &x[1], &fvec[1], &fjac[fjac_offset], ldfjac,
	     ml, mu, epsfcn, &wa1[1], &wa2[1]);
    *nfev += msum;
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
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	diag[j] = wa2[j];
	if (wa2[j] == 0.) {
	    diag[j] = 1.;
	}
/* L40: */
    }
L50:

/*        on the first iteration, calculate the norm of the scaled x */
/*        and initialize the step bound delta. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	wa3[j] = diag[j] * x[j];
/* L60: */
    }
    xnorm = enorm(n, &wa3[1]);
    delta = factor * xnorm;
    if (delta == 0.) {
	delta = factor;
    }
L70:

/*        form (q transpose)*fvec and store in qtf. */

    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	qtf[i__] = fvec[i__];
/* L80: */
    }
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
/* L120: */
	;
    }

/*        copy the triangular factor of the qr factorization into r. */

    sing = FALSE_;
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	l = j;
	jm1 = j - 1;
	if (jm1 < 1) {
	    goto L140;
	}
	i__2 = jm1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    r__[l] = fjac[i__ + j * fjac_dim1];
	    l = l + n - i__;
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
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	d__1 = diag[j], d__2 = wa2[j];
	diag[j] = max(d__1,d__2);
/* L160: */
    }
L170:

/*        beginning of the inner loop. */

L180:

/*           if requested, call fcn to enable printing of iterates. */

    if (nprint <= 0) {
	goto L190;
    }
    iflag = 0;
    if ((iter - 1) % nprint == 0) {
	iflag = (*fcn)(p, n, &x[1], &fvec[1], 0);
    }
    if (iflag < 0) {
	goto L300;
    }
L190:

/*           determine the direction p. */

    dogleg(n, &r__[1], lr, &diag[1], &qtf[1], delta, &wa1[1], &wa2[1], &wa3[
	    1]);

/*           store the direction p and x + p. calculate the norm of p. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	wa1[j] = -wa1[j];
	wa2[j] = x[j] + wa1[j];
	wa3[j] = diag[j] * wa1[j];
/* L200: */
    }
    pnorm = enorm(n, &wa3[1]);

/*           on the first iteration, adjust the initial step bound. */

    if (iter == 1) {
	delta = min(delta,pnorm);
    }

/*           evaluate the function at x + p and calculate its norm. */

    iflag = (*fcn)(p, n, &wa2[1], &wa4[1], 1);
    ++(*nfev);
    if (iflag < 0) {
	goto L300;
    }
    fnorm1 = enorm(n, &wa4[1]);

/*           compute the scaled actual reduction. */

    actred = -1.;
    if (fnorm1 < fnorm) {
/* Computing 2nd power */
	d__1 = fnorm1 / fnorm;
	actred = 1. - d__1 * d__1;
    }

/*           compute the scaled predicted reduction. */

    l = 1;
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	sum = 0.;
	i__2 = n;
	for (j = i__; j <= i__2; ++j) {
	    sum += r__[l] * wa1[j];
	    ++l;
/* L210: */
	}
	wa3[i__] = qtf[i__] + sum;
/* L220: */
    }
    temp = enorm(n, &wa3[1]);
    prered = 0.;
    if (temp < fnorm) {
/* Computing 2nd power */
	d__1 = temp / fnorm;
	prered = 1. - d__1 * d__1;
    }

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
    if (ratio >= p5 || ncsuc > 1) {
/* Computing MAX */
	d__1 = delta, d__2 = pnorm / p5;
	delta = max(d__1,d__2);
    }
    if (fabs(ratio - 1.) <= p1) {
	delta = pnorm / p5;
    }
L240:

/*           test for successful iteration. */

    if (ratio < p0001) {
	goto L260;
    }

/*           successful iteration. update x, fvec, and their norms. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	x[j] = wa2[j];
	wa2[j] = diag[j] * x[j];
	fvec[j] = wa4[j];
/* L250: */
    }
    xnorm = enorm(n, &wa2[1]);
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

    if (*nfev >= maxfev) {
	info = 2;
    }
/* Computing MAX */
    d__1 = p1 * delta;
    if (p1 * max(d__1,pnorm) <= epsilon<T>() * xnorm) {
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

/*           criterion for recalculating jacobian approximation */
/*           by forward differences. */

    if (ncfail == 2) {
	goto L290;
    }

/*           calculate the rank one modification to the jacobian */
/*           and update qtf if necessary. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	sum = 0.;
	i__2 = n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    sum += fjac[i__ + j * fjac_dim1] * wa4[i__];
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
	(*fcn)(p, n, &x[1], &fvec[1], 0);
    }
    return info;

/*     last card of subroutine hybrd. */

} /* hybrd_ */

