
template <typename Scalar>
void ei_lmpar(int n, Scalar *r__, int ldr, 
	const int *ipvt, const Scalar *diag, const Scalar *qtb, Scalar delta, 
	Scalar &par, Scalar *x, Scalar *sdiag, Scalar *wa1, 
	Scalar *wa2)
{
    /* System generated locals */
    int r_dim1, r_offset;

    /* Local variables */
    int i, j, k, l;
    Scalar fp;
    int jm1, jp1;
    Scalar sum, parc, parl;
    int iter;
    Scalar temp, paru;
    int nsing;
    Scalar gnorm;
    Scalar dxnorm;

    /* Parameter adjustments */
    --wa2;
    --wa1;
    --sdiag;
    --x;
    --qtb;
    --diag;
    --ipvt;
    r_dim1 = ldr;
    r_offset = 1 + r_dim1 * 1;
    r__ -= r_offset;

    /* Function Body */
    const Scalar dwarf = std::numeric_limits<Scalar>::min();

/*     compute and store in x the gauss-newton direction. if the */
/*     jacobian is rank-deficient, obtain a least squares solution. */

    nsing = n;
    for (j = 1; j <= n; ++j) {
	wa1[j] = qtb[j];
	if (r__[j + j * r_dim1] == 0. && nsing == n) {
	    nsing = j - 1;
	}
	if (nsing < n) {
	    wa1[j] = 0.;
	}
/* L10: */
    }
    if (nsing < 1) {
	goto L50;
    }
    for (k = 1; k <= nsing; ++k) {
	j = nsing - k + 1;
	wa1[j] /= r__[j + j * r_dim1];
	temp = wa1[j];
	jm1 = j - 1;
	if (jm1 < 1) {
	    goto L30;
	}
	for (i = 1; i <= jm1; ++i) {
	    wa1[i] -= r__[i + j * r_dim1] * temp;
/* L20: */
	}
L30:
/* L40: */
	;
    }
L50:
    for (j = 1; j <= n; ++j) {
	l = ipvt[j];
	x[l] = wa1[j];
/* L60: */
    }

/*     initialize the iteration counter. */
/*     evaluate the function at the origin, and test */
/*     for acceptance of the gauss-newton direction. */

    iter = 0;
    for (j = 1; j <= n; ++j) {
	wa2[j] = diag[j] * x[j];
/* L70: */
    }
    dxnorm = Map< Matrix< Scalar, Dynamic, 1 > >(&wa2[1],n).blueNorm();
    fp = dxnorm - delta;
    if (fp <= Scalar(0.1) * delta) {
	goto L220;
    }

/*     if the jacobian is not rank deficient, the newton */
/*     step provides a lower bound, parl, for the zero of */
/*     the function. otherwise set this bound to zero. */

    parl = 0.;
    if (nsing < n) {
	goto L120;
    }
    for (j = 1; j <= n; ++j) {
	l = ipvt[j];
	wa1[j] = diag[l] * (wa2[l] / dxnorm);
/* L80: */
    }
    for (j = 1; j <= n; ++j) {
	sum = 0.;
	jm1 = j - 1;
	if (jm1 < 1) {
	    goto L100;
	}
	for (i = 1; i <= jm1; ++i) {
	    sum += r__[i + j * r_dim1] * wa1[i];
/* L90: */
	}
L100:
	wa1[j] = (wa1[j] - sum) / r__[j + j * r_dim1];
/* L110: */
    }
    temp = Map< Matrix< Scalar, Dynamic, 1 > >(&wa1[1],n).blueNorm();
    parl = fp / delta / temp / temp;
L120:

/*     calculate an upper bound, paru, for the zero of the function. */

    for (j = 1; j <= n; ++j) {
	sum = 0.;
	for (i = 1; i <= j; ++i) {
	    sum += r__[i + j * r_dim1] * qtb[i];
/* L130: */
	}
	l = ipvt[j];
	wa1[j] = sum / diag[l];
/* L140: */
    }
    gnorm = Map< Matrix< Scalar, Dynamic, 1 > >(&wa1[1],n).stableNorm();
    paru = gnorm / delta;
    if (paru == 0.) {
	paru = dwarf / std::min(delta,Scalar(0.1));
    }

/*     if the input par lies outside of the interval (parl,paru), */
/*     set par to the closer endpoint. */

    par = std::max(par,parl);
    par = std::min(par,paru);
    if (par == 0.) {
	par = gnorm / dxnorm;
    }

/*     beginning of an iteration. */

L150:
    ++iter;

/*        evaluate the function at the current value of par. */

    if (par == 0.) {
/* Computing MAX */
	par = std::max(dwarf,Scalar(.001) * paru);
    }
    temp = ei_sqrt(par);
    for (j = 1; j <= n; ++j) {
	wa1[j] = temp * diag[j];
/* L160: */
    }
    ei_qrsolv<Scalar>(n, &r__[r_offset], ldr, &ipvt[1], &wa1[1], &qtb[1], &x[1], &sdiag[1], &wa2[1]);
    for (j = 1; j <= n; ++j) {
	wa2[j] = diag[j] * x[j];
/* L170: */
    }
    dxnorm = Map< Matrix< Scalar, Dynamic, 1 > >(&wa2[1],n).blueNorm();
    temp = fp;
    fp = dxnorm - delta;

/*        if the function is small enough, accept the current value */
/*        of par. also test for the exceptional cases where parl */
/*        is zero or the number of iterations has reached 10. */

    if (ei_abs(fp) <= Scalar(0.1) * delta || (parl == 0. && fp <= temp && temp < 0.) ||
	     iter == 10) {
	goto L220;
    }

/*        compute the newton correction. */

    for (j = 1; j <= n; ++j) {
	l = ipvt[j];
	wa1[j] = diag[l] * (wa2[l] / dxnorm);
/* L180: */
    }
    for (j = 1; j <= n; ++j) {
	wa1[j] /= sdiag[j];
	temp = wa1[j];
	jp1 = j + 1;
	if (n < jp1) {
	    goto L200;
	}
	for (i = jp1; i <= n; ++i) {
	    wa1[i] -= r__[i + j * r_dim1] * temp;
/* L190: */
	}
L200:
/* L210: */
	;
    }
    temp = Map< Matrix< Scalar, Dynamic, 1 > >(&wa1[1],n).blueNorm();
    parc = fp / delta / temp / temp;

/*        depending on the sign of the function, update parl or paru. */

    if (fp > 0.) {
	parl = std::max(parl,par);
    }
    if (fp < 0.) {
	paru = std::min(paru,par);
    }

/*        compute an improved estimate for par. */

/* Computing MAX */
    par = std::max(parl,par+parc);

/*        end of an iteration. */

    goto L150;
L220:

/*     termination. */

    if (iter == 0) {
	par = 0.;
    }
    return;

/*     last card of subroutine lmpar. */

} /* lmpar_ */

