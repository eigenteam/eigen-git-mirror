
#define chkder_log10e 0.43429448190325182765
#define chkder_factor 100.

/* Table of constant values */

template<typename T>
void chkder_template(int m, int n, const T *x, 
	T *fvec, T *fjac, int ldfjac, T *xp, 
	T *fvecp, int mode, T *err)
{
    /* System generated locals */
    int fjac_dim1, fjac_offset, i__1, i__2;

    /* Local variables */
    int i__, j;
    T eps, epsf, temp, epsmch;
    T epslog;

    /* Parameter adjustments */
    --err;
    --fvecp;
    --fvec;
    --xp;
    --x;
    fjac_dim1 = ldfjac;
    fjac_offset = 1 + fjac_dim1 * 1;
    fjac -= fjac_offset;

    /* Function Body */

/*     epsmch is the machine precision. */

    epsmch = dpmpar(1);

    eps = sqrt(epsmch);

    if (mode == 2) {
	goto L20;
    }

/*        mode = 1. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	temp = eps * fabs(x[j]);
	if (temp == 0.) {
	    temp = eps;
	}
	xp[j] = x[j] + temp;
/* L10: */
    }
    /* goto L70; */
    return;
L20:

/*        mode = 2. */

    epsf = chkder_factor * epsmch;
    epslog = chkder_log10e * log(eps);
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	err[i__] = 0.;
/* L30: */
    }
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	temp = fabs(x[j]);
	if (temp == 0.) {
	    temp = 1.;
	}
	i__2 = m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    err[i__] += temp * fjac[i__ + j * fjac_dim1];
/* L40: */
	}
/* L50: */
    }
    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	temp = 1.;
	if (fvec[i__] != 0. && fvecp[i__] != 0. && fabs(fvecp[i__] - 
		fvec[i__]) >= epsf * fabs(fvec[i__]))
		 {
	    temp = eps * fabs((fvecp[i__] - fvec[i__]) / eps - err[i__]) 
		    / (fabs(fvec[i__]) +
                       fabs(fvecp[i__]));
	}
	err[i__] = 1.;
	if (temp > epsmch && temp < eps) {
	    err[i__] = (chkder_log10e * log(temp) - epslog) / epslog;
	}
	if (temp >= eps) {
	    err[i__] = 0.;
	}
/* L60: */
    }
/* L70: */

    /* return 0; */

/*     last card of subroutine chkder. */

} /* chkder_ */

