
template<typename T>
int hybrj1_template(minpack_funcder_nn fcn, void *p, int n, T *x, T *
	fvec, T *fjac, int ldfjac, T tol,
	T *wa, int lwa)
{
    /* Initialized data */

    const T factor = 100.;

    /* System generated locals */
    int fjac_dim1, fjac_offset, i__1;

    /* Local variables */
    int j, lr, mode, nfev, njev;
    T xtol;
    int maxfev, nprint;
    int info;

    /* Parameter adjustments */
    --fvec;
    --x;
    fjac_dim1 = ldfjac;
    fjac_offset = 1 + fjac_dim1 * 1;
    fjac -= fjac_offset;
    --wa;

    /* Function Body */
    info = 0;

/*     check the input parameters for errors. */

    if (n <= 0 || ldfjac < n || tol < 0. || lwa < n * (n + 13) / 2) {
	/* goto L20; */
        return info;
    }

/*     call hybrj. */

    maxfev = (n + 1) * 100;
    xtol = tol;
    mode = 2;
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	wa[j] = 1.;
/* L10: */
    }
    nprint = 0;
    lr = n * (n + 1) / 2;
    info = hybrj(fcn, p, n, &x[1], &fvec[1], &fjac[fjac_offset], ldfjac, xtol,
	    maxfev, &wa[1], mode, factor, nprint, &nfev, &njev, &wa[
	    n * 6 + 1], lr, &wa[n + 1], &wa[(n << 1) + 1], &wa[n * 3 + 1],
	     &wa[(n << 2) + 1], &wa[n * 5 + 1]);
    if (info == 5) {
	info = 4;
    }
/* L20: */
    return info;

/*     last card of subroutine hybrj1. */

} /* hybrj1_ */

