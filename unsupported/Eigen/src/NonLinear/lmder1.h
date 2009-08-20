
template<typename Scalar>
int lmder1_template(minpack_funcder_mn fcn, void *p, int m, int n, Scalar *x, 
	Scalar *fvec, Scalar *fjac, int ldfjac, Scalar tol, 
	int *ipvt, Scalar *wa, int lwa)
{
    /* Initialized data */

    const Scalar factor = 100.;

    /* System generated locals */
    int fjac_dim1, fjac_offset;

    /* Local variables */
    int mode, nfev, njev;
    Scalar ftol, gtol, xtol;
    int maxfev, nprint;
    int info;

    /* Parameter adjustments */
    --fvec;
    --ipvt;
    --x;
    fjac_dim1 = ldfjac;
    fjac_offset = 1 + fjac_dim1 * 1;
    fjac -= fjac_offset;
    --wa;

    /* Function Body */
    info = 0;

/*     check the input parameters for errors. */

    if (n <= 0 || m < n || ldfjac < m || tol < 0. || lwa < n * 5 +
	    m) {
	/* goto L10; */
        printf("lmder1 bad args : m,n,tol,...");
        return info;
    }

/*     call lmder. */

    maxfev = (n + 1) * 100;
    ftol = tol;
    xtol = tol;
    gtol = 0.;
    mode = 1;
    nprint = 0;
    info = lmder(fcn, p, m, n, &x[1], &fvec[1], &fjac[fjac_offset], ldfjac,
	    ftol, xtol, gtol, maxfev, &wa[1], mode, factor, nprint, 
	    &nfev, &njev, &ipvt[1], &wa[n + 1], &wa[(n << 1) + 1], &
	    wa[n * 3 + 1], &wa[(n << 2) + 1], &wa[n * 5 + 1]);
    if (info == 8) {
	info = 4;
    }
/* L10: */
    return info;

/*     last card of subroutine lmder1. */

} /* lmder1_ */

