
template<typename T>
int lmdif1_template(minpack_func_mn fcn, void *p, int m, int n, T *x, 
	T *fvec, T tol, int *iwa, 
	T *wa, int lwa)
{
    /* Initialized data */

    const T factor = 100.;

    int mp5n, mode, nfev;
    T ftol, gtol, xtol;
    T epsfcn;
    int maxfev, nprint;
    int info;

    /* Parameter adjustments */
    --fvec;
    --iwa;
    --x;
    --wa;

    /* Function Body */
    info = 0;

/*     check the input parameters for errors. */

    if (n <= 0 || m < n || tol < 0. || lwa < m * n + n * 5 + m) {
	/* goto L10; */
        return info;
    }

/*     call lmdif. */

    maxfev = (n + 1) * 200;
    ftol = tol;
    xtol = tol;
    gtol = 0.;
    epsfcn = 0.;
    mode = 1;
    nprint = 0;
    mp5n = m + n * 5;
    info = lmdif(fcn, p, m, n, &x[1], &fvec[1], ftol, xtol, gtol, maxfev,
	    epsfcn, &wa[1], mode, factor, nprint, &nfev, &wa[mp5n + 
	    1], m, &iwa[1], &wa[n + 1], &wa[(n << 1) + 1], &wa[n * 3 + 1], 
	    &wa[(n << 2) + 1], &wa[n * 5 + 1]);
    if (info == 8) {
	info = 4;
    }
/* L10: */
    return info;

/*     last card of subroutine lmdif1. */

} /* lmdif1_ */

