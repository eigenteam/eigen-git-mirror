
template<typename T>
int hybrd1_template(minpack_func_nn fcn, void *p, int n, T *x, T *
	fvec, T tol, T *wa, int lwa)
{
    /* Initialized data */

    const T factor = 100.;

    /* System generated locals */
    int i__1;

    /* Local variables */
    int j, ml, lr, mu, mode, nfev;
    T xtol;
    int index;
    T epsfcn;
    int maxfev, nprint;
    int info;

    /* Parameter adjustments */
    --fvec;
    --x;
    --wa;

    /* Function Body */
    info = 0;

/*     check the input parameters for errors. */

    if (n <= 0 || tol < 0. || lwa < n * (n * 3 + 13) / 2) {
	/* goto L20; */
        return info;
    }

/*     call hybrd. */

    maxfev = (n + 1) * 200;
    xtol = tol;
    ml = n - 1;
    mu = n - 1;
    epsfcn = 0.;
    mode = 2;
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	wa[j] = 1.;
/* L10: */
    }
    nprint = 0;
    lr = n * (n + 1) / 2;
    index = n * 6 + lr;
    info = hybrd(fcn, p, n, &x[1], &fvec[1], xtol, maxfev, ml, mu, epsfcn, &
	    wa[1], mode, factor, nprint, &nfev, &wa[index + 1], n, &
	    wa[n * 6 + 1], lr, &wa[n + 1], &wa[(n << 1) + 1], &wa[n * 3 
	    + 1], &wa[(n << 2) + 1], &wa[n * 5 + 1]);
    if (info == 5) {
	info = 4;
    }
/* L20: */
    return info;

}

