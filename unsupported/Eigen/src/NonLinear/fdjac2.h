
template <typename Scalar>
int ei_fdjac2(minpack_func_mn fcn, void *p, int m, int n, Scalar *x, 
	const Scalar *fvec, Scalar *fjac, int ldfjac,
	Scalar epsfcn, Scalar *wa)
{
    /* System generated locals */
    int fjac_dim1, fjac_offset;

    /* Local variables */
    Scalar h__;
    int i, j;
    Scalar eps, temp, epsmch;
    int iflag;

    /* Parameter adjustments */
    --wa;
    --fvec;
    --x;
    fjac_dim1 = ldfjac;
    fjac_offset = 1 + fjac_dim1 * 1;
    fjac -= fjac_offset;

    /* Function Body */

/*     epsmch is the machine precision. */

    epsmch = epsilon<Scalar>();

    eps = ei_sqrt((std::max(epsfcn,epsmch)));
    for (j = 1; j <= n; ++j) {
	temp = x[j];
	h__ = eps * ei_abs(temp);
	if (h__ == 0.) {
	    h__ = eps;
	}
	x[j] = temp + h__;
	iflag = (*fcn)(p, m, n, &x[1], &wa[1], 1);
	if (iflag < 0) {
	    /* goto L30; */
            return iflag;
	}
	x[j] = temp;
	for (i = 1; i <= m; ++i) {
	    fjac[i + j * fjac_dim1] = (wa[i] - fvec[i]) / h__;
/* L10: */
	}
/* L20: */
    }
/* L30: */
    return iflag;

/*     last card of subroutine fdjac2. */

} /* fdjac2_ */

