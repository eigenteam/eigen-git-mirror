
template <typename Scalar>
int ei_fdjac1(minpack_func_nn fcn, void *p, int n, Scalar *x, const Scalar *
	fvec, Scalar *fjac, int ldfjac, int ml, 
	int mu, Scalar epsfcn, Scalar *wa1, Scalar *wa2)
{
    /* System generated locals */
    int fjac_dim1, fjac_offset;

    /* Local variables */
    Scalar h__;
    int i, j, k;
    Scalar eps, temp;
    int msum;
    Scalar epsmch;
    int iflag = 0;

    /* Parameter adjustments */
    --wa2;
    --wa1;
    --fvec;
    --x;
    fjac_dim1 = ldfjac;
    fjac_offset = 1 + fjac_dim1 * 1;
    fjac -= fjac_offset;

    /* Function Body */

/*     epsmch is the machine precision. */

    epsmch = epsilon<Scalar>();

    eps = ei_sqrt((std::max(epsfcn,epsmch)));
    msum = ml + mu + 1;
    if (msum < n) {
	goto L40;
    }

/*        computation of dense approximate jacobian. */

    for (j = 1; j <= n; ++j) {
	temp = x[j];
	h__ = eps * ei_abs(temp);
	if (h__ == 0.) {
	    h__ = eps;
	}
	x[j] = temp + h__;
	iflag = (*fcn)(p, n, &x[1], &wa1[1], 1);
	if (iflag < 0) {
	    goto L30;
	}
	x[j] = temp;
	for (i = 1; i <= n; ++i) {
	    fjac[i + j * fjac_dim1] = (wa1[i] - fvec[i]) / h__;
/* L10: */
	}
/* L20: */
    }
L30:
    /* goto L110; */
    return iflag;
L40:

/*        computation of banded approximate jacobian. */

    for (k = 1; k <= msum; ++k) {
	for (j = k; msum< 0 ? j >= n: j <= n; j += msum) {
	    wa2[j] = x[j];
	    h__ = eps * ei_abs(wa2[j]);
	    if (h__ == 0.) {
		h__ = eps;
	    }
	    x[j] = wa2[j] + h__;
/* L60: */
	}
	iflag = (*fcn)(p, n, &x[1], &wa1[1], 1);
	if (iflag < 0) {
	    /* goto L100; */
            return iflag;
	}
	for (j = k; msum< 0 ? j >= n: j <= n; j += msum) {
	    x[j] = wa2[j];
	    h__ = eps * ei_abs(wa2[j]);
	    if (h__ == 0.) {
		h__ = eps;
	    }
	    for (i = 1; i <= n; ++i) {
		fjac[i + j * fjac_dim1] = 0.;
		if (i >= j - mu && i <= j + ml) {
		    fjac[i + j * fjac_dim1] = (wa1[i] - fvec[i]) / h__;
		}
/* L70: */
	    }
/* L80: */
	}
/* L90: */
    }
/* L100: */
/* L110: */
    return iflag;

/*     last card of subroutine fdjac1. */

} /* fdjac1_ */

