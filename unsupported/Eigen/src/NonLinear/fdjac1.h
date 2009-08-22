
template <typename Scalar>
int ei_fdjac1(minpack_func_nn fcn, void *p, int n, Scalar *x, const Scalar *
	fvec, Scalar *fjac, int ldfjac, int ml, 
	int mu, Scalar epsfcn, Scalar *wa1, Scalar *wa2)
{
    /* System generated locals */
    int fjac_dim1, fjac_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    Scalar h__;
    int i__, j, k;
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

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
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
	i__2 = n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    fjac[i__ + j * fjac_dim1] = (wa1[i__] - fvec[i__]) / h__;
/* L10: */
	}
/* L20: */
    }
L30:
    /* goto L110; */
    return iflag;
L40:

/*        computation of banded approximate jacobian. */

    i__1 = msum;
    for (k = 1; k <= i__1; ++k) {
	i__2 = n;
	i__3 = msum;
	for (j = k; i__3 < 0 ? j >= i__2 : j <= i__2; j += i__3) {
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
	i__3 = n;
	i__2 = msum;
	for (j = k; i__2 < 0 ? j >= i__3 : j <= i__3; j += i__2) {
	    x[j] = wa2[j];
	    h__ = eps * ei_abs(wa2[j]);
	    if (h__ == 0.) {
		h__ = eps;
	    }
	    i__4 = n;
	    for (i__ = 1; i__ <= i__4; ++i__) {
		fjac[i__ + j * fjac_dim1] = 0.;
		if (i__ >= j - mu && i__ <= j + ml) {
		    fjac[i__ + j * fjac_dim1] = (wa1[i__] - fvec[i__]) / h__;
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

