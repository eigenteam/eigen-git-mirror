
template <typename Scalar>
int ei_fdjac1(minpack_func_nn fcn,
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        Matrix< Scalar, Dynamic, Dynamic > &fjac,
        int ml, int mu,
        Scalar epsfcn,
        Matrix< Scalar, Dynamic, 1 >  &wa1,
        Matrix< Scalar, Dynamic, 1 >  &wa2)
{
    /* Local variables */
    Scalar h;
    int i, j, k;
    Scalar eps, temp;
    int msum;
    int iflag = 0;

    /* Function Body */
    const Scalar epsmch = epsilon<Scalar>();
    const int n = x.size();

    eps = ei_sqrt((std::max(epsfcn,epsmch)));
    msum = ml + mu + 1;
    if (msum < n) {
	goto L40;
    }

/*        computation of dense approximate jacobian. */

    for (j = 0; j < n; ++j) {
	temp = x[j];
	h = eps * ei_abs(temp);
	if (h == 0.)
	    h = eps;
	x[j] = temp + h;
	iflag = (*fcn)(n, x.data(), wa1.data(), 1);
	if (iflag < 0)
	    goto L30;
	x[j] = temp;
	for (i = 0; i < n; ++i) {
	    fjac(i,j) = (wa1[i] - fvec[i]) / h;
/* L10: */
	}
/* L20: */
    }
L30:
    /* goto L110; */
    return iflag;
L40:

/*        computation of banded approximate jacobian. */

    for (k = 0; k < msum; ++k) {
	for (j = k; msum< 0 ? j > n: j < n; j += msum) {
	    wa2[j] = x[j];
	    h = eps * ei_abs(wa2[j]);
	    if (h == 0.) h = eps;
	    x[j] = wa2[j] + h;
/* L60: */
	}
	iflag = (*fcn)(n, x.data(), wa1.data(), 1);
	if (iflag < 0) {
	    /* goto L100; */
            return iflag;
	}
	for (j = k; msum< 0 ? j > n: j < n; j += msum) {
	    x[j] = wa2[j];
	    h = eps * ei_abs(wa2[j]);
	    if (h == 0.) h = eps;
	    for (i = 0; i < n; ++i) {
		fjac(i,j) = 0.;
		if (i >= j - mu && i <= j + ml) {
		    fjac(i,j) = (wa1[i] - fvec[i]) / h;
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

