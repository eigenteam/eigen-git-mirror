
template <typename Scalar>
int ei_fdjac2(minpack_func_mn fcn,
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        Matrix< Scalar, Dynamic, Dynamic > &fjac,
	    Scalar epsfcn,
        Matrix< Scalar, Dynamic, 1 >  &wa)
{
    /* Local variables */
    Scalar h;
    int i, j;
    Scalar eps, temp;
    int iflag;

    /* Function Body */
    const Scalar epsmch = epsilon<Scalar>();
    const int n = x.size();
    const int m = fvec.size();

    eps = ei_sqrt((std::max(epsfcn,epsmch)));
    for (j = 0; j < n; ++j) {
	temp = x[j];
	h = eps * ei_abs(temp);
	if (h == 0.) {
	    h = eps;
	}
	x[j] = temp + h;
	iflag = (*fcn)(m, n, x.data(), wa.data(), 1);
	if (iflag < 0) {
	    /* goto L30; */
            return iflag;
	}
	x[j] = temp;
	for (i = 0; i < m; ++i) {
	    fjac(i,j) = (wa[i] - fvec[i]) / h;
/* L10: */
	}
/* L20: */
    }
/* L30: */
    return iflag;

/*     last card of subroutine fdjac2. */

} /* fdjac2_ */

