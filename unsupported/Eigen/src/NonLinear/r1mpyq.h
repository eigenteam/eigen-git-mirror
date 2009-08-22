
template <typename Scalar>
void ei_r1mpyq(int m, int n, Scalar *a, int
	lda, const Scalar *v, const Scalar *w)
{
    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2;
    Scalar d__1, d__2;

    /* Local variables */
    int i__, j, nm1, nmj;
    Scalar cos__, sin__, temp;

    /* Parameter adjustments */
    --w;
    --v;
    a_dim1 = lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;

    /* Function Body */

/*     apply the first set of givens rotations to a. */

    nm1 = n - 1;
    if (nm1 < 1) {
	/* goto L50; */
        return;
    }
    i__1 = nm1;
    for (nmj = 1; nmj <= i__1; ++nmj) {
	j = n - nmj;
	if ((d__1 = v[j], ei_abs(d__1)) > 1.) {
	    cos__ = 1. / v[j];
	}
	if ((d__1 = v[j], ei_abs(d__1)) > 1.) {
/* Computing 2nd power */
	    d__2 = cos__;
	    sin__ = ei_sqrt(1. - d__2 * d__2);
	}
	if ((d__1 = v[j], ei_abs(d__1)) <= 1.) {
	    sin__ = v[j];
	}
	if ((d__1 = v[j], ei_abs(d__1)) <= 1.) {
/* Computing 2nd power */
	    d__2 = sin__;
	    cos__ = ei_sqrt(1. - d__2 * d__2);
	}
	i__2 = m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    temp = cos__ * a[i__ + j * a_dim1] - sin__ * a[i__ + n * a_dim1];
	    a[i__ + n * a_dim1] = sin__ * a[i__ + j * a_dim1] + cos__ * a[
		    i__ + n * a_dim1];
	    a[i__ + j * a_dim1] = temp;
/* L10: */
	}
/* L20: */
    }

/*     apply the second set of givens rotations to a. */

    i__1 = nm1;
    for (j = 1; j <= i__1; ++j) {
	if ((d__1 = w[j], ei_abs(d__1)) > 1.) {
	    cos__ = 1. / w[j];
	}
	if ((d__1 = w[j], ei_abs(d__1)) > 1.) {
/* Computing 2nd power */
	    d__2 = cos__;
	    sin__ = ei_sqrt(1. - d__2 * d__2);
	}
	if ((d__1 = w[j], ei_abs(d__1)) <= 1.) {
	    sin__ = w[j];
	}
	if ((d__1 = w[j], ei_abs(d__1)) <= 1.) {
/* Computing 2nd power */
	    d__2 = sin__;
	    cos__ = ei_sqrt(1. - d__2 * d__2);
	}
	i__2 = m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    temp = cos__ * a[i__ + j * a_dim1] + sin__ * a[i__ + n * a_dim1];
	    a[i__ + n * a_dim1] = -sin__ * a[i__ + j * a_dim1] + cos__ * a[
		    i__ + n * a_dim1];
	    a[i__ + j * a_dim1] = temp;
/* L30: */
	}
/* L40: */
    }
/* L50: */
    return;

/*     last card of subroutine r1mpyq. */

} /* r1mpyq_ */

