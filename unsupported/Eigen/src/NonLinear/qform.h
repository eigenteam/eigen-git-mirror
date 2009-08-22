
template <typename Scalar>
void ei_qform(int m, int n, Scalar *q, int
	ldq, Scalar *wa)
{
    /* System generated locals */
    int q_dim1, q_offset, i__1, i__2, i__3;

    /* Local variables */
    int i__, j, k, l, jm1, np1;
    Scalar sum, temp;
    int minmn;

    /* Parameter adjustments */
    --wa;
    q_dim1 = ldq;
    q_offset = 1 + q_dim1 * 1;
    q -= q_offset;

    /* Function Body */

/*     zero out upper triangle of q in the first min(m,n) columns. */

    minmn = std::min(m,n);
    if (minmn < 2) {
	goto L30;
    }
    i__1 = minmn;
    for (j = 2; j <= i__1; ++j) {
	jm1 = j - 1;
	i__2 = jm1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    q[i__ + j * q_dim1] = 0.;
/* L10: */
	}
/* L20: */
    }
L30:

/*     initialize remaining columns to those of the identity matrix. */

    np1 = n + 1;
    if (m < np1) {
	goto L60;
    }
    i__1 = m;
    for (j = np1; j <= i__1; ++j) {
	i__2 = m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    q[i__ + j * q_dim1] = 0.;
/* L40: */
	}
	q[j + j * q_dim1] = 1.;
/* L50: */
    }
L60:

/*     accumulate q from its factored form. */

    i__1 = minmn;
    for (l = 1; l <= i__1; ++l) {
	k = minmn - l + 1;
	i__2 = m;
	for (i__ = k; i__ <= i__2; ++i__) {
	    wa[i__] = q[i__ + k * q_dim1];
	    q[i__ + k * q_dim1] = 0.;
/* L70: */
	}
	q[k + k * q_dim1] = 1.;
	if (wa[k] == 0.) {
	    goto L110;
	}
	i__2 = m;
	for (j = k; j <= i__2; ++j) {
	    sum = 0.;
	    i__3 = m;
	    for (i__ = k; i__ <= i__3; ++i__) {
		sum += q[i__ + j * q_dim1] * wa[i__];
/* L80: */
	    }
	    temp = sum / wa[k];
	    i__3 = m;
	    for (i__ = k; i__ <= i__3; ++i__) {
		q[i__ + j * q_dim1] -= temp * wa[i__];
/* L90: */
	    }
/* L100: */
	}
L110:
/* L120: */
	;
    }
    return;

/*     last card of subroutine qform. */

} /* qform_ */

