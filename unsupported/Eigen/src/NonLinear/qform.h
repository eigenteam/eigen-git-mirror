
template <typename Scalar>
void ei_qform(int m, int n, Scalar *q, int
	ldq, Scalar *wa)
{
    /* System generated locals */
    int q_dim1, q_offset;

    /* Local variables */
    int i, j, k, l, jm1, np1;
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
    for (j = 2; j <= minmn; ++j) {
	jm1 = j - 1;
	for (i = 1; i <= jm1; ++i) {
	    q[i + j * q_dim1] = 0.;
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
    for (j = np1; j <= m; ++j) {
	for (i = 1; i <= m; ++i) {
	    q[i + j * q_dim1] = 0.;
/* L40: */
	}
	q[j + j * q_dim1] = 1.;
/* L50: */
    }
L60:

/*     accumulate q from its factored form. */

    for (l = 1; l <= minmn; ++l) {
	k = minmn - l + 1;
	for (i = k; i <= m; ++i) {
	    wa[i] = q[i + k * q_dim1];
	    q[i + k * q_dim1] = 0.;
/* L70: */
	}
	q[k + k * q_dim1] = 1.;
	if (wa[k] == 0.) {
	    goto L110;
	}
	for (j = k; j <= m; ++j) {
	    sum = 0.;
	    for (i = k; i <= m; ++i) {
		sum += q[i + j * q_dim1] * wa[i];
/* L80: */
	    }
	    temp = sum / wa[k];
	    for (i = k; i <= m; ++i) {
		q[i + j * q_dim1] -= temp * wa[i];
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

