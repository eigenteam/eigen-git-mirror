
template <typename Scalar>
void ei_covar(int n, Scalar *r__, int ldr, 
	const int *ipvt, Scalar tol, Scalar *wa)
{
    /* System generated locals */
    int r_dim1, r_offset, i__1, i__2, i__3;

    /* Local variables */
    int i__, j, k, l, ii, jj, km1;
    int sing;
    Scalar temp, tolr;

    /* Parameter adjustments */
    --wa;
    --ipvt;
    tolr = tol * ei_abs(r__[0]);
    r_dim1 = ldr;
    r_offset = 1 + r_dim1;
    r__ -= r_offset;

    /* Function Body */

/*     form the inverse of r in the full upper triangle of r. */

    l = 0;
    i__1 = n;
    for (k = 1; k <= i__1; ++k) {
	if (ei_abs(r__[k + k * r_dim1]) <= tolr) {
	    goto L50;
	}
	r__[k + k * r_dim1] = 1. / r__[k + k * r_dim1];
	km1 = k - 1;
	if (km1 < 1) {
	    goto L30;
	}
	i__2 = km1;
	for (j = 1; j <= i__2; ++j) {
	    temp = r__[k + k * r_dim1] * r__[j + k * r_dim1];
	    r__[j + k * r_dim1] = 0.;
	    i__3 = j;
	    for (i__ = 1; i__ <= i__3; ++i__) {
		r__[i__ + k * r_dim1] -= temp * r__[i__ + j * r_dim1];
/* L10: */
	    }
/* L20: */
	}
L30:
	l = k;
/* L40: */
    }
L50:

/*     form the full upper triangle of the inverse of (r transpose)*r */
/*     in the full upper triangle of r. */

    if (l < 1) {
	goto L110;
    }
    i__1 = l;
    for (k = 1; k <= i__1; ++k) {
	km1 = k - 1;
	if (km1 < 1) {
	    goto L80;
	}
	i__2 = km1;
	for (j = 1; j <= i__2; ++j) {
	    temp = r__[j + k * r_dim1];
	    i__3 = j;
	    for (i__ = 1; i__ <= i__3; ++i__) {
		r__[i__ + j * r_dim1] += temp * r__[i__ + k * r_dim1];
/* L60: */
	    }
/* L70: */
	}
L80:
	temp = r__[k + k * r_dim1];
	i__2 = k;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    r__[i__ + k * r_dim1] = temp * r__[i__ + k * r_dim1];
/* L90: */
	}
/* L100: */
    }
L110:

/*     form the full lower triangle of the covariance matrix */
/*     in the strict lower triangle of r and in wa. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	jj = ipvt[j];
	sing = j > l;
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    if (sing) {
		r__[i__ + j * r_dim1] = 0.;
	    }
	    ii = ipvt[i__];
	    if (ii > jj) {
		r__[ii + jj * r_dim1] = r__[i__ + j * r_dim1];
	    }
	    if (ii < jj) {
		r__[jj + ii * r_dim1] = r__[i__ + j * r_dim1];
	    }
/* L120: */
	}
	wa[jj] = r__[j + j * r_dim1];
/* L130: */
    }

/*     symmetrize the covariance matrix in r. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    r__[i__ + j * r_dim1] = r__[j + i__ * r_dim1];
/* L140: */
	}
	r__[j + j * r_dim1] = wa[j];
/* L150: */
    }
    /*return 0;*/

/*     last card of subroutine covar. */

} /* covar_ */

