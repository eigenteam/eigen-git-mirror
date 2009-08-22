
template <typename Scalar>
void ei_qrfac(int m, int n, Scalar *a, int
	lda, int pivot, int *ipvt, int /* lipvt */, Scalar *rdiag,
	 Scalar *acnorm, Scalar *wa)
{
    /* Initialized data */

#define p05 .05

    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2, i__3;
    Scalar d__1, d__2, d__3;

    /* Local variables */
    int i__, j, k, jp1;
    Scalar sum;
    int kmax;
    Scalar temp;
    int minmn;
    Scalar epsmch;
    Scalar ajnorm;

    /* Parameter adjustments */
    --wa;
    --acnorm;
    --rdiag;
    a_dim1 = lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --ipvt;

    /* Function Body */

/*     epsmch is the machine precision. */

    epsmch = epsilon<Scalar>();

/*     compute the initial column norms and initialize several arrays. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	acnorm[j] = Map< Matrix< Scalar, Dynamic, 1 > >(&a[j * a_dim1 + 1],m).blueNorm();
	rdiag[j] = acnorm[j];
	wa[j] = rdiag[j];
	if (pivot) {
	    ipvt[j] = j;
	}
/* L10: */
    }

/*     reduce a to r with householder transformations. */

    minmn = std::min(m,n);
    i__1 = minmn;
    for (j = 1; j <= i__1; ++j) {
	if (! (pivot)) {
	    goto L40;
	}

/*        bring the column of largest norm into the pivot position. */

	kmax = j;
	i__2 = n;
	for (k = j; k <= i__2; ++k) {
	    if (rdiag[k] > rdiag[kmax]) {
		kmax = k;
	    }
/* L20: */
	}
	if (kmax == j) {
	    goto L40;
	}
	i__2 = m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    temp = a[i__ + j * a_dim1];
	    a[i__ + j * a_dim1] = a[i__ + kmax * a_dim1];
	    a[i__ + kmax * a_dim1] = temp;
/* L30: */
	}
	rdiag[kmax] = rdiag[j];
	wa[kmax] = wa[j];
	k = ipvt[j];
	ipvt[j] = ipvt[kmax];
	ipvt[kmax] = k;
L40:

/*        compute the householder transformation to reduce the */
/*        j-th column of a to a multiple of the j-th unit vector. */

	i__2 = m - j + 1;
	ajnorm = Map< Matrix< Scalar, Dynamic, 1 > >(&a[j + j * a_dim1],i__2).blueNorm();
	if (ajnorm == 0.) {
	    goto L100;
	}
	if (a[j + j * a_dim1] < 0.) {
	    ajnorm = -ajnorm;
	}
	i__2 = m;
	for (i__ = j; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] /= ajnorm;
/* L50: */
	}
	a[j + j * a_dim1] += 1.;

/*        apply the transformation to the remaining columns */
/*        and update the norms. */

	jp1 = j + 1;
	if (n < jp1) {
	    goto L100;
	}
	i__2 = n;
	for (k = jp1; k <= i__2; ++k) {
	    sum = 0.;
	    i__3 = m;
	    for (i__ = j; i__ <= i__3; ++i__) {
		sum += a[i__ + j * a_dim1] * a[i__ + k * a_dim1];
/* L60: */
	    }
	    temp = sum / a[j + j * a_dim1];
	    i__3 = m;
	    for (i__ = j; i__ <= i__3; ++i__) {
		a[i__ + k * a_dim1] -= temp * a[i__ + j * a_dim1];
/* L70: */
	    }
	    if (! (pivot) || rdiag[k] == 0.) {
		goto L80;
	    }
	    temp = a[j + k * a_dim1] / rdiag[k];
/* Computing MAX */
/* Computing 2nd power */
	    d__3 = temp;
	    d__1 = 0., d__2 = 1. - d__3 * d__3;
	    rdiag[k] *= ei_sqrt((std::max(d__1,d__2)));
/* Computing 2nd power */
	    d__1 = rdiag[k] / wa[k];
	    if (p05 * (d__1 * d__1) > epsmch) {
		goto L80;
	    }
	    i__3 = m - j;
	    rdiag[k] = Map< Matrix< Scalar, Dynamic, 1 > >(&a[jp1 + k * a_dim1],i__3).blueNorm();
	    wa[k] = rdiag[k];
L80:
/* L90: */
	    ;
	}
L100:
	rdiag[j] = -ajnorm;
/* L110: */
    }
    return;

/*     last card of subroutine qrfac. */

} /* qrfac_ */

