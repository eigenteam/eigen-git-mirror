
template <typename Scalar>
void ei_qrfac(int m, int n, Scalar *a, int
	lda, int pivot, int *ipvt, int /* lipvt */, Scalar *rdiag,
	 Scalar *acnorm, Scalar *wa)
{
    /* System generated locals */
    int a_dim1, a_offset;

    /* Local variables */
    int i, j, k, jp1;
    Scalar sum;
    int kmax;
    Scalar temp;
    int minmn;
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
    const Scalar epsmch = epsilon<Scalar>();

/*     compute the initial column norms and initialize several arrays. */

    for (j = 1; j <= n; ++j) {
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
    for (j = 1; j <= minmn; ++j) {
	if (! (pivot)) {
	    goto L40;
	}

/*        bring the column of largest norm into the pivot position. */

	kmax = j;
	for (k = j; k <= n; ++k) {
	    if (rdiag[k] > rdiag[kmax]) {
		kmax = k;
	    }
/* L20: */
	}
	if (kmax == j) {
	    goto L40;
	}
	for (i = 1; i <= m; ++i) {
	    temp = a[i + j * a_dim1];
	    a[i + j * a_dim1] = a[i + kmax * a_dim1];
	    a[i + kmax * a_dim1] = temp;
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

	ajnorm = Map< Matrix< Scalar, Dynamic, 1 > >(&a[j + j * a_dim1],m-j+1).blueNorm();
	if (ajnorm == 0.) {
	    goto L100;
	}
	if (a[j + j * a_dim1] < 0.) {
	    ajnorm = -ajnorm;
	}
	for (i = j; i <= m; ++i) {
	    a[i + j * a_dim1] /= ajnorm;
/* L50: */
	}
	a[j + j * a_dim1] += 1.;

/*        apply the transformation to the remaining columns */
/*        and update the norms. */

	jp1 = j + 1;
	if (n < jp1) {
	    goto L100;
	}
	for (k = jp1; k <= n; ++k) {
	    sum = 0.;
	    for (i = j; i <= m; ++i) {
		sum += a[i + j * a_dim1] * a[i + k * a_dim1];
/* L60: */
	    }
	    temp = sum / a[j + j * a_dim1];
	    for (i = j; i <= m; ++i) {
		a[i + k * a_dim1] -= temp * a[i + j * a_dim1];
/* L70: */
	    }
	    if (! (pivot) || rdiag[k] == 0.) {
		goto L80;
	    }
	    temp = a[j + k * a_dim1] / rdiag[k];
/* Computing MAX */
/* Computing 2nd power */
	    rdiag[k] *= ei_sqrt((std::max(Scalar(0.), Scalar(1.)-ei_abs2(temp))));
/* Computing 2nd power */
	    if (Scalar(.05) * ei_abs2(rdiag[k] / wa[k]) > epsmch) {
		goto L80;
	    }
	    rdiag[k] = Map< Matrix< Scalar, Dynamic, 1 > >(&a[jp1 + k * a_dim1],m-j).blueNorm();
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

