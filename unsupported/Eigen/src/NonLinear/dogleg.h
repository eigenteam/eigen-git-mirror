
template <typename Scalar>
void ei_dogleg(int n, const Scalar *r__, int /* lr*/ , 
	const Scalar *diag, const Scalar *qtb, Scalar delta, Scalar *x, 
	Scalar *wa1, Scalar *wa2)
{
    /* System generated locals */
    int i__1, i__2;
    Scalar d__1, d__2, d__3, d__4;

    /* Local variables */
    int i__, j, k, l, jj, jp1;
    Scalar sum, temp, alpha, bnorm;
    Scalar gnorm, qnorm, epsmch;
    Scalar sgnorm;

    /* Parameter adjustments */
    --wa2;
    --wa1;
    --x;
    --qtb;
    --diag;
    --r__;

    /* Function Body */

/*     epsmch is the machine precision. */

    epsmch = epsilon<Scalar>();

/*     first, calculate the gauss-newton direction. */

    jj = n * (n + 1) / 2 + 1;
    i__1 = n;
    for (k = 1; k <= i__1; ++k) {
	j = n - k + 1;
	jp1 = j + 1;
	jj -= k;
	l = jj + 1;
	sum = 0.;
	if (n < jp1) {
	    goto L20;
	}
	i__2 = n;
	for (i__ = jp1; i__ <= i__2; ++i__) {
	    sum += r__[l] * x[i__];
	    ++l;
/* L10: */
	}
L20:
	temp = r__[jj];
	if (temp != 0.) {
	    goto L40;
	}
	l = j;
	i__2 = j;
	for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
	    d__2 = temp, d__3 = fabs(r__[l]);
	    temp = std::max(d__2,d__3);
	    l = l + n - i__;
/* L30: */
	}
	temp = epsmch * temp;
	if (temp == 0.) {
	    temp = epsmch;
	}
L40:
	x[j] = (qtb[j] - sum) / temp;
/* L50: */
    }

/*     test whether the gauss-newton direction is acceptable. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	wa1[j] = 0.;
	wa2[j] = diag[j] * x[j];
/* L60: */
    }
    qnorm = Map< Matrix< Scalar, Dynamic, 1 > >(&wa2[1],n).stableNorm();
    if (qnorm <= delta) {
	/* goto L140; */
        return;
    }

/*     the gauss-newton direction is not acceptable. */
/*     next, calculate the scaled gradient direction. */

    l = 1;
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	temp = qtb[j];
	i__2 = n;
	for (i__ = j; i__ <= i__2; ++i__) {
	    wa1[i__] += r__[l] * temp;
	    ++l;
/* L70: */
	}
	wa1[j] /= diag[j];
/* L80: */
    }

/*     calculate the norm of the scaled gradient and test for */
/*     the special case in which the scaled gradient is zero. */

    gnorm = Map< Matrix< Scalar, Dynamic, 1 > >(&wa1[1],n).stableNorm();
    sgnorm = 0.;
    alpha = delta / qnorm;
    if (gnorm == 0.) {
	goto L120;
    }

/*     calculate the point along the scaled gradient */
/*     at which the quadratic is minimized. */

    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	wa1[j] = wa1[j] / gnorm / diag[j];
/* L90: */
    }
    l = 1;
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	sum = 0.;
	i__2 = n;
	for (i__ = j; i__ <= i__2; ++i__) {
	    sum += r__[l] * wa1[i__];
	    ++l;
/* L100: */
	}
	wa2[j] = sum;
/* L110: */
    }
    temp = Map< Matrix< Scalar, Dynamic, 1 > >(&wa2[1],n).stableNorm();
    sgnorm = gnorm / temp / temp;

/*     test whether the scaled gradient direction is acceptable. */

    alpha = 0.;
    if (sgnorm >= delta) {
	goto L120;
    }

/*     the scaled gradient direction is not acceptable. */
/*     finally, calculate the point along the dogleg */
/*     at which the quadratic is minimized. */

    bnorm = Map< Matrix< Scalar, Dynamic, 1 > >(&qtb[1],n).stableNorm();
    temp = bnorm / gnorm * (bnorm / qnorm) * (sgnorm / delta);
/* Computing 2nd power */
    d__1 = sgnorm / delta;
/* Computing 2nd power */
    d__2 = temp - delta / qnorm;
/* Computing 2nd power */
    d__3 = delta / qnorm;
/* Computing 2nd power */
    d__4 = sgnorm / delta;
    temp = temp - delta / qnorm * (d__1 * d__1) + sqrt(d__2 * d__2 + (1. - 
	    d__3 * d__3) * (1. - d__4 * d__4));
/* Computing 2nd power */
    d__1 = sgnorm / delta;
    alpha = delta / qnorm * (1. - d__1 * d__1) / temp;
L120:

/*     form appropriate convex combination of the gauss-newton */
/*     direction and the scaled gradient direction. */

    temp = (1. - alpha) * std::min(sgnorm,delta);
    i__1 = n;
    for (j = 1; j <= i__1; ++j) {
	x[j] = temp * wa1[j] + alpha * x[j];
/* L130: */
    }
/* L140: */
    return;

/*     last card of subroutine dogleg. */

} /* dogleg_ */

