
template <typename Scalar>
void ei_r1updt(int m, int n, Scalar *s, int /* ls */, const Scalar *u, Scalar *v, Scalar *w, int *sing)
{
    /* Initialized data */

#define p5 .5
#define p25 .25

    /* System generated locals */
    int i__1, i__2;
    Scalar d__1, d__2;

    /* Local variables */
    int i__, j, l, jj, nm1;
    Scalar tan__;
    int nmj;
    Scalar cos__, sin__, tau, temp, giant, cotan;

    /* Parameter adjustments */
    --w;
    --u;
    --v;
    --s;

    /* Function Body */

/*     giant is the largest magnitude. */

    giant = std::numeric_limits<Scalar>::max();

/*     initialize the diagonal element pointer. */

    jj = n * ((m << 1) - n + 1) / 2 - (m - n);

/*     move the nontrivial part of the last column of s into w. */

    l = jj;
    i__1 = m;
    for (i__ = n; i__ <= i__1; ++i__) {
	w[i__] = s[l];
	++l;
/* L10: */
    }

/*     rotate the vector v into a multiple of the n-th unit vector */
/*     in such a way that a spike is introduced into w. */

    nm1 = n - 1;
    if (nm1 < 1) {
	goto L70;
    }
    i__1 = nm1;
    for (nmj = 1; nmj <= i__1; ++nmj) {
	j = n - nmj;
	jj -= m - j + 1;
	w[j] = 0.;
	if (v[j] == 0.) {
	    goto L50;
	}

/*        determine a givens rotation which eliminates the */
/*        j-th element of v. */

	if ((d__1 = v[n], ei_abs(d__1)) >= (d__2 = v[j], ei_abs(d__2))) {
	    goto L20;
	}
	cotan = v[n] / v[j];
/* Computing 2nd power */
	d__1 = cotan;
	sin__ = p5 / ei_sqrt(p25 + p25 * (d__1 * d__1));
	cos__ = sin__ * cotan;
	tau = 1.;
	if (ei_abs(cos__) * giant > 1.) {
	    tau = 1. / cos__;
	}
	goto L30;
L20:
	tan__ = v[j] / v[n];
/* Computing 2nd power */
	d__1 = tan__;
	cos__ = p5 / ei_sqrt(p25 + p25 * (d__1 * d__1));
	sin__ = cos__ * tan__;
	tau = sin__;
L30:

/*        apply the transformation to v and store the information */
/*        necessary to recover the givens rotation. */

	v[n] = sin__ * v[j] + cos__ * v[n];
	v[j] = tau;

/*        apply the transformation to s and extend the spike in w. */

	l = jj;
	i__2 = m;
	for (i__ = j; i__ <= i__2; ++i__) {
	    temp = cos__ * s[l] - sin__ * w[i__];
	    w[i__] = sin__ * s[l] + cos__ * w[i__];
	    s[l] = temp;
	    ++l;
/* L40: */
	}
L50:
/* L60: */
	;
    }
L70:

/*     add the spike from the rank 1 update to w. */

    i__1 = m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	w[i__] += v[n] * u[i__];
/* L80: */
    }

/*     eliminate the spike. */

    *sing = false;
    if (nm1 < 1) {
	goto L140;
    }
    i__1 = nm1;
    for (j = 1; j <= i__1; ++j) {
	if (w[j] == 0.) {
	    goto L120;
	}

/*        determine a givens rotation which eliminates the */
/*        j-th element of the spike. */

	if ((d__1 = s[jj], ei_abs(d__1)) >= (d__2 = w[j], ei_abs(d__2))) {
	    goto L90;
	}
	cotan = s[jj] / w[j];
/* Computing 2nd power */
	d__1 = cotan;
	sin__ = p5 / ei_sqrt(p25 + p25 * (d__1 * d__1));
	cos__ = sin__ * cotan;
	tau = 1.;
	if (ei_abs(cos__) * giant > 1.) {
	    tau = 1. / cos__;
	}
	goto L100;
L90:
	tan__ = w[j] / s[jj];
/* Computing 2nd power */
	d__1 = tan__;
	cos__ = p5 / ei_sqrt(p25 + p25 * (d__1 * d__1));
	sin__ = cos__ * tan__;
	tau = sin__;
L100:

/*        apply the transformation to s and reduce the spike in w. */

	l = jj;
	i__2 = m;
	for (i__ = j; i__ <= i__2; ++i__) {
	    temp = cos__ * s[l] + sin__ * w[i__];
	    w[i__] = -sin__ * s[l] + cos__ * w[i__];
	    s[l] = temp;
	    ++l;
/* L110: */
	}

/*        store the information necessary to recover the */
/*        givens rotation. */

	w[j] = tau;
L120:

/*        test for zero diagonal elements in the output s. */

	if (s[jj] == 0.) {
	    *sing = true;
	}
	jj += m - j + 1;
/* L130: */
    }
L140:

/*     move w back into the last column of the output s. */

    l = jj;
    i__1 = m;
    for (i__ = n; i__ <= i__1; ++i__) {
	s[l] = w[i__];
	++l;
/* L150: */
    }
    if (s[jj] == 0.) {
	*sing = true;
    }
    return;

/*     last card of subroutine r1updt. */

} /* r1updt_ */

