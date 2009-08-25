
template <typename Scalar>
void ei_r1updt(int m, int n, Scalar *s, int /* ls */, const Scalar *u, Scalar *v, Scalar *w, bool *sing)
{
    /* Local variables */
    int i, j, l, jj, nm1;
    Scalar tan__;
    int nmj;
    Scalar cos__, sin__, tau, temp, cotan;

    /* Parameter adjustments */
    --w;
    --u;
    --v;
    --s;

    /* Function Body */
    const Scalar giant = std::numeric_limits<Scalar>::max();

/*     initialize the diagonal element pointer. */

    jj = n * ((m << 1) - n + 1) / 2 - (m - n);

/*     move the nontrivial part of the last column of s into w. */

    l = jj;
    for (i = n; i <= m; ++i) {
	w[i] = s[l];
	++l;
/* L10: */
    }

/*     rotate the vector v into a multiple of the n-th unit vector */
/*     in such a way that a spike is introduced into w. */

    nm1 = n - 1;
    if (nm1 < 1) {
	goto L70;
    }
    for (nmj = 1; nmj <= nm1; ++nmj) {
	j = n - nmj;
	jj -= m - j + 1;
	w[j] = 0.;
	if (v[j] == 0.) {
	    goto L50;
	}

/*        determine a givens rotation which eliminates the */
/*        j-th element of v. */

	if (ei_abs(v[n]) >= ei_abs(v[j]))
	    goto L20;
	cotan = v[n] / v[j];
/* Computing 2nd power */
	sin__ = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(cotan));
	cos__ = sin__ * cotan;
	tau = 1.;
	if (ei_abs(cos__) * giant > 1.) {
	    tau = 1. / cos__;
	}
	goto L30;
L20:
	tan__ = v[j] / v[n];
/* Computing 2nd power */
	cos__ = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(tan__));
	sin__ = cos__ * tan__;
	tau = sin__;
L30:

/*        apply the transformation to v and store the information */
/*        necessary to recover the givens rotation. */

	v[n] = sin__ * v[j] + cos__ * v[n];
	v[j] = tau;

/*        apply the transformation to s and extend the spike in w. */

	l = jj;
	for (i = j; i <= m; ++i) {
	    temp = cos__ * s[l] - sin__ * w[i];
	    w[i] = sin__ * s[l] + cos__ * w[i];
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

    for (i = 1; i <= m; ++i) {
	w[i] += v[n] * u[i];
/* L80: */
    }

/*     eliminate the spike. */

    *sing = false;
    if (nm1 < 1) {
	goto L140;
    }
    for (j = 1; j <= nm1; ++j) {
	if (w[j] == 0.) {
	    goto L120;
	}

/*        determine a givens rotation which eliminates the */
/*        j-th element of the spike. */

	if (ei_abs(s[jj]) >= ei_abs(w[j]))
	    goto L90;
	cotan = s[jj] / w[j];
/* Computing 2nd power */
	sin__ = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(cotan));
	cos__ = sin__ * cotan;
	tau = 1.;
	if (ei_abs(cos__) * giant > 1.) {
	    tau = 1. / cos__;
	}
	goto L100;
L90:
	tan__ = w[j] / s[jj];
/* Computing 2nd power */
	cos__ = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(tan__));
	sin__ = cos__ * tan__;
	tau = sin__;
L100:

/*        apply the transformation to s and reduce the spike in w. */

	l = jj;
	for (i = j; i <= m; ++i) {
	    temp = cos__ * s[l] + sin__ * w[i];
	    w[i] = -sin__ * s[l] + cos__ * w[i];
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
    for (i = n; i <= m; ++i) {
	s[l] = w[i];
	++l;
/* L150: */
    }
    if (s[jj] == 0.) {
	*sing = true;
    }
    return;

/*     last card of subroutine r1updt. */

} /* r1updt_ */

