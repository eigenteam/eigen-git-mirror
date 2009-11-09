
template <typename Scalar>
void ei_rwupdt(int n, Scalar *r__, int ldr, 
	const Scalar *w, Scalar *b, Scalar *alpha, Scalar *cos__, 
	Scalar *sin__)
{
    /* System generated locals */
    int r_dim1, r_offset;

    /* Local variables */
    int i, j, jm1;
    Scalar tan__, temp, rowj, cotan;

    /* Parameter adjustments */
    --sin__;
    --cos__;
    --b;
    --w;
    r_dim1 = ldr;
    r_offset = 1 + r_dim1 * 1;
    r__ -= r_offset;

    /* Function Body */

    for (j = 1; j <= n; ++j) {
	rowj = w[j];
	jm1 = j - 1;

/*        apply the previous transformations to */
/*        r(i,j), i=1,2,...,j-1, and to w(j). */

	if (jm1 < 1) {
	    goto L20;
	}
	for (i = 1; i <= jm1; ++i) {
	    temp = cos__[i] * r__[i + j * r_dim1] + sin__[i] * rowj;
	    rowj = -sin__[i] * r__[i + j * r_dim1] + cos__[i] * rowj;
	    r__[i + j * r_dim1] = temp;
/* L10: */
	}
L20:

/*        determine a givens rotation which eliminates w(j). */

	cos__[j] = 1.;
	sin__[j] = 0.;
	if (rowj == 0.) {
	    goto L50;
	}
	if (ei_abs(r__[j + j * r_dim1]) >= ei_abs(rowj))
	    goto L30;
	cotan = r__[j + j * r_dim1] / rowj;
/* Computing 2nd power */
	sin__[j] = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(cotan));
	cos__[j] = sin__[j] * cotan;
	goto L40;
L30:
	tan__ = rowj / r__[j + j * r_dim1];
/* Computing 2nd power */
	cos__[j] = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(tan__));
	sin__[j] = cos__[j] * tan__;
L40:

/*        apply the current transformation to r(j,j), b(j), and alpha. */

	r__[j + j * r_dim1] = cos__[j] * r__[j + j * r_dim1] + sin__[j] * 
		rowj;
	temp = cos__[j] * b[j] + sin__[j] * *alpha;
	*alpha = -sin__[j] * b[j] + cos__[j] * *alpha;
	b[j] = temp;
L50:
/* L60: */
	;
    }
    return;

/*     last card of subroutine rwupdt. */

} /* rwupdt_ */

