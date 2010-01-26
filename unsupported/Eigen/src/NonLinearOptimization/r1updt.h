
template <typename Scalar>
void ei_r1updt(int m, int n, Matrix< Scalar, Dynamic, Dynamic > &s, const Scalar *u, Scalar *v, Scalar *w, bool *sing)
{
    /* Local variables */
    int i, j, nm1;
    Scalar tan__;
    int nmj;
    Scalar cos__, sin__, tau, temp, cotan;

    // ei_r1updt had a broader usecase, but we dont use it here. And, more
    // importantly, we can not test it.
    assert(m==n);

    /* Parameter adjustments */
    --w;
    --u;
    --v;

    /* Function Body */
    const Scalar giant = std::numeric_limits<Scalar>::max();

    /*     move the nontrivial part of the last column of s into w. */
    w[n] = s(n-1,n-1);

    /*     rotate the vector v into a multiple of the n-th unit vector */
    /*     in such a way that a spike is introduced into w. */
    nm1 = n - 1;
    if (nm1 >= 1)
        for (nmj = 1; nmj <= nm1; ++nmj) {
            j = n - nmj;
            w[j] = 0.;
            if (v[j] != 0.) {
                /*        determine a givens rotation which eliminates the */
                /*        j-th element of v. */
                if (ei_abs(v[n]) < ei_abs(v[j])) {
                    cotan = v[n] / v[j];
                    /* Computing 2nd power */
                    sin__ = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(cotan));
                    cos__ = sin__ * cotan;
                    tau = 1.;
                    if (ei_abs(cos__) * giant > 1.) {
                        tau = 1. / cos__;
                    }
                } else {
                    tan__ = v[j] / v[n];
                    /* Computing 2nd power */
                    cos__ = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(tan__));
                    sin__ = cos__ * tan__;
                    tau = sin__;
                }

                /*        apply the transformation to v and store the information */
                /*        necessary to recover the givens rotation. */
                v[n] = sin__ * v[j] + cos__ * v[n];
                v[j] = tau;

                /*        apply the transformation to s and extend the spike in w. */
                for (i = j; i <= m; ++i) {
                    temp = cos__ * s(j-1,i-1) - sin__ * w[i];
                    w[i] = sin__ * s(j-1,i-1) + cos__ * w[i];
                    s(j-1,i-1) = temp;
                }
            }
        }

    /*     add the spike from the rank 1 update to w. */
    for (i = 1; i <= m; ++i)
        w[i] += v[n] * u[i];

    /*     eliminate the spike. */
    *sing = false;
    if (nm1 >=  1)
        for (j = 1; j <= nm1; ++j) {
            if (w[j] != 0.) {
                /*        determine a givens rotation which eliminates the */
                /*        j-th element of the spike. */
                if (ei_abs(s(j-1,j-1)) < ei_abs(w[j])) {
                    cotan = s(j-1,j-1) / w[j];
                    /* Computing 2nd power */
                    sin__ = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(cotan));
                    cos__ = sin__ * cotan;
                    tau = 1.;
                    if (ei_abs(cos__) * giant > 1.) {
                        tau = 1. / cos__;
                    }
                } else {
                    tan__ = w[j] / s(j-1,j-1);
                    /* Computing 2nd power */
                    cos__ = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(tan__));
                    sin__ = cos__ * tan__;
                    tau = sin__;
                }

                /*        apply the transformation to s and reduce the spike in w. */
                for (i = j; i <= m; ++i) {
                    temp = cos__ * s(j-1,i-1) + sin__ * w[i];
                    w[i] = -sin__ * s(j-1,i-1) + cos__ * w[i];
                    s(j-1,i-1) = temp;
                }

                /*        store the information necessary to recover the */
                /*        givens rotation. */
                w[j] = tau;
            }

            /*        test for zero diagonal elements in the output s. */
            if (s(j-1,j-1) == 0.) {
                *sing = true;
            }
        }
    /*     move w back into the last column of the output s. */
    s(n-1,n-1) = w[n];

    if (s(j-1,j-1) == 0.) {
        *sing = true;
    }
    return;
}

