
template <typename Scalar>
void ei_r1updt(Matrix< Scalar, Dynamic, Dynamic > &s, const Scalar *u,
        std::vector<PlanarRotation<Scalar> > &v_givens,
        std::vector<PlanarRotation<Scalar> > &w_givens,
        Scalar *v, Scalar *w, bool *sing)
{
    /* Local variables */
    const int m = s.rows();
    const int n = s.cols();
    int i, j=1;
    Scalar temp;
    PlanarRotation<Scalar> givens;

    // ei_r1updt had a broader usecase, but we dont use it here. And, more
    // importantly, we can not test it.
    assert(m==n);

    /* Parameter adjustments */
    --w;
    --u;
    --v;

    /*     move the nontrivial part of the last column of s into w. */
    w[n] = s(n-1,n-1);

    /*     rotate the vector v into a multiple of the n-th unit vector */
    /*     in such a way that a spike is introduced into w. */
    for (j=n-1; j>=1; --j) {
        w[j] = 0.;
        if (v[j] != 0.) {
            /*        determine a givens rotation which eliminates the */
            /*        j-th element of v. */
            givens.makeGivens(-v[n], v[j]);

            /*        apply the transformation to v and store the information */
            /*        necessary to recover the givens rotation. */
            v[n] = givens.s() * v[j] + givens.c() * v[n];
            v_givens[j-1] = givens;

            /*        apply the transformation to s and extend the spike in w. */
            for (i = j; i <= m; ++i) {
                temp = givens.c() * s(j-1,i-1) - givens.s() * w[i];
                w[i] = givens.s() * s(j-1,i-1) + givens.c() * w[i];
                s(j-1,i-1) = temp;
            }
        }
    }

    /*     add the spike from the rank 1 update to w. */
    for (i = 1; i <= m; ++i)
        w[i] += v[n] * u[i];

    /*     eliminate the spike. */
    *sing = false;
    for (j = 1; j <= n-1; ++j) {
        if (w[j] != 0.) {
            /*        determine a givens rotation which eliminates the */
            /*        j-th element of the spike. */
            givens.makeGivens(-s(j-1,j-1), w[j]);

            /*        apply the transformation to s and reduce the spike in w. */
            for (i = j; i <= m; ++i) {
                temp = givens.c() * s(j-1,i-1) + givens.s() * w[i];
                w[i] = -givens.s() * s(j-1,i-1) + givens.c() * w[i];
                s(j-1,i-1) = temp;
            }

            /*        store the information necessary to recover the */
            /*        givens rotation. */
            w_givens[j-1] = givens;
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

