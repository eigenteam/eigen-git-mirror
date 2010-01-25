
template <typename Scalar>
void ei_qform(int m, int n, Scalar *q, int
        ldq, Scalar *wa)
{
    /* System generated locals */
    int q_dim1, q_offset;

    /* Local variables */
    int i, j, k, l;
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
    for (j = 2; j <= minmn; ++j) {
        for (i = 1; i <= j-1; ++i)
            q[i + j * q_dim1] = 0.;
    }

    /*     initialize remaining columns to those of the identity matrix. */

    for (j = n+1; j <= m; ++j) {
        for (i = 1; i <= m; ++i)
            q[i + j * q_dim1] = 0.;
        q[j + j * q_dim1] = 1.;
    }

    /*     accumulate q from its factored form. */

    for (l = 1; l <= minmn; ++l) {
        k = minmn - l + 1;
        for (i = k; i <= m; ++i) {
            wa[i] = q[i + k * q_dim1];
            q[i + k * q_dim1] = 0.;
        }
        q[k + k * q_dim1] = 1.;
        if (wa[k] == 0.)
            continue;
        for (j = k; j <= m; ++j) {
            sum = 0.;
            for (i = k; i <= m; ++i)
                sum += q[i + j * q_dim1] * wa[i];
            temp = sum / wa[k];
            for (i = k; i <= m; ++i)
                q[i + j * q_dim1] -= temp * wa[i];
        }
    }

}

