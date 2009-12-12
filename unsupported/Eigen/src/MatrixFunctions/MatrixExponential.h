// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_MATRIX_EXPONENTIAL
#define EIGEN_MATRIX_EXPONENTIAL

#ifdef _MSC_VER
  template <typename Scalar> Scalar log2(Scalar v) { return std::log(v)/std::log(Scalar(2)); }
#endif

/** \ingroup MatrixFunctions_Module
 *
 * \brief Compute the matrix exponential. 
 *
 * \param M      matrix whose exponential is to be computed. 
 * \param result pointer to the matrix in which to store the result.
 *
 * The matrix exponential of \f$ M \f$ is defined by
 * \f[ \exp(M) = \sum_{k=0}^\infty \frac{M^k}{k!}. \f]
 * The matrix exponential can be used to solve linear ordinary
 * differential equations: the solution of \f$ y' = My \f$ with the
 * initial condition \f$ y(0) = y_0 \f$ is given by 
 * \f$ y(t) = \exp(M) y_0 \f$.
 *
 * The cost of the computation is approximately \f$ 20 n^3 \f$ for
 * matrices of size \f$ n \f$. The number 20 depends weakly on the
 * norm of the matrix.
 *
 * The matrix exponential is computed using the scaling-and-squaring
 * method combined with Pad&eacute; approximation. The matrix is first
 * rescaled, then the exponential of the reduced matrix is computed
 * approximant, and then the rescaling is undone by repeated
 * squaring. The degree of the Pad&eacute; approximant is chosen such
 * that the approximation error is less than the round-off
 * error. However, errors may accumulate during the squaring phase.
 * 
 * Details of the algorithm can be found in: Nicholas J. Higham, "The
 * scaling and squaring method for the matrix exponential revisited,"
 * <em>SIAM J. %Matrix Anal. Applic.</em>, <b>26</b>:1179&ndash;1193,
 * 2005. 
 *
 * Example: The following program checks that
 * \f[ \exp \left[ \begin{array}{ccc} 
 *       0 & \frac14\pi & 0 \\ 
 *       -\frac14\pi & 0 & 0 \\
 *       0 & 0 & 0 
 *     \end{array} \right] = \left[ \begin{array}{ccc}
 *       \frac12\sqrt2 & -\frac12\sqrt2 & 0 \\
 *       \frac12\sqrt2 & \frac12\sqrt2 & 0 \\
 *       0 & 0 & 1
 *     \end{array} \right]. \f]
 * This corresponds to a rotation of \f$ \frac14\pi \f$ radians around
 * the z-axis.

 * \include MatrixExponential.cpp
 * Output: \verbinclude MatrixExponential.out
 *
 * \note \p M has to be a matrix of \c float, \c double, 
 * \c complex<float> or \c complex<double> .
 */
template <typename Derived>
EIGEN_STRONG_INLINE void ei_matrix_exponential(const MatrixBase<Derived> &M, 
					       typename MatrixBase<Derived>::PlainMatrixType* result);

/** \ingroup MatrixFunctions_Module
  * \brief Class for computing the matrix exponential.
  */
template <typename MatrixType>
class MatrixExponential {

  public:
  
    /** \brief Compute the matrix exponential. 
     *
     * \param M      matrix whose exponential is to be computed. 
     * \param result pointer to the matrix in which to store the result.
     */
    MatrixExponential(const MatrixType &M, MatrixType *result);  

  private:

    // Prevent copying
    MatrixExponential(const MatrixExponential&);
    MatrixExponential& operator=(const MatrixExponential&);

    /** \brief Compute the (3,3)-Pad&eacute; approximant to the exponential.
     *  
     *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
     *  approximant of \f$ \exp(A) \f$ around \f$ A = 0 \f$.
     *
     *  \param A   Argument of matrix exponential
     */
    void pade3(const MatrixType &A);

    /** \brief Compute the (5,5)-Pad&eacute; approximant to the exponential.
     *  
     *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
     *  approximant of \f$ \exp(A) \f$ around \f$ A = 0 \f$.
     *
     *  \param A   Argument of matrix exponential
     */
    void pade5(const MatrixType &A);

    /** \brief Compute the (7,7)-Pad&eacute; approximant to the exponential.
     *  
     *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
     *  approximant of \f$ \exp(A) \f$ around \f$ A = 0 \f$.
     *
     *  \param A   Argument of matrix exponential
     */
    void pade7(const MatrixType &A);

    /** \brief Compute the (9,9)-Pad&eacute; approximant to the exponential.
     *  
     *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
     *  approximant of \f$ \exp(A) \f$ around \f$ A = 0 \f$.
     *
     *  \param A   Argument of matrix exponential
     */
    void pade9(const MatrixType &A);

    /** \brief Compute the (13,13)-Pad&eacute; approximant to the exponential.
     *  
     *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
     *  approximant of \f$ \exp(A) \f$ around \f$ A = 0 \f$.
     *
     *  \param A   Argument of matrix exponential
     */
    void pade13(const MatrixType &A);

    /** \brief Compute Pad&eacute; approximant to the exponential. 
     *  
     * Computes \c m_U, \c m_V and \c m_squarings such that 
     * \f$ (V+U)(V-U)^{-1} \f$ is a Pad&eacute; of 
     * \f$ \exp(2^{-\mbox{squarings}}M) \f$ around \f$ M = 0 \f$. The
     * degree of the Pad&eacute; approximant and the value of
     * squarings are chosen such that the approximation error is no
     * more than the round-off error.
     *
     * The argument of this function should correspond with the (real
     * part of) the entries of \c m_M.  It is used to select the
     * correct implementation using overloading.
     */
    void computeUV(double);

    /** \brief Compute Pad&eacute; approximant to the exponential. 
     *
     *  \sa computeUV(double);
     */
    void computeUV(float);

    typedef typename ei_traits<MatrixType>::Scalar Scalar;
    typedef typename NumTraits<typename ei_traits<MatrixType>::Scalar>::Real RealScalar;

    /** \brief Pointer to matrix whose exponential is to be computed. */
    const MatrixType* m_M; 

    /** \brief Even-degree terms in numerator of Pad&eacute; approximant. */
    MatrixType m_U;

    /** \brief Odd-degree terms in numerator of Pad&eacute; approximant. */
    MatrixType m_V;

    /** \brief Used for temporary storage. */
    MatrixType m_tmp1;

    /** \brief Used for temporary storage. */
    MatrixType m_tmp2;

    /** \brief Identity matrix of the same size as \c m_M. */
    MatrixType m_Id;

    /** \brief Number of squarings required in the last step. */
    int m_squarings;

    /** \brief L1 norm of m_M. */
    float m_l1norm;
};

template <typename MatrixType>
MatrixExponential<MatrixType>::MatrixExponential(const MatrixType &M, MatrixType *result) :
  m_M(&M), 
  m_U(M.rows(),M.cols()), 
  m_V(M.rows(),M.cols()), 
  m_tmp1(M.rows(),M.cols()), 
  m_tmp2(M.rows(),M.cols()), 
  m_Id(MatrixType::Identity(M.rows(), M.cols())), 
  m_squarings(0), 
  m_l1norm(static_cast<float>(M.cwise().abs().colwise().sum().maxCoeff()))
{
  computeUV(RealScalar());
  m_tmp1 = m_U + m_V;	// numerator of Pade approximant
  m_tmp2 = -m_U + m_V;	// denominator of Pade approximant
  *result = m_tmp2.partialPivLu().solve(m_tmp1);
  for (int i=0; i<m_squarings; i++)
    *result *= *result;		// undo scaling by repeated squaring
}

template <typename MatrixType>
EIGEN_STRONG_INLINE void MatrixExponential<MatrixType>::pade3(const MatrixType &A)
{
  const Scalar b[] = {120., 60., 12., 1.};
  m_tmp1.noalias() = A * A;
  m_tmp2 = b[3]*m_tmp1 + b[1]*m_Id;
  m_U.noalias() = A * m_tmp2;
  m_V = b[2]*m_tmp1 + b[0]*m_Id;
}

template <typename MatrixType>
EIGEN_STRONG_INLINE void MatrixExponential<MatrixType>::pade5(const MatrixType &A)
{
  const Scalar b[] = {30240., 15120., 3360., 420., 30., 1.};
  MatrixType A2 = A * A;
  m_tmp1.noalias() = A2 * A2;
  m_tmp2 = b[5]*m_tmp1 + b[3]*A2 + b[1]*m_Id;
  m_U.noalias() = A * m_tmp2;
  m_V = b[4]*m_tmp1 + b[2]*A2 + b[0]*m_Id;
}

template <typename MatrixType>
EIGEN_STRONG_INLINE void MatrixExponential<MatrixType>::pade7(const MatrixType &A)
{
  const Scalar b[] = {17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.};
  MatrixType A2 = A * A;
  MatrixType A4 = A2 * A2;
  m_tmp1.noalias() = A4 * A2;
  m_tmp2 = b[7]*m_tmp1 + b[5]*A4 + b[3]*A2 + b[1]*m_Id;
  m_U.noalias() = A * m_tmp2;
  m_V = b[6]*m_tmp1 + b[4]*A4 + b[2]*A2 + b[0]*m_Id;
}

template <typename MatrixType>
EIGEN_STRONG_INLINE void MatrixExponential<MatrixType>::pade9(const MatrixType &A)
{
  const Scalar b[] = {17643225600., 8821612800., 2075673600., 302702400., 30270240.,
  		      2162160., 110880., 3960., 90., 1.};
  MatrixType A2 = A * A;
  MatrixType A4 = A2 * A2;
  MatrixType A6 = A4 * A2;
  m_tmp1.noalias() = A6 * A2;
  m_tmp2 = b[9]*m_tmp1 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*m_Id;
  m_U.noalias() = A * m_tmp2;
  m_V = b[8]*m_tmp1 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*m_Id;
}

template <typename MatrixType>
EIGEN_STRONG_INLINE void MatrixExponential<MatrixType>::pade13(const MatrixType &A)
{
  const Scalar b[] = {64764752532480000., 32382376266240000., 7771770303897600., 
  		      1187353796428800., 129060195264000., 10559470521600., 670442572800., 
  		      33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.};
  MatrixType A2 = A * A;
  MatrixType A4 = A2 * A2;
  m_tmp1.noalias() = A4 * A2;
  m_V = b[13]*m_tmp1 + b[11]*A4 + b[9]*A2; // used for temporary storage
  m_tmp2.noalias() = m_tmp1 * m_V;
  m_tmp2 += b[7]*m_tmp1 + b[5]*A4 + b[3]*A2 + b[1]*m_Id;
  m_U.noalias() = A * m_tmp2;
  m_tmp2 = b[12]*m_tmp1 + b[10]*A4 + b[8]*A2;
  m_V.noalias() = m_tmp1 * m_tmp2;
  m_V += b[6]*m_tmp1 + b[4]*A4 + b[2]*A2 + b[0]*m_Id;
}

template <typename MatrixType>
void MatrixExponential<MatrixType>::computeUV(float)
{
  if (m_l1norm < 4.258730016922831e-001) {
    pade3(*m_M);
  } else if (m_l1norm < 1.880152677804762e+000) {
    pade5(*m_M);
  } else {
    const float maxnorm = 3.925724783138660f;
    m_squarings = std::max(0, (int)ceil(log2(m_l1norm / maxnorm)));
    MatrixType A = *m_M / std::pow(Scalar(2), Scalar(static_cast<RealScalar>(m_squarings)));
    pade7(A);
  }
}

template <typename MatrixType>
void MatrixExponential<MatrixType>::computeUV(double)
{
  if (m_l1norm < 1.495585217958292e-002) {
    pade3(*m_M);
  } else if (m_l1norm < 2.539398330063230e-001) {
    pade5(*m_M);
  } else if (m_l1norm < 9.504178996162932e-001) {
    pade7(*m_M);
  } else if (m_l1norm < 2.097847961257068e+000) {
    pade9(*m_M);
  } else {
    const double maxnorm = 5.371920351148152;
    m_squarings = std::max(0, (int)ceil(log2(m_l1norm / maxnorm)));
    MatrixType A = *m_M / std::pow(Scalar(2), Scalar(m_squarings));
    pade13(A);
  }
}

template <typename Derived>
EIGEN_STRONG_INLINE void ei_matrix_exponential(const MatrixBase<Derived> &M, 
					       typename MatrixBase<Derived>::PlainMatrixType* result)
{
  ei_assert(M.rows() == M.cols());
  MatrixExponential<typename MatrixBase<Derived>::PlainMatrixType>(M, result);
}

#endif // EIGEN_MATRIX_EXPONENTIAL
