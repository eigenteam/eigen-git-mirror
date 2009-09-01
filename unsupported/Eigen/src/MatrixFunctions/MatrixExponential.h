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

/** \brief Compute the matrix exponential. 
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
 * \note \p M has to be a matrix of \c float, \c double, 
 * \c complex<float> or \c complex<double> .
 */
template <typename Derived>
EIGEN_STRONG_INLINE void ei_matrix_exponential(const MatrixBase<Derived> &M, 
					       typename MatrixBase<Derived>::PlainMatrixType* result);


/** \internal \brief Internal helper functions for computing the
 *  matrix exponential.
 */
namespace MatrixExponentialInternal {

#ifdef _MSC_VER
  template <typename Scalar> Scalar log2(Scalar v) { return std::log(v)/std::log(Scalar(2)); }
#endif

  /** \internal \brief Compute the (3,3)-Pad&eacute; approximant to
   *  the exponential.
   *  
   *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
   *  approximant of \f$ \exp(M) \f$ around \f$ M = 0 \f$.
   *
   *  \param M   Argument of matrix exponential
   *  \param Id  Identity matrix of same size as M
   *  \param tmp Temporary storage, to be provided by the caller
   *  \param M2  Temporary storage, to be provided by the caller
   *  \param U   Even-degree terms in numerator of Pad&eacute; approximant
   *  \param V   Odd-degree terms in numerator of Pad&eacute; approximant
   */
  template <typename MatrixType>
  EIGEN_STRONG_INLINE void pade3(const MatrixType &M, const MatrixType& Id, MatrixType& tmp, 
				 MatrixType& M2, MatrixType& U, MatrixType& V)
  {
    typedef typename ei_traits<MatrixType>::Scalar Scalar;
    const Scalar b[] = {120., 60., 12., 1.};
    M2.noalias() = M * M;
    tmp = b[3]*M2 + b[1]*Id;
    U.noalias() = M * tmp;
    V = b[2]*M2 + b[0]*Id;
  }
  
  /** \internal \brief Compute the (5,5)-Pad&eacute; approximant to
   *  the exponential.
   *  
   *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
   *  approximant of \f$ \exp(M) \f$ around \f$ M = 0 \f$.
   *
   *  \param M   Argument of matrix exponential
   *  \param Id  Identity matrix of same size as M
   *  \param tmp Temporary storage, to be provided by the caller
   *  \param M2  Temporary storage, to be provided by the caller
   *  \param U   Even-degree terms in numerator of Pad&eacute; approximant
   *  \param V   Odd-degree terms in numerator of Pad&eacute; approximant
   */
  template <typename MatrixType>
  EIGEN_STRONG_INLINE void pade5(const MatrixType &M, const MatrixType& Id, MatrixType& tmp, 
				 MatrixType& M2, MatrixType& U, MatrixType& V)
  {
    typedef typename ei_traits<MatrixType>::Scalar Scalar;
    const Scalar b[] = {30240., 15120., 3360., 420., 30., 1.};
    M2.noalias() = M * M;
    MatrixType M4 = M2 * M2;
    tmp = b[5]*M4 + b[3]*M2 + b[1]*Id;
    U.noalias() = M * tmp;
    V = b[4]*M4 + b[2]*M2 + b[0]*Id;
  }
  
  /** \internal \brief Compute the (7,7)-Pad&eacute; approximant to
   *  the exponential.
   *  
   *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
   *  approximant of \f$ \exp(M) \f$ around \f$ M = 0 \f$.
   *
   *  \param M   Argument of matrix exponential
   *  \param Id  Identity matrix of same size as M
   *  \param tmp Temporary storage, to be provided by the caller
   *  \param M2  Temporary storage, to be provided by the caller
   *  \param U   Even-degree terms in numerator of Pad&eacute; approximant
   *  \param V   Odd-degree terms in numerator of Pad&eacute; approximant
   */
  template <typename MatrixType>
  EIGEN_STRONG_INLINE void pade7(const MatrixType &M, const MatrixType& Id, MatrixType& tmp, 
				 MatrixType& M2, MatrixType& U, MatrixType& V)
  {
    typedef typename ei_traits<MatrixType>::Scalar Scalar;
    const Scalar b[] = {17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.};
    M2.noalias() = M * M;
    MatrixType M4 = M2 * M2;
    MatrixType M6 = M4 * M2;
    tmp = b[7]*M6 + b[5]*M4 + b[3]*M2 + b[1]*Id;
    U.noalias() = M * tmp;
    V = b[6]*M6 + b[4]*M4 + b[2]*M2 + b[0]*Id;
  }
  
  /** \internal \brief Compute the (9,9)-Pad&eacute; approximant to
   *  the exponential.
   *  
   *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
   *  approximant of \f$ \exp(M) \f$ around \f$ M = 0 \f$.
   *
   *  \param M   Argument of matrix exponential
   *  \param Id  Identity matrix of same size as M
   *  \param tmp Temporary storage, to be provided by the caller
   *  \param M2  Temporary storage, to be provided by the caller
   *  \param U   Even-degree terms in numerator of Pad&eacute; approximant
   *  \param V   Odd-degree terms in numerator of Pad&eacute; approximant
   */
  template <typename MatrixType>
  EIGEN_STRONG_INLINE void pade9(const MatrixType &M, const MatrixType& Id, MatrixType& tmp, 
				 MatrixType& M2, MatrixType& U, MatrixType& V)
  {
    typedef typename ei_traits<MatrixType>::Scalar Scalar;
    const Scalar b[] = {17643225600., 8821612800., 2075673600., 302702400., 30270240.,
  		      2162160., 110880., 3960., 90., 1.};
    M2.noalias() = M * M;
    MatrixType M4 = M2 * M2;
    MatrixType M6 = M4 * M2;
    MatrixType M8 = M6 * M2;
    tmp = b[9]*M8 + b[7]*M6 + b[5]*M4 + b[3]*M2 + b[1]*Id;
    U.noalias() = M * tmp;
    V = b[8]*M8 + b[6]*M6 + b[4]*M4 + b[2]*M2 + b[0]*Id;
  }
  
  /** \internal \brief Compute the (13,13)-Pad&eacute; approximant to
   *  the exponential.
   *  
   *  After exit, \f$ (V+U)(V-U)^{-1} \f$ is the Pad&eacute;
   *  approximant of \f$ \exp(M) \f$ around \f$ M = 0 \f$.
   *
   *  \param M   Argument of matrix exponential
   *  \param Id  Identity matrix of same size as M
   *  \param tmp Temporary storage, to be provided by the caller
   *  \param M2  Temporary storage, to be provided by the caller
   *  \param U   Even-degree terms in numerator of Pad&eacute; approximant
   *  \param V   Odd-degree terms in numerator of Pad&eacute; approximant
   */
  template <typename MatrixType>
  EIGEN_STRONG_INLINE void pade13(const MatrixType &M, const MatrixType& Id, MatrixType& tmp, 
				  MatrixType& M2, MatrixType& U, MatrixType& V)
  {
    typedef typename ei_traits<MatrixType>::Scalar Scalar;
    const Scalar b[] = {64764752532480000., 32382376266240000., 7771770303897600., 
  		      1187353796428800., 129060195264000., 10559470521600., 670442572800., 
  		      33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.};
    M2.noalias() = M * M;
    MatrixType M4 = M2 * M2;
    MatrixType M6 = M4 * M2;
    V = b[13]*M6 + b[11]*M4 + b[9]*M2;
    tmp.noalias() = M6 * V;
    tmp += b[7]*M6 + b[5]*M4 + b[3]*M2 + b[1]*Id;
    U.noalias() = M * tmp;
    tmp = b[12]*M6 + b[10]*M4 + b[8]*M2;
    V.noalias() = M6 * tmp;
    V += b[6]*M6 + b[4]*M4 + b[2]*M2 + b[0]*Id;
  }
  
  /** \internal \brief Helper class for computing Pad&eacute;
   *  approximants to the exponential.
   */
  template <typename MatrixType, typename RealScalar = typename NumTraits<typename ei_traits<MatrixType>::Scalar>::Real>
  struct computeUV_selector
  {
    /** \internal \brief Compute Pad&eacute; approximant to the exponential. 
     *  
     *  Computes \p U, \p V and \p squarings such that \f$ (V+U)(V-U)^{-1} \f$ 
     *  is a Pad&eacute; of \f$ \exp(2^{-\mbox{squarings}}M) \f$
     *  around \f$ M = 0 \f$. The degree of the Pad&eacute;
     *  approximant and the value of squarings are chosen such that
     *  the approximation error is no more than the round-off error.
     *
     *  \param M         Argument of matrix exponential
     *  \param Id        Identity matrix of same size as M
     *  \param tmp1      Temporary storage, to be provided by the caller
     *  \param tmp2      Temporary storage, to be provided by the caller
     *  \param U         Even-degree terms in numerator of Pad&eacute; approximant
     *  \param V         Odd-degree terms in numerator of Pad&eacute; approximant
     *  \param l1norm    L<sub>1</sub> norm of M
     *  \param squarings Pointer to integer containing number of times
     *                   that the result needs to be squared to find the
     *                   matrix exponential 
     */
    static void run(const MatrixType &M, const MatrixType& Id, MatrixType& tmp1, MatrixType& tmp2, 
		    MatrixType& U, MatrixType& V, float l1norm, int* squarings);
  };
  
  template <typename MatrixType>
  struct computeUV_selector<MatrixType, float>
  {
    static void run(const MatrixType &M, const MatrixType& Id, MatrixType& tmp1, MatrixType& tmp2, 
		    MatrixType& U, MatrixType& V, float l1norm, int* squarings)
    {
      *squarings = 0;
      if (l1norm < 4.258730016922831e-001) {
        pade3(M, Id, tmp1, tmp2, U, V);
      } else if (l1norm < 1.880152677804762e+000) {
        pade5(M, Id, tmp1, tmp2, U, V);
      } else {
        const float maxnorm = 3.925724783138660f;
        *squarings = std::max(0, (int)ceil(log2(l1norm / maxnorm)));
        MatrixType A = M / std::pow(typename ei_traits<MatrixType>::Scalar(2), *squarings);
        pade7(A, Id, tmp1, tmp2, U, V);
      }
    }
  };
  
  template <typename MatrixType>
  struct computeUV_selector<MatrixType, double>
  {
    static void run(const MatrixType &M, const MatrixType& Id, MatrixType& tmp1, MatrixType& tmp2, 
		    MatrixType& U, MatrixType& V, float l1norm, int* squarings)
    {
      *squarings = 0;
      if (l1norm < 1.495585217958292e-002) {
        pade3(M, Id, tmp1, tmp2, U, V);
      } else if (l1norm < 2.539398330063230e-001) {
        pade5(M, Id, tmp1, tmp2, U, V);
      } else if (l1norm < 9.504178996162932e-001) {
        pade7(M, Id, tmp1, tmp2, U, V);
      } else if (l1norm < 2.097847961257068e+000) {
        pade9(M, Id, tmp1, tmp2, U, V);
      } else {
        const double maxnorm = 5.371920351148152;
        *squarings = std::max(0, (int)ceil(log2(l1norm / maxnorm)));
        MatrixType A = M / std::pow(typename ei_traits<MatrixType>::Scalar(2), *squarings);
        pade13(A, Id, tmp1, tmp2, U, V);
      }
    }
  };
  
  /** \internal \brief Compute the matrix exponential. 
   *
   * \param M      matrix whose exponential is to be computed. 
   * \param result pointer to the matrix in which to store the result.
   */
  template <typename MatrixType>
  void compute(const MatrixType &M, MatrixType* result)
  {
    MatrixType num(M.rows(), M.cols());
    MatrixType den(M.rows(), M.cols());
    MatrixType U(M.rows(), M.cols());
    MatrixType V(M.rows(), M.cols());
    MatrixType Id = MatrixType::Identity(M.rows(), M.cols());
    float l1norm = static_cast<float>(M.cwise().abs().colwise().sum().maxCoeff());
    int squarings;
    computeUV_selector<MatrixType>::run(M, Id, num, den, U, V, l1norm, &squarings);
    num = U + V;			// numerator of Pade approximant
    den = -U + V;			// denominator of Pade approximant
    den.partialLu().solve(num, result);
    for (int i=0; i<squarings; i++)
      *result *= *result;		// undo scaling by repeated squaring
  }

} // end of namespace MatrixExponentialInternal

template <typename Derived>
EIGEN_STRONG_INLINE void ei_matrix_exponential(const MatrixBase<Derived> &M, 
					       typename MatrixBase<Derived>::PlainMatrixType* result)
{
  ei_assert(M.rows() == M.cols());
  MatrixExponentialInternal::compute(M.eval(), result);
}

#endif // EIGEN_MATRIX_EXPONENTIAL
