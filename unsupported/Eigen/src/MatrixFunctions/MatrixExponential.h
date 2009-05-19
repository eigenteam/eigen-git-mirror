// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
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

/** Compute the matrix exponential. 
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
 * \note Currently, \p M has to be a matrix of \c double .
 */
template <typename Derived>
void ei_matrix_exponential(const MatrixBase<Derived> &M, typename ei_plain_matrix_type<Derived>::type* result)
{
  typedef typename ei_traits<Derived>::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef typename ei_plain_matrix_type<Derived>::type PlainMatrixType;

  ei_assert(M.rows() == M.cols());
  EIGEN_STATIC_ASSERT(NumTraits<Scalar>::HasFloatingPoint,NUMERIC_TYPE_MUST_BE_FLOATING_POINT)

  PlainMatrixType num, den, U, V;
  PlainMatrixType Id = PlainMatrixType::Identity(M.rows(), M.cols());
  typename ei_eval<Derived>::type Meval = M.eval();
  RealScalar l1norm = Meval.cwise().abs().colwise().sum().maxCoeff();
  int squarings = 0;
  
  // Choose degree of Pade approximant, depending on norm of M
  if (l1norm < 1.495585217958292e-002) {
    
    // Use (3,3)-Pade
    const Scalar b[] = {120., 60., 12., 1.};
    PlainMatrixType M2;
    M2 = (Meval * Meval).lazy();
    num = b[3]*M2 + b[1]*Id;
    U = (Meval * num).lazy();
    V = b[2]*M2 + b[0]*Id;

  } else if (l1norm < 2.539398330063230e-001) {

    // Use (5,5)-Pade
    const Scalar b[] = {30240., 15120., 3360., 420., 30., 1.};
    PlainMatrixType M2, M4;
    M2 = (Meval * Meval).lazy();
    M4 = (M2 * M2).lazy();
    num = b[5]*M4 + b[3]*M2 + b[1]*Id;
    U = (Meval * num).lazy();
    V = b[4]*M4 + b[2]*M2 + b[0]*Id;

  } else if (l1norm < 9.504178996162932e-001) {

    // Use (7,7)-Pade
    const Scalar b[] = {17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.};
    PlainMatrixType M2, M4, M6;
    M2 = (Meval * Meval).lazy();
    M4 = (M2 * M2).lazy();
    M6 = (M4 * M2).lazy();
    num = b[7]*M6 + b[5]*M4 + b[3]*M2 + b[1]*Id;
    U = (Meval * num).lazy();
    V = b[6]*M6 + b[4]*M4 + b[2]*M2 + b[0]*Id;

  } else if (l1norm < 2.097847961257068e+000) {

    // Use (9,9)-Pade
    const Scalar b[] = {17643225600., 8821612800., 2075673600., 302702400., 30270240.,
                         2162160., 110880., 3960., 90., 1.};
    PlainMatrixType M2, M4, M6, M8;
    M2 = (Meval * Meval).lazy();
    M4 = (M2 * M2).lazy();
    M6 = (M4 * M2).lazy();
    M8 = (M6 * M2).lazy();
    num = b[9]*M8 + b[7]*M6 + b[5]*M4 + b[3]*M2 + b[1]*Id;
    U = (Meval * num).lazy();
    V = b[8]*M8 + b[6]*M6 + b[4]*M4 + b[2]*M2 + b[0]*Id;

  } else {

    // Use (13,13)-Pade; scale matrix by power of 2 so that its norm
    // is small enough 

    const Scalar maxnorm = 5.371920351148152;
    const Scalar b[] = {64764752532480000., 32382376266240000., 7771770303897600., 
			1187353796428800., 129060195264000., 10559470521600., 670442572800., 
			33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.};
    
    squarings = std::max(0, (int)ceil(log2(l1norm / maxnorm)));
    PlainMatrixType A, A2, A4, A6;
    A = Meval / pow(Scalar(2), squarings);
    A2 = (A * A).lazy();
    A4 = (A2 * A2).lazy();
    A6 = (A4 * A2).lazy();
    num = b[13]*A6 + b[11]*A4 + b[9]*A2;
    V = (A6 * num).lazy();
    num = V + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*Id;
    U = (A * num).lazy();
    num = b[12]*A6 + b[10]*A4 + b[8]*A2;
    V = (A6 * num).lazy() + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*Id;
  }

  num = U + V;			// numerator of Pade approximant
  den = -U + V;			// denominator of Pade approximant
  den.lu().solve(num, result);

  // Undo scaling by repeated squaring
  for (int i=0; i<squarings; i++)
    *result *= *result;
}

#endif // EIGEN_MATRIX_EXPONENTIAL
