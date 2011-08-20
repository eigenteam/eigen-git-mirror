// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Jitse Niesen <jitse@maths.leeds.ac.uk>
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

#ifndef EIGEN_MATRIX_LOGARITHM
#define EIGEN_MATRIX_LOGARITHM

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950L
#endif

/** \ingroup MatrixFunctions_Module
  * \class MatrixLogarithmAtomic
  * \brief Helper class for computing matrix logarithm of atomic matrices.
  *
  * \internal
  * Here, an atomic matrix is a triangular matrix whose diagonal
  * entries are close to each other.
  *
  * \sa class MatrixFunctionAtomic, MatrixBase::log()
  */
template <typename MatrixType>
class MatrixLogarithmAtomic
{
public:

  typedef typename MatrixType::Scalar Scalar;
  // typedef typename MatrixType::Index Index;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  // typedef typename internal::stem_function<Scalar>::type StemFunction;
  // typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  /** \brief Constructor. */
  MatrixLogarithmAtomic() { }

  /** \brief Compute matrix logarithm of atomic matrix
    * \param[in]  A  argument of matrix logarithm, should be upper triangular and atomic
    * \returns  The logarithm of \p A.
    */
  MatrixType compute(const MatrixType& A);

private:

  void compute2x2(const MatrixType& A, MatrixType& result);
  void computeBig(const MatrixType& A, MatrixType& result);
  static Scalar atanh(Scalar x);
  int getPadeDegree(typename MatrixType::RealScalar normTminusI);
  void computePade(MatrixType& result, const MatrixType& T, int degree);
  void computePade3(MatrixType& result, const MatrixType& T);
  void computePade4(MatrixType& result, const MatrixType& T);
  void computePade5(MatrixType& result, const MatrixType& T);
  void computePade6(MatrixType& result, const MatrixType& T);
  void computePade7(MatrixType& result, const MatrixType& T);

  static const double maxNormForPade[];
  static const int minPadeDegree = 3;
  static const int maxPadeDegree = 7;

  // Prevent copying
  MatrixLogarithmAtomic(const MatrixLogarithmAtomic&);
  MatrixLogarithmAtomic& operator=(const MatrixLogarithmAtomic&);
};

template <typename MatrixType>
const double MatrixLogarithmAtomic<MatrixType>::maxNormForPade[] = { 0.0162 /* degree = 3 */, 0.0539, 0.114, 0.187, 0.264 };

/** \brief Compute logarithm of triangular matrix with clustered eigenvalues. */
template <typename MatrixType>
MatrixType MatrixLogarithmAtomic<MatrixType>::compute(const MatrixType& A)
{
  using std::log;
  MatrixType result(A.rows(), A.rows());
  if (A.rows() == 1)
    result(0,0) = log(A(0,0));
  else if (A.rows() == 2)
    compute2x2(A, result);
  else
    computeBig(A, result);
  return result;
}

/** \brief Compute atanh (inverse hyperbolic tangent). */
template <typename MatrixType>
typename MatrixType::Scalar MatrixLogarithmAtomic<MatrixType>::atanh(typename MatrixType::Scalar x)
{
  using std::abs;
  using std::sqrt;
  if (abs(x) > sqrt(NumTraits<Scalar>::epsilon()))
    return Scalar(0.5) * log((Scalar(1) + x) / (Scalar(1) - x));
  else
    return x + x*x*x / Scalar(3);
}

/** \brief Compute logarithm of 2x2 triangular matrix. */
template <typename MatrixType>
void MatrixLogarithmAtomic<MatrixType>::compute2x2(const MatrixType& A, MatrixType& result)
{
  using std::abs;
  using std::ceil;
  using std::imag;
  using std::log;

  Scalar logA00 = log(A(0,0));
  Scalar logA11 = log(A(1,1));

  result(0,0) = logA00;
  result(1,0) = Scalar(0);
  result(1,1) = logA11;

  if (A(0,0) == A(1,1)) {
    result(0,1) = A(0,1) / A(0,0);
  } else if ((abs(A(0,0)) < 0.5*abs(A(1,1))) || (abs(A(0,0)) > 2*abs(A(1,1)))) {
    result(0,1) = A(0,1) * (logA11 - logA00) / (A(1,1) - A(0,0));
  } else {
    // computation in previous branch is inaccurate if A(1,1) \approx A(0,0)
    int unwindingNumber = ceil((imag(logA11 - logA00) - M_PI) / (2*M_PI));
    Scalar z = (A(1,1) - A(0,0)) / (A(1,1) + A(0,0));
    result(0,1) = A(0,1) * (Scalar(2) * atanh(z) + Scalar(0,2*M_PI*unwindingNumber)) / (A(1,1) - A(0,0));
  }
}

/** \brief Compute logarithm of triangular matrices with size > 2. 
  * \details This uses a inverse scale-and-square algorithm. */
template <typename MatrixType>
void MatrixLogarithmAtomic<MatrixType>::computeBig(const MatrixType& A, MatrixType& result)
{
  int numberOfSquareRoots = 0;
  int numberOfExtraSquareRoots = 0;
  int degree;
  MatrixType T = A;

  while (true) {
    RealScalar normTminusI = (T - MatrixType::Identity(T.rows(), T.rows())).cwiseAbs().colwise().sum().maxCoeff();
    if (normTminusI < maxNormForPade[maxPadeDegree - minPadeDegree]) {
      degree = getPadeDegree(normTminusI);
      int degree2 = getPadeDegree(normTminusI / RealScalar(2));
      if ((degree - degree2 <= 1) || (numberOfExtraSquareRoots == 1)) 
	break;
      ++numberOfExtraSquareRoots;
    }
    T = T.sqrt();
    ++numberOfSquareRoots;
  }

  computePade(result, T, degree);
  result *= pow(RealScalar(2), numberOfSquareRoots);
}

/* \brief Get suitable degree for Pade approximation. */
template <typename MatrixType>
int MatrixLogarithmAtomic<MatrixType>::getPadeDegree(typename MatrixType::RealScalar normTminusI)
{
  for (int degree = 3; degree <= maxPadeDegree; ++degree) 
    if (normTminusI <= maxNormForPade[degree - minPadeDegree])
      return degree;
  assert(false); // this line should never be reached
}

/* \brief Compute Pade approximation to matrix logarithm */
template <typename MatrixType>
void MatrixLogarithmAtomic<MatrixType>::computePade(MatrixType& result, const MatrixType& T, int degree)
{
  switch (degree) {
    case 3:  computePade3(result, T); break;
    case 4:  computePade4(result, T); break;
    case 5:  computePade5(result, T); break;
    case 6:  computePade6(result, T); break;
    case 7:  computePade7(result, T); break;
    default: assert(false); // should never happen
  }
} 

template <typename MatrixType>
void MatrixLogarithmAtomic<MatrixType>::computePade3(MatrixType& result, const MatrixType& T)
{
  const int degree = 3;
  double nodes[]   = { 0.112701665379258, 0.500000000000000, 0.887298334620742 };
  double weights[] = { 0.277777777777778, 0.444444444444444, 0.277777777777778 };
  MatrixType TminusI = T - MatrixType::Identity(T.rows(), T.rows());
  result.setZero(T.rows(), T.rows());
  for (int k = 0; k < degree; ++k) 
    result += weights[k] * (MatrixType::Identity(T.rows(), T.rows()) + nodes[k] * TminusI)
                           .template triangularView<Upper>().solve(TminusI);
}

template <typename MatrixType>
void MatrixLogarithmAtomic<MatrixType>::computePade4(MatrixType& result, const MatrixType& T)
{
  const int degree = 4;
  double nodes[]   = { 0.069431844202974, 0.330009478207572, 0.669990521792428, 0.930568155797026 };
  double weights[] = { 0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727 };
  MatrixType TminusI = T - MatrixType::Identity(T.rows(), T.rows());
  result.setZero(T.rows(), T.rows());
  for (int k = 0; k < degree; ++k) 
    result += weights[k] * (MatrixType::Identity(T.rows(), T.rows()) + nodes[k] * TminusI)
                           .template triangularView<Upper>().solve(TminusI);
}

template <typename MatrixType>
void MatrixLogarithmAtomic<MatrixType>::computePade5(MatrixType& result, const MatrixType& T)
{
  const int degree = 5;
  double nodes[]   = { 0.046910077030668, 0.230765344947158, 0.500000000000000,
		       0.769234655052841, 0.953089922969332 };
  double weights[] = { 0.118463442528095, 0.239314335249683, 0.284444444444444,
		       0.239314335249683, 0.118463442528094 };
  MatrixType TminusI = T - MatrixType::Identity(T.rows(), T.rows());
  result.setZero(T.rows(), T.rows());
  for (int k = 0; k < degree; ++k) 
    result += weights[k] * (MatrixType::Identity(T.rows(), T.rows()) + nodes[k] * TminusI)
                           .template triangularView<Upper>().solve(TminusI);
}

template <typename MatrixType>
void MatrixLogarithmAtomic<MatrixType>::computePade6(MatrixType& result, const MatrixType& T)
{
  const int degree = 6;
  double nodes[]   = { 0.033765242898424, 0.169395306766868, 0.380690406958402,
		       0.619309593041598, 0.830604693233132, 0.966234757101576 };
  double weights[] = { 0.085662246189585, 0.180380786524069, 0.233956967286345,
		       0.233956967286346, 0.180380786524069, 0.085662246189585 };
  MatrixType TminusI = T - MatrixType::Identity(T.rows(), T.rows());
  result.setZero(T.rows(), T.rows());
  for (int k = 0; k < degree; ++k) 
    result += weights[k] * (MatrixType::Identity(T.rows(), T.rows()) + nodes[k] * TminusI)
                           .template triangularView<Upper>().solve(TminusI);
}

template <typename MatrixType>
void MatrixLogarithmAtomic<MatrixType>::computePade7(MatrixType& result, const MatrixType& T)
{
  const int degree = 7;
  double nodes[]   = { 0.025446043828621, 0.129234407200303, 0.297077424311301, 0.500000000000000,
		       0.702922575688699, 0.870765592799697, 0.974553956171379 };
  double weights[] = { 0.064742483084435, 0.139852695744638, 0.190915025252559, 0.208979591836734,
		       0.190915025252560, 0.139852695744638, 0.064742483084435 };
  MatrixType TminusI = T - MatrixType::Identity(T.rows(), T.rows());
  result.setZero(T.rows(), T.rows());
  for (int k = 0; k < degree; ++k) 
    result += weights[k] * (MatrixType::Identity(T.rows(), T.rows()) + nodes[k] * TminusI)
                           .template triangularView<Upper>().solve(TminusI);
}

/** \ingroup MatrixFunctions_Module
  *
  * \brief Proxy for the matrix logarithm of some matrix (expression).
  *
  * \tparam Derived  Type of the argument to the matrix function.
  *
  * This class holds the argument to the matrix function until it is
  * assigned or evaluated for some other reason (so the argument
  * should not be changed in the meantime). It is the return type of
  * matrixBase::matrixLogarithm() and most of the time this is the
  * only way it is used.
  */
template<typename Derived> class MatrixLogarithmReturnValue
: public ReturnByValue<MatrixLogarithmReturnValue<Derived> >
{
public:

  typedef typename Derived::Scalar Scalar;
  typedef typename Derived::Index Index;

  /** \brief Constructor.
    *
    * \param[in]  A  %Matrix (expression) forming the argument of the matrix logarithm.
    */
  MatrixLogarithmReturnValue(const Derived& A) : m_A(A) { }
  
  /** \brief Compute the matrix logarithm.
    *
    * \param[out]  result  Logarithm of \p A, where \A is as specified in the constructor.
    */
  template <typename ResultType>
  inline void evalTo(ResultType& result) const
  {
    typedef typename Derived::PlainObject PlainObject;
    typedef internal::traits<PlainObject> Traits;
    static const int RowsAtCompileTime = Traits::RowsAtCompileTime;
    static const int ColsAtCompileTime = Traits::ColsAtCompileTime;
    static const int Options = PlainObject::Options;
    typedef std::complex<typename NumTraits<Scalar>::Real> ComplexScalar;
    typedef Matrix<ComplexScalar, Dynamic, Dynamic, Options, RowsAtCompileTime, ColsAtCompileTime> DynMatrixType;
    typedef MatrixLogarithmAtomic<DynMatrixType> AtomicType;
    AtomicType atomic;
    
    const PlainObject Aevaluated = m_A.eval();
    MatrixFunction<PlainObject, AtomicType> mf(Aevaluated, atomic);
    mf.compute(result);
  }

  Index rows() const { return m_A.rows(); }
  Index cols() const { return m_A.cols(); }
  
private:
  typename internal::nested<Derived>::type m_A;
  
  MatrixLogarithmReturnValue& operator=(const MatrixLogarithmReturnValue&);
};

namespace internal {
  template<typename Derived>
  struct traits<MatrixLogarithmReturnValue<Derived> >
  {
    typedef typename Derived::PlainObject ReturnType;
  };
}


/********** MatrixBase method **********/


template <typename Derived>
const MatrixLogarithmReturnValue<Derived> MatrixBase<Derived>::log() const
{
  eigen_assert(rows() == cols());
  return MatrixLogarithmReturnValue<Derived>(derived());
}

#endif // EIGEN_MATRIX_LOGARITHM
