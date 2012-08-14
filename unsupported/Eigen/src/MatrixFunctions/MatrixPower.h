// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_POWER
#define EIGEN_MATRIX_POWER

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279503L
#endif

namespace Eigen {

/**
 * \ingroup MatrixFunctions_Module
 *
 * \brief Class for computing matrix powers.
 *
 * \tparam MatrixType   type of the base, expected to be an instantiation
 * of the Matrix class template.
 * \tparam RealScalar   type of the exponent, a real scalar.
 * \tparam PlainObject  type of the multiplier.
 * \tparam IsInteger    used internally to select correct specialization.
 */
template <typename MatrixType, typename RealScalar, typename PlainObject = MatrixType,
	  int IsInteger = NumTraits<RealScalar>::IsInteger>
class MatrixPower
{
  public:
    /**
     * \brief Constructor.
     *
     * \param[in] A  the base of the matrix power.
     * \param[in] p  the exponent of the matrix power.
     * \param[in] b  the multiplier.
     */
    MatrixPower(const MatrixType& A, const RealScalar& p, const PlainObject& b) :
      m_A(A),
      m_p(p),
      m_b(b),
      m_dimA(A.cols()),
      m_dimb(b.cols())
    { /* empty body */ }

    /**
     * \brief Compute the matrix power.
     *
     * \param[out] result  \f$ A^p b \f$, as specified in the constructor.
     */
    template <typename ResultType> void compute(ResultType& result);
 
  private:
    typedef internal::traits<MatrixType> Traits;
    static const int Rows = Traits::RowsAtCompileTime;
    static const int Cols = Traits::ColsAtCompileTime;
    static const int Options = Traits::Options;
    static const int MaxRows = Traits::MaxRowsAtCompileTime;
    static const int MaxCols = Traits::MaxColsAtCompileTime;

    typedef typename MatrixType::Scalar Scalar;
    typedef std::complex<RealScalar> ComplexScalar;
    typedef typename MatrixType::Index Index;
    typedef Matrix<ComplexScalar, Rows, Cols, Options, MaxRows, MaxCols> ComplexMatrix;
    typedef Array<ComplexScalar, Rows, 1, ColMajor, MaxRows> ComplexArray;

    /**
     * \brief Compute the matrix power.
     *
     * If \c b is \em fatter than \c A, it computes \f$ A^{p_{\textrm int}}
     * \f$ first, and then multiplies it with \c b. Otherwise,
     * #computeChainProduct optimizes the expression.
     *
     * \sa computeChainProduct(ResultType&);
     */
    template <typename ResultType>
    void computeIntPower(ResultType& result);

    /**
     * \brief Convert integral power of the matrix into chain product.
     *
     * This optimizes the matrix expression. It automatically chooses binary
     * powering or matrix chain multiplication or solving the linear system
     * repetitively, according to which algorithm costs less.
     */
    template <typename ResultType>
    void computeChainProduct(ResultType&);

    /** \brief Compute the cost of binary powering. */
    int computeCost(RealScalar);

    /** \brief Solve the linear system repetitively. */
    template <typename ResultType>
    void partialPivLuSolve(RealScalar, ResultType&);

    /** \brief Compute Schur decomposition of #m_A. */
    void computeSchurDecomposition();

    /**
     * \brief Split #m_p into integral part and fractional part.
     *
     * This method stores the integral part \f$ p_{\textrm int} \f$ into
     * #m_pint and the fractional part \f$ p_{\textrm frac} \f$ into
     * #m_pfrac, where #m_pfrac is in the interval \f$ (-1,1) \f$. To
     * choose between the possibilities below, it considers the computation
     * of \f$ A^{p_1} \f$ and \f$ A^{p_2} \f$ and determines which of these
     * computations is the better conditioned.
     */
    void getFractionalExponent();

    /** \brief Compute atanh (inverse hyperbolic tangent). */
    ComplexScalar atanh(const ComplexScalar& x);

    /** \brief Compute power of 2x2 triangular matrix. */
    void compute2x2(const RealScalar& p);

    /**
     * \brief Compute power of triangular matrices with size > 2. 
     * \details This uses a Schur-Pad&eacute; algorithm.
     */
    void computeBig();

    /** \brief Get suitable degree for Pade approximation. (specialized for \c RealScalar = \c double) */
    inline int getPadeDegree(double);
/* TODO
 *  inline int getPadeDegree(float);
 *
 *  inline int getPadeDegree(long double);
 */
    /** \brief Compute Pad&eacute; approximation to matrix fractional power. */
    void computePade(int degree, const ComplexMatrix& IminusT);

    /** \brief Get a certain coefficient of the Pad&eacute; approximation. */
    inline RealScalar coeff(int degree);

    /**
     * \brief Store the fractional power into #m_tmp.
     *
     * This intended for complex matrices.
     */
    void computeTmp(ComplexScalar);

    /**
     * \brief Store the fractional power into #m_tmp.
     *
     * This is intended for real matrices. It takes the real part of
     * \f$ U T^{p_{\textrm frac}} U^H \f$.
     *
     * \sa computeTmp(ComplexScalar);
     */
    void computeTmp(RealScalar);

    const MatrixType& m_A;   ///< \brief Reference to the matrix base.
    const RealScalar& m_p;   ///< \brief Reference to the real exponent.
    const PlainObject& m_b;  ///< \brief Reference to the multiplier.
    const Index m_dimA;      ///< \brief The dimension of #m_A, equivalent to %m_A.cols().
    const Index m_dimb;      ///< \brief The dimension of #m_b, equivalent to %m_b.cols().
    MatrixType m_tmp;        ///< \brief Used for temporary storage.
    RealScalar m_pint;       ///< \brief Integer part of #m_p.
    RealScalar m_pfrac;      ///< \brief Fractional part of #m_p.
    ComplexMatrix m_T;       ///< \brief Triangular part of Schur decomposition.
    ComplexMatrix m_U;       ///< \brief Unitary part of Schur decomposition.
    ComplexMatrix m_fT;      ///< \brief #m_T to the power of #m_pfrac.
    ComplexArray m_logTdiag; ///< \brief Logarithm of the main diagonal of #m_T.
};

/**
 * \internal \ingroup MatrixFunctions_Module
 * \brief Partial specialization for integral exponents.
 */
template <typename MatrixType, typename IntExponent, typename PlainObject>
class MatrixPower<MatrixType, IntExponent, PlainObject, 1>
{
  public:
    /**
     * \brief Constructor.
     *
     * \param[in] A  the base of the matrix power.
     * \param[in] p  the exponent of the matrix power.
     * \param[in] b  the multiplier.
     */
    MatrixPower(const MatrixType& A, const IntExponent& p, const PlainObject& b) :
      m_A(A),
      m_p(p),
      m_b(b),
      m_dimA(A.cols()),
      m_dimb(b.cols())
    { /* empty body */ }

    /**
     * \brief Compute the matrix power.
     *
     * If \c b is \em fatter than \c A, it computes \f$ A^p \f$ first, and
     * then multiplies it with \c b. Otherwise, #computeChainProduct
     * optimizes the expression.
     *
     * \param[out] result  \f$ A^p b \f$, as specified in the constructor.
     *
     * \sa computeChainProduct(ResultType&);
     */
    template <typename ResultType> void compute(ResultType& result);

  private:
    typedef typename MatrixType::Index Index;

    const MatrixType& m_A;  ///< \brief Reference to the matrix base.
    const IntExponent& m_p; ///< \brief Reference to the real exponent.
    const PlainObject& m_b; ///< \brief Reference to the multiplier.
    const Index m_dimA;     ///< \brief The dimension of #m_A, equivalent to %m_A.cols().
    const Index m_dimb;     ///< \brief The dimension of #m_b, equivalent to %m_b.cols().
    MatrixType m_tmp;       ///< \brief Used for temporary storage.

    /**
     * \brief Convert matrix power into chain product.
     *
     * This optimizes the matrix expression. It automatically chooses binary
     * powering or matrix chain multiplication or solving the linear system
     * repetitively, according to which algorithm costs less.
     */
    template <typename ResultType> void computeChainProduct(ResultType& result);

    /** \brief Compute the cost of binary powering. */
    int computeCost(const IntExponent& p);

    /** \brief Solve the linear system repetitively. */
    template <typename ResultType>
    void partialPivLuSolve(IntExponent p, ResultType& result);
};

/**
 * \internal \ingroup MatrixFunctions_Module
 * \brief Partial specialization for complex matrices raised to complex exponents.
 */
template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
class MatrixPower<MatrixType, std::complex<RealScalar>, PlainObject, IsInteger>
{
  private:
    typedef internal::traits<MatrixType> Traits;
    static const int Rows = Traits::RowsAtCompileTime;
    static const int Cols = Traits::ColsAtCompileTime;
    static const int Options = Traits::Options;
    static const int MaxRows = Traits::MaxRowsAtCompileTime;
    static const int MaxCols = Traits::MaxColsAtCompileTime;

    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
    typedef Array<Scalar, Rows, 1, ColMajor, MaxRows> ArrayType;

  public:
    /**
     * \brief Constructor.
     *
     * \param[in] A  the base of the matrix power.
     * \param[in] p  the exponent of the matrix power.
     * \param[in] b  the multiplier.
     */
    MatrixPower(const MatrixType& A, const Scalar& p, const PlainObject& b) :
      m_A(A),
      m_p(p),
      m_b(b),
      m_dimA(A.cols()),
      m_dimb(b.cols())
    { EIGEN_STATIC_ASSERT(false, COMPLEX_POWER_OF_A_MATRIX_IS_UNDER_CONSTRUCTION) }

    /**
     * \brief Compute the matrix power.
     *
     * \param[out] result  \f$ A^p b \f$, as specified in the constructor.
     */
    template <typename ResultType> void compute(ResultType& result);
 
  private:
    /** \brief Compute Schur decomposition of #m_A. */
    void computeSchurDecomposition();

    /** \brief Compute atanh (inverse hyperbolic tangent). */
    Scalar atanh(const Scalar& x);

    /** \brief Compute power of 2x2 triangular matrix. */
    void compute2x2(const Scalar& p);

    /**
     * \brief Compute power of triangular matrices with size > 2. 
     * \details This uses a Schur-Pad&eacute; algorithm.
     */
    void computeBig();

    /** \brief Get suitable degree for Pade approximation. (specialized for \c RealScalar = \c double) */
    inline int getPadeDegree(double);
/* TODO
 *  inline int getPadeDegree(float);
 *
 *  inline int getPadeDegree(long double);
 */
    /** \brief Compute Pad&eacute; approximation to matrix fractional power. */
    void computePade(int degree, const MatrixType& IminusT);

    /** \brief Get a certain coefficient of the Pad&eacute; approximation. */
    inline Scalar coeff(int degree);

    const MatrixType& m_A;  ///< \brief Reference to the matrix base.
    const Scalar& m_p;      ///< \brief Reference to the real exponent.
    const PlainObject& m_b; ///< \brief Reference to the multiplier.
    const Index m_dimA;     ///< \brief The dimension of #m_A, equivalent to %m_A.cols().
    const Index m_dimb;     ///< \brief The dimension of #m_b, equivalent to %m_b.cols().
    MatrixType m_tmp;       ///< \brief Used for temporary storage.
    MatrixType m_T;         ///< \brief Triangular part of Schur decomposition.
    MatrixType m_U;         ///< \brief Unitary part of Schur decomposition.
    MatrixType m_fT;        ///< \brief #m_T to the power of #m_pfrac.
    ArrayType m_logTdiag;   ///< \brief Logarithm of the main diagonal of #m_T.
};

/******* Specialized for real exponents *******/

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
template <typename ResultType>
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::compute(ResultType& result)
{
  using std::floor;
  using std::pow;

  m_pint = floor(m_p);
  m_pfrac = m_p - m_pint;

  if (m_pfrac == RealScalar(0))
    computeIntPower(result);
  else if (m_dimA == 1)
    result = pow(m_A(0,0), m_p) * m_b;
  else {
    computeSchurDecomposition();
    getFractionalExponent();
    computeIntPower(result);
    if (m_dimA == 2)
      compute2x2(m_pfrac);
    else
      computeBig();
    computeTmp(Scalar());
    result *= m_tmp;
  }
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
template <typename ResultType>
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::computeIntPower(ResultType& result)
{
  if (m_dimb > m_dimA) {
    MatrixType tmp = MatrixType::Identity(m_A.rows(), m_A.cols());
    computeChainProduct(tmp);
    result = tmp * m_b;
  } else {
    result = m_b;
    computeChainProduct(result);
  }
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
template <typename ResultType>
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::computeChainProduct(ResultType& result)
{
  using std::frexp;
  using std::ldexp;

  const bool pIsNegative = m_pint < RealScalar(0);
  RealScalar p = pIsNegative? -m_pint: m_pint;
  int cost = computeCost(p);

  if (pIsNegative) {
    if (p * m_dimb <= cost * m_dimA) {
      partialPivLuSolve(p, result);
      return;
    } else {
      m_tmp = m_A.inverse();
    }
  } else {
    m_tmp = m_A;
  }
  while (p * m_dimb > cost * m_dimA) {
    if (fmod(p, RealScalar(2)) >= RealScalar(1)) {
      result = m_tmp * result;
      cost--;
    }
    m_tmp *= m_tmp;
    cost--;
    p = ldexp(p, -1);
  }
  for (; p >= RealScalar(1); p--)
    result = m_tmp * result;
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
int MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::computeCost(RealScalar p)
{
  using std::frexp;
  using std::ldexp;
  int cost, tmp;
  frexp(p, &cost);
  while (frexp(p, &tmp), tmp > 0) {
    p -= ldexp(RealScalar(0.5), tmp);
    cost++;
  }
  return cost;
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
template <typename ResultType>
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::partialPivLuSolve(RealScalar p, ResultType& result)
{
  const PartialPivLU<MatrixType> Asolver(m_A);
  for (; p >= RealScalar(1); p--)
    result = Asolver.solve(result);
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::computeSchurDecomposition()
{
  const ComplexSchur<MatrixType> schurOfA(m_A);
  m_T = schurOfA.matrixT();
  m_U = schurOfA.matrixU();
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::getFractionalExponent()
{
  using std::pow;

  typedef Array<RealScalar, Rows, 1, ColMajor, MaxRows> RealArray;
  const ComplexArray Tdiag = m_T.diagonal();
  RealScalar maxAbsEival, minAbsEival, *begin, *end;
  RealArray absTdiag;

  m_logTdiag = Tdiag.log();
  absTdiag = Tdiag.abs();
  maxAbsEival = minAbsEival = absTdiag[0];
  begin = absTdiag.data();
  end = begin + m_dimA;

  // This avoids traversing the array twice.
  for (RealScalar *ptr = begin + 1; ptr < end; ptr++) {
    if (*ptr > maxAbsEival)
      maxAbsEival = *ptr;
    else if (*ptr < minAbsEival)
      minAbsEival = *ptr;
  }
  if (m_pfrac > RealScalar(0.5) &&  // This is just a shortcut.
      m_pfrac > (RealScalar(1) - m_pfrac) * pow(maxAbsEival/minAbsEival, m_pfrac)) {
    m_pfrac--;
    m_pint++;
  }
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
std::complex<RealScalar> MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::atanh(const ComplexScalar& x)
{
  using std::abs;
  using std::log;
  using std::sqrt;

  if (abs(x) > sqrt(NumTraits<RealScalar>::epsilon()))
    return RealScalar(0.5) * log((RealScalar(1) + x) / (RealScalar(1) - x));
  else
    return x + x*x*x / RealScalar(3);
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::compute2x2(const RealScalar& p)
{
  using std::abs;
  using std::ceil;
  using std::exp;
  using std::imag;
  using std::ldexp;
  using std::log;
  using std::pow;
  using std::sinh;

  int i, j, unwindingNumber;
  ComplexScalar w;

  m_fT(0,0) = pow(m_T(0,0), p);

  for (j = 1; j < m_dimA; j++) {
    i = j - 1;
    m_fT(j,j) = pow(m_T(j,j), p);

    if (m_T(i,i) == m_T(j,j))
      m_fT(i,j) = p * pow(m_T(i,j), p - RealScalar(1));
    else if (abs(m_T(i,i)) < ldexp(abs(m_T(j,j)), -1) || abs(m_T(j,j)) < ldexp(abs(m_T(i,i)), -1))
      m_fT(i,j) = m_T(i,j) * (m_fT(j,j) - m_fT(i,i)) / (m_T(j,j) - m_T(i,i));
    else {
      // computation in previous branch is inaccurate if abs(m_T(j,j)) \approx abs(m_T(i,i))
      unwindingNumber = static_cast<int>(ceil((imag(m_logTdiag[j] - m_logTdiag[i]) - M_PI) / (2 * M_PI)));
      w = atanh((m_T(j,j) - m_T(i,i)) / (m_T(j,j) + m_T(i,i))) + ComplexScalar(0, M_PI * unwindingNumber);
      m_fT(i,j) = m_T(i,j) * RealScalar(2) * exp(RealScalar(0.5) * p * (m_logTdiag[j] + m_logTdiag[i])) *
	  sinh(p * w) / (m_T(j,j) - m_T(i,i));
    }
  }
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::computeBig()
{
  using std::ldexp;

  const RealScalar maxNormForPade = 2.787629930862099e-1;
  int degree, degree2, numberOfSquareRoots = 0, numberOfExtraSquareRoots = 0;
  ComplexMatrix IminusT, sqrtT, T = m_T;
  RealScalar normIminusT;

  while (true) {
    IminusT = ComplexMatrix::Identity(m_A.rows(), m_A.cols()) - T;
    normIminusT = IminusT.cwiseAbs().colwise().sum().maxCoeff();
    if (normIminusT < maxNormForPade) {
      degree = getPadeDegree(normIminusT);
      degree2 = getPadeDegree(normIminusT * RealScalar(0.5));
      if (degree - degree2 <= 1 || numberOfExtraSquareRoots)
	break;
      numberOfExtraSquareRoots++;
    }
    MatrixSquareRootTriangular<ComplexMatrix>(T).compute(sqrtT);
    T = sqrtT;
    numberOfSquareRoots++;
  }
  computePade(degree, IminusT);

  for (; numberOfSquareRoots; numberOfSquareRoots--) {
    compute2x2(ldexp(m_pfrac, -numberOfSquareRoots));
    m_fT *= m_fT;
  }
  compute2x2(m_pfrac);
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
inline int MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::getPadeDegree(double normIminusT)
{
  const double maxNormForPade[] = { 1.882832775783885e-2 /* degree = 3 */ , 6.036100693089764e-2,
      1.239372725584911e-1, 1.998030690604271e-1, 2.787629930862099e-1 };
  for (int degree = 3; degree <= 7; degree++)
    if (normIminusT <= maxNormForPade[degree - 3])
      return degree;
  assert(false); // this line should never be reached
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::computePade(int degree, const ComplexMatrix& IminusT)
{
  degree <<= 1;
  m_fT = coeff(degree) * IminusT;

  for (int i = degree - 1; i; i--) {
    m_fT = (ComplexMatrix::Identity(m_A.rows(), m_A.cols()) + m_fT).template triangularView<Upper>()
	.solve(coeff(i) * IminusT).eval();
  }
  m_fT += ComplexMatrix::Identity(m_A.rows(), m_A.cols());
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
inline RealScalar MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::coeff(int i)
{
  if (i == 1)
    return -m_pfrac;
  else if (i & 1)
    return (-m_pfrac - RealScalar(i)) / RealScalar((i<<2) + 2);
  else
    return (m_pfrac - RealScalar(i)) / RealScalar((i<<2) - 2);
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::computeTmp(RealScalar)
{ m_tmp = (m_U * m_fT * m_U.adjoint()).real(); }

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::computeTmp(ComplexScalar)
{ m_tmp = (m_U * m_fT * m_U.adjoint()).eval(); }

/******* Specialized for integral exponents *******/

template <typename MatrixType, typename IntExponent, typename PlainObject>
template <typename ResultType>
void MatrixPower<MatrixType,IntExponent,PlainObject,1>::compute(ResultType& result)
{
  if (m_dimb > m_dimA) {
    MatrixType tmp = MatrixType::Identity(m_dimA, m_dimA);
    computeChainProduct(tmp);
    result = tmp * m_b;
  } else {
    result = m_b;
    computeChainProduct(result);
  }
}

template <typename MatrixType, typename IntExponent, typename PlainObject>
int MatrixPower<MatrixType,IntExponent,PlainObject,1>::computeCost(const IntExponent& p)
{
  int cost = 0;
  IntExponent tmp = p;
  for (tmp = p >> 1; tmp; tmp >>= 1)
    cost++;
  for (tmp = IntExponent(1); tmp <= p; tmp <<= 1)
    if (tmp & p) cost++;
  return cost;
}

template <typename MatrixType, typename IntExponent, typename PlainObject>
template <typename ResultType>
void MatrixPower<MatrixType,IntExponent,PlainObject,1>::partialPivLuSolve(IntExponent p, ResultType& result)
{
  const PartialPivLU<MatrixType> Asolver(m_A);
  for(; p; p--)
    result = Asolver.solve(result);
}

template <typename MatrixType, typename IntExponent, typename PlainObject>
template <typename ResultType>
void MatrixPower<MatrixType,IntExponent,PlainObject,1>::computeChainProduct(ResultType& result)
{
  const bool pIsNegative = m_p < IntExponent(0);
  IntExponent p = pIsNegative? -m_p: m_p;
  int cost = computeCost(p);

  if (pIsNegative) {
    if (p * m_dimb <= cost * m_dimA) {
      partialPivLuSolve(p, result);
      return;
    } else { m_tmp = m_A.inverse(); }
  } else { m_tmp = m_A; }
 
  while (p * m_dimb > cost * m_dimA) {
    if (p & 1) {
      result = m_tmp * result;
      cost--;
    }
    m_tmp *= m_tmp;
    cost--;
    p >>= 1;
  }

  for (; p; p--)
    result = m_tmp * result;
}

/******* Specialized for complex exponents *******/

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
template <typename ResultType>
void MatrixPower<MatrixType,std::complex<RealScalar>,PlainObject,IsInteger>::compute(ResultType& result)
{
  using std::floor;
  using std::pow;

  if (m_dimA == 1)
    result = pow(m_A(0,0), m_p) * m_b;
  else {
    computeSchurDecomposition();
    if (m_dimA == 2)
      compute2x2(m_p);
    else
      computeBig();
    result = m_U * m_fT * m_U.adjoint();
  }
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
void MatrixPower<MatrixType,std::complex<RealScalar>,PlainObject,IsInteger>::computeSchurDecomposition()
{
  const ComplexSchur<MatrixType> schurOfA(m_A);
  m_T = schurOfA.matrixT();
  m_U = schurOfA.matrixU();
  m_logTdiag = m_T.diagonal().array().log();
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
typename MatrixType::Scalar MatrixPower<MatrixType,std::complex<RealScalar>,PlainObject,IsInteger>::atanh(const Scalar& x)
{
  using std::abs;
  using std::log;
  using std::sqrt;

  if (abs(x) > sqrt(NumTraits<RealScalar>::epsilon()))
    return RealScalar(0.5) * log((RealScalar(1) + x) / (RealScalar(1) - x));
  else
    return x + x*x*x / RealScalar(3);
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
void MatrixPower<MatrixType,std::complex<RealScalar>,PlainObject,IsInteger>::compute2x2(const Scalar& p)
{
  using std::abs;
  using std::ceil;
  using std::exp;
  using std::imag;
  using std::ldexp;
  using std::log;
  using std::pow;
  using std::sinh;

  int i, j, unwindingNumber;
  Scalar w;

  m_fT(0,0) = pow(m_T(0,0), p);

  for (j = 1; j < m_dimA; j++) {
    i = j - 1;
    m_fT(j,j) = pow(m_T(j,j), p);

    if (m_T(i,i) == m_T(j,j))
      m_fT(i,j) = p * pow(m_T(i,j), p - RealScalar(1));
    else if (abs(m_T(i,i)) < ldexp(abs(m_T(j,j)), -1) || abs(m_T(j,j)) < ldexp(abs(m_T(i,i)), -1))
      m_fT(i,j) = m_T(i,j) * (m_fT(j,j) - m_fT(i,i)) / (m_T(j,j) - m_T(i,i));
    else {
      // computation in previous branch is inaccurate if abs(m_T(j,j)) \approx abs(m_T(i,i))
      unwindingNumber = static_cast<int>(ceil((imag(m_logTdiag[j] - m_logTdiag[i]) - M_PI) / (2 * M_PI)));
      w = atanh((m_T(j,j) - m_T(i,i)) / (m_T(j,j) + m_T(i,i))) + Scalar(0, M_PI * unwindingNumber);
      m_fT(i,j) = m_T(i,j) * RealScalar(2) * exp(RealScalar(0.5) * p * (m_logTdiag[j] + m_logTdiag[i])) *
	  sinh(p * w) / (m_T(j,j) - m_T(i,i));
    }
  }
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
void MatrixPower<MatrixType,std::complex<RealScalar>,PlainObject,IsInteger>::computeBig()
{
  using std::abs;
  using std::ceil;
  using std::frexp;
  using std::ldexp;

  const RealScalar maxNormForPade = 2.787629930862099e-1;
  int degree, degree2, numberOfSquareRoots, numberOfExtraSquareRoots = 0;
  MatrixType IminusT, sqrtT, T = m_T;
  RealScalar normIminusT;
  Scalar p;
/*
  frexp(abs(m_p), &numberOfSquareRoots);
  if (numberOfSquareRoots > 0)
    p = m_p * ldexp(RealScalar(1), -numberOfSquareRoots);
  else {
    p = m_p;
    numberOfSquareRoots = 0;
  }
*/
  while (true) {
    IminusT = MatrixType::Identity(m_A.rows(), m_A.cols()) - T;
    normIminusT = IminusT.cwiseAbs().colwise().sum().maxCoeff();
    if (normIminusT < maxNormForPade) {
      degree = getPadeDegree(normIminusT);
      degree2 = getPadeDegree(normIminusT * RealScalar(0.5));
      if (degree - degree2 <= 1 || numberOfExtraSquareRoots)
	break;
      numberOfExtraSquareRoots++;
    }
    MatrixSquareRootTriangular<MatrixType>(T).compute(sqrtT);
    T = sqrtT;
    numberOfSquareRoots++;
  }
  computePade(degree, IminusT);

  for (; numberOfSquareRoots; numberOfSquareRoots--) {
    compute2x2(p * ldexp(RealScalar(1), -numberOfSquareRoots));
    m_fT *= m_fT;
  }
  compute2x2(p);
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
inline int MatrixPower<MatrixType,std::complex<RealScalar>,PlainObject,IsInteger>::getPadeDegree(double normIminusT)
{
  const double maxNormForPade[] = { 1.882832775783885e-2 /* degree = 3 */ , 6.036100693089764e-2,
      1.239372725584911e-1, 1.998030690604271e-1, 2.787629930862099e-1 };
  for (int degree = 3; degree <= 7; degree++)
    if (normIminusT <= maxNormForPade[degree - 3])
      return degree;
  assert(false); // this line should never be reached
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
void MatrixPower<MatrixType,std::complex<RealScalar>,PlainObject,IsInteger>::computePade(int degree, const MatrixType& IminusT)
{
  degree <<= 1;
  m_fT = coeff(degree) * IminusT;

  for (int i = degree - 1; i; i--) {
    m_fT = (MatrixType::Identity(m_A.rows(), m_A.cols()) + m_fT).template triangularView<Upper>()
	.solve(coeff(i) * IminusT).eval();
  }
  m_fT += MatrixType::Identity(m_A.rows(), m_A.cols());
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
inline typename MatrixType::Scalar MatrixPower<MatrixType,std::complex<RealScalar>,PlainObject,IsInteger>::coeff(int i)
{
  if (i == 1)
    return -m_p;
  else if (i & 1)
    return (-m_p - RealScalar(i)) / RealScalar((i<<2) + 2);
  else
    return (m_p - RealScalar(i)) / RealScalar((i<<2) - 2);
}

/**
 * \ingroup MatrixFunctions_Module
 *
 * \brief Proxy for the matrix power multiplied by another matrix
 * (expression).
 *
 * \tparam MatrixType    type of the base, a matrix (expression).
 * \tparam ExponentType  type of the exponent, a scalar.
 * \tparam Derived       type of the multiplier, a matrix (expression).
 *
 * This class holds the arguments to the matrix expression until it is
 * assigned or evaluated for some other reason (so the argument
 * should not be changed in the meantime). It is the return type of
 * MatrixPowerReturnValue::operator*() and most of the time this is the
 * only way it is used.
 */
template<typename MatrixType, typename ExponentType, typename Derived> class MatrixPowerMultiplied
: public ReturnByValue<MatrixPowerMultiplied<MatrixType, ExponentType, Derived> >
{
  public:
    typedef typename Derived::Index Index;

    /**
     * \brief Constructor.
     *
     * \param[in] A  %Matrix (expression), the base of the matrix power.
     * \param[in] p  scalar, the exponent of the matrix power.
     * \param[in] b  %Matrix (expression), the multiplier.
     */
    MatrixPowerMultiplied(const MatrixType& A, const ExponentType& p, const Derived& b)
    : m_A(A), m_p(p), m_b(b) { }

    /**
     * \brief Compute the matrix exponential.
     *
     * \param[out] result  \f$ A^p b \f$ where \c A ,\c p and \c b are as in
     * the constructor.
     */
    template <typename ResultType>
    inline void evalTo(ResultType& result) const
    {
      typedef typename Derived::PlainObject PlainObject;
      const typename MatrixType::PlainObject Aevaluated = m_A.eval();
      const PlainObject bevaluated = m_b.eval();
      MatrixPower<MatrixType, ExponentType, PlainObject> mp(Aevaluated, m_p, bevaluated);
      mp.compute(result);
    }

    Index rows() const { return m_b.rows(); }
    Index cols() const { return m_b.cols(); }

  private:
    const MatrixType& m_A;
    const ExponentType& m_p;
    const Derived& m_b;

    MatrixPowerMultiplied& operator=(const MatrixPowerMultiplied&);
};

/**
 * \ingroup MatrixFunctions_Module
 *
 * \brief Proxy for the matrix power of some matrix (expression).
 *
 * \tparam Derived       type of the base, a matrix (expression).
 * \tparam ExponentType  type of the exponent, a scalar.
 *
 * This class holds the arguments to the matrix power until it is
 * assigned or evaluated for some other reason (so the argument
 * should not be changed in the meantime). It is the return type of
 * MatrixBase::pow() and related functions and most of the
 * time this is the only way it is used.
 */
template<typename Derived, typename ExponentType> class MatrixPowerReturnValue
: public ReturnByValue<MatrixPowerReturnValue<Derived, ExponentType> >
{
  public:
    typedef typename Derived::Index Index;

    /**
     * \brief Constructor.
     *
     * \param[in] A  %Matrix (expression), the base of the matrix power.
     * \param[in] p  scalar, the exponent of the matrix power.
     */
    MatrixPowerReturnValue(const Derived& A, const ExponentType& p)
    : m_A(A), m_p(p) { }

    /**
     * \brief Return the matrix power multiplied by %Matrix \c b.
     *
     * The %MatrixPower class can optimize \f$ A^p b \f$ computing, and this
     * method provides an elegant way to call it:
     *
     * \param[in] b  %Matrix (exporession), the multiplier.
     */
    template <typename OtherDerived>
    const MatrixPowerMultiplied<Derived, ExponentType, OtherDerived> operator*(const MatrixBase<OtherDerived>& b) const
    { return MatrixPowerMultiplied<Derived, ExponentType, OtherDerived>(m_A, m_p, b.derived()); }

    /**
     * \brief Compute the matrix power.
     *
     * \param[out] result  \f$ A^p \f$ where \c A and \c p are as in the
     * constructor.
     */
    template <typename ResultType>
    inline void evalTo(ResultType& result) const
    {
      typedef typename Derived::PlainObject PlainObject;
      const PlainObject Aevaluated = m_A.eval();
      const PlainObject Identity = PlainObject::Identity(m_A.rows(), m_A.cols());
      MatrixPower<PlainObject, ExponentType> mp(Aevaluated, m_p, Identity);
      mp.compute(result);
    }

    Index rows() const { return m_A.rows(); }
    Index cols() const { return m_A.cols(); }

  private:
    const Derived& m_A;
    const ExponentType& m_p;

    MatrixPowerReturnValue& operator=(const MatrixPowerReturnValue&);
};

namespace internal {
  template<typename MatrixType, typename ExponentType, typename Derived>
  struct traits<MatrixPowerMultiplied<MatrixType, ExponentType, Derived> >
  {
    typedef typename Derived::PlainObject ReturnType;
  };

  template<typename Derived, typename ExponentType>
  struct traits<MatrixPowerReturnValue<Derived, ExponentType> >
  {
    typedef typename Derived::PlainObject ReturnType;
  };
}

template <typename Derived>
template <typename ExponentType>
const MatrixPowerReturnValue<Derived, ExponentType> MatrixBase<Derived>::pow(const ExponentType& p) const
{
  eigen_assert(rows() == cols());
  return MatrixPowerReturnValue<Derived, ExponentType>(derived(), p);
}

} // end namespace Eigen

#endif // EIGEN_MATRIX_POWER
