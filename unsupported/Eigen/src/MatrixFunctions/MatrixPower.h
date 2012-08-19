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
     * If \p b is \em fatter than \p A, it computes \f$ A^{p_{\textrm int}}
     * \f$ first, and then multiplies it with \p b. Otherwise,
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
    void partialPivLuSolve(ResultType&, RealScalar);

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

    /** \brief Get suitable degree for Pade approximation. (specialized for RealScalar = double) */
    inline int getPadeDegree(double);

    /** \brief Get suitable degree for Pade approximation. (specialized for RealScalar = float) */
    inline int getPadeDegree(float);
  
    /** \brief Get suitable degree for Pade approximation. (specialized for RealScalar = long double) */
    inline int getPadeDegree(long double);

    /** \brief Compute Pad&eacute; approximation to matrix fractional power. */
    void computePade(const int& degree, const ComplexMatrix& IminusT);

    /** \brief Get a certain coefficient of the Pad&eacute; approximation. */
    inline RealScalar coeff(const int& degree);

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
     * If \p b is \em fatter than \p A, it computes \f$ A^p \f$ first, and
     * then multiplies it with \p b. Otherwise, #computeChainProduct
     * optimizes the expression.
     *
     * \param[out] result  \f$ A^p b \f$, as specified in the constructor.
     *
     * \sa computeChainProduct(ResultType&);
     */
    template <typename ResultType>
    void compute(ResultType& result);

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
    template <typename ResultType>
    void computeChainProduct(ResultType& result);

    /** \brief Compute the cost of binary powering. */
    int computeCost(const IntExponent& p);

    /** \brief Solve the linear system repetitively. */
    template <typename ResultType>
    void partialPivLuSolve(ResultType&, IntExponent);
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
      partialPivLuSolve(result, p);
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
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::partialPivLuSolve(ResultType& result, RealScalar p)
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
  const int digits = std::numeric_limits<RealScalar>::digits;
  const RealScalar maxNormForPade = digits <=  24? 4.3268868e-1f:                           // sigle precision
                                    digits <=  53? 2.787629930861592e-1:                    // double precision
				    digits <=  64? 2.4461702976649554343e-1L:               // extended precision
				    digits <= 106? 1.1015697751808768849251777304538e-01:   // double-double
				                   9.133823549851655878933476070874651e-02; // quadruple precision
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
inline int MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::getPadeDegree(float normIminusT)
{
  const float maxNormForPade[] = { 2.7996156e-1f /* degree = 3 */ , 4.3268868e-1f };
  int degree = 3;
  for (; degree <= 4; degree++)
    if (normIminusT <= maxNormForPade[degree - 3])
      break;
  return degree;
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
inline int MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::getPadeDegree(double normIminusT)
{
  const double maxNormForPade[] = { 1.882832775783710e-2 /* degree = 3 */ , 6.036100693089536e-2,
      1.239372725584857e-1, 1.998030690604104e-1, 2.787629930861592e-1 };
  int degree = 3;
  for (; degree <= 7; degree++)
    if (normIminusT <= maxNormForPade[degree - 3])
      break;
  return degree;
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
inline int MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::getPadeDegree(long double normIminusT)
{
#if LDBL_MANT_DIG == 53
  const int maxPadeDegree = 7;
  const double maxNormForPade[] = { 1.882832775783710e-2L /* degree = 3 */ , 6.036100693089536e-2L,
      1.239372725584857e-1L, 1.998030690604104e-1L, 2.787629930861592e-1L };

#elif LDBL_MANT_DIG <= 64
  const int maxPadeDegree = 8;
  const double maxNormForPade[] = { 6.3813036421433454225e-3L /* degree = 3 */ , 2.6385399995942000637e-2L,
      6.4197808148473250951e-2L, 1.1697754827125334716e-1L, 1.7898159424022851851e-1L, 2.4461702976649554343e-1L };

#elif LDBL_MANT_DIG <= 106
  const int maxPadeDegree = 10;
  const double maxNormForPade[] = { 1.0007009771231429252734273435258e-4L /* degree = 3 */ ,
      1.0538187257176867284131299608423e-3L, 4.7061962004060435430088460028236e-3L, 1.3218912040677196137566177023204e-2L,
      2.8060971416164795541562544777056e-2L, 4.9621804942978599802645569010027e-2L, 7.7360065339071543892274529471454e-2L,
      1.1015697751808768849251777304538e-1L };
#else
  const int maxPadeDegree = 10;
  const double maxNormForPade[] = { 5.524459874082058900800655900644241e-5L /* degree = 3 */ ,
      6.640087564637450267909344775414015e-4L, 3.227189204209204834777703035324315e-3L,
      9.618565213833446441025286267608306e-3L, 2.134419664210632655600344879830298e-2L,
      3.907876732697568523164749432441966e-2L, 6.266303975524852476985111609267074e-2L,
      9.133823549851655878933476070874651e-2L };
#endif
  int degree = 3;
  for (; degree <= maxPadeDegree; degree++)
    if (normIminusT <= maxNormForPade[degree - 3])
      break;
  return degree;
}
template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
void MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::computePade(const int& degree, const ComplexMatrix& IminusT)
{
  int i = degree << 1;
  m_fT = coeff(i) * IminusT;
  for (i--; i; i--) {
    m_fT = (ComplexMatrix::Identity(m_A.rows(), m_A.cols()) + m_fT).template triangularView<Upper>()
	.solve(coeff(i) * IminusT).eval();
  }
  m_fT += ComplexMatrix::Identity(m_A.rows(), m_A.cols());
}

template <typename MatrixType, typename RealScalar, typename PlainObject, int IsInteger>
inline RealScalar MatrixPower<MatrixType,RealScalar,PlainObject,IsInteger>::coeff(const int& i)
{
  if (i == 1)
    return -m_pfrac;
  else if (i & 1)
    return (-m_pfrac - RealScalar(i >> 1)) / RealScalar(i << 1);
  else
    return (m_pfrac - RealScalar(i >> 1)) / RealScalar(i-1 << 1);
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
void MatrixPower<MatrixType,IntExponent,PlainObject,1>::partialPivLuSolve(ResultType& result, IntExponent p)
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
      partialPivLuSolve(result, p);
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
     * \param[out] result  \f$ A^p b \f$ where \p A ,\p p and \p b are as in
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
     * \brief Return the matrix power multiplied by %Matrix \p b.
     *
     * The %MatrixPower class can optimize \f$ A^p b \f$ computing, and this
     * method provides an elegant way to call it:
     *
     * \param[in] b  %Matrix (expression), the multiplier.
     */
    template <typename OtherDerived>
    const MatrixPowerMultiplied<Derived, ExponentType, OtherDerived> operator*(const MatrixBase<OtherDerived>& b) const
    { return MatrixPowerMultiplied<Derived, ExponentType, OtherDerived>(m_A, m_p, b.derived()); }

    /**
     * \brief Compute the matrix power.
     *
     * \param[out] result  \f$ A^p \f$ where \p A and \p p are as in the
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
