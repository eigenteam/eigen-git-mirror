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
 * \tparam PlainObject  type of the multiplier.
 */
template<typename MatrixType, typename PlainObject = MatrixType>
class MatrixPower
{
  private:
    typedef internal::traits<MatrixType> Traits;
    static const int Rows = Traits::RowsAtCompileTime;
    static const int Cols = Traits::ColsAtCompileTime;
    static const int Options = Traits::Options;
    static const int MaxRows = Traits::MaxRowsAtCompileTime;
    static const int MaxCols = Traits::MaxColsAtCompileTime;

    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef std::complex<RealScalar> ComplexScalar;
    typedef typename MatrixType::Index Index;
    typedef Matrix<ComplexScalar, Rows, Cols, Options, MaxRows, MaxCols> ComplexMatrix;
    typedef Array<ComplexScalar, Rows, 1, ColMajor, MaxRows> ComplexArray;

  public:
    /**
     * \brief Constructor.
     *
     * \param[in] A  the base of the matrix power.
     * \param[in] p  the exponent of the matrix power.
     * \param[in] b  the multiplier.
     */
    MatrixPower(const MatrixType& A, RealScalar p, const PlainObject& b) :
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
    template<typename ResultType> void compute(ResultType& result);
 
  private:
    /**
     * \brief Compute the matrix to integral power.
     *
     * If \p b is \em fatter than \p A, it computes \f$ A^{p_{\textrm int}}
     * \f$ first, and then multiplies it with \p b. Otherwise,
     * #computeChainProduct optimizes the expression.
     *
     * \sa computeChainProduct(ResultType&);
     */
    template<typename ResultType>
    void computeIntPower(ResultType& result);

    /**
     * \brief Convert integral power of the matrix into chain product.
     *
     * This optimizes the matrix expression. It automatically chooses binary
     * powering or matrix chain multiplication or solving the linear system
     * repetitively, according to which algorithm costs less.
     */
    template<typename ResultType>
    void computeChainProduct(ResultType&);

    /** \brief Compute the cost of binary powering. */
    static int computeCost(RealScalar);

    /** \brief Solve the linear system repetitively. */
    template<typename ResultType>
    void partialPivLuSolve(ResultType&, RealScalar);

    /** \brief Compute Schur decomposition of #m_A. */
    void computeSchurDecomposition();

    /**
     * \brief Split #m_p into integral part and fractional part.
     *
     * This method stores the integral part \f$ p_{\textrm int} \f$ into
     * #m_pInt and the fractional part \f$ p_{\textrm frac} \f$ into
     * #m_pFrac, where #m_pFrac is in the interval \f$ (-1,1) \f$. To
     * choose between the possibilities below, it considers the computation
     * of \f$ A^{p_1} \f$ and \f$ A^{p_2} \f$ and determines which of these
     * computations is the better conditioned.
     */
    void getFractionalExponent();

    /** \brief Compute atanh (inverse hyperbolic tangent) for \f$ y / x \f$. */
    static ComplexScalar atanh2(const ComplexScalar& y, const ComplexScalar& x);

    /** \brief Compute power of 2x2 triangular matrix. */
    void compute2x2(RealScalar p);

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
    const RealScalar m_p;    ///< \brief The real exponent.
    const PlainObject& m_b;  ///< \brief Reference to the multiplier.
    const Index m_dimA;      ///< \brief The dimension of #m_A, equivalent to %m_A.cols().
    const Index m_dimb;      ///< \brief The dimension of #m_b, equivalent to %m_b.cols().
    MatrixType m_tmp;        ///< \brief Used for temporary storage.
    RealScalar m_pInt;       ///< \brief Integral part of #m_p.
    RealScalar m_pFrac;      ///< \brief Fractional part of #m_p.
    ComplexMatrix m_T;       ///< \brief Triangular part of Schur decomposition.
    ComplexMatrix m_U;       ///< \brief Unitary part of Schur decomposition.
    ComplexMatrix m_fT;      ///< \brief #m_T to the power of #m_pFrac.
    ComplexArray m_logTdiag; ///< \brief Logarithm of the main diagonal of #m_T.
};

template<typename MatrixType, typename PlainObject>
template<typename ResultType>
void MatrixPower<MatrixType,PlainObject>::compute(ResultType& result)
{
  using std::floor;
  using std::pow;

  m_pInt = floor(m_p + RealScalar(0.5));
  m_pFrac = m_p - m_pInt;

  if (!m_pFrac) {
    computeIntPower(result);
  } else if (m_dimA == 1)
    result = pow(m_A(0,0), m_p) * m_b;
  else {
    computeSchurDecomposition();
    getFractionalExponent();
    computeIntPower(result);
    if (m_dimA == 2)
      compute2x2(m_pFrac);
    else
      computeBig();
    computeTmp(Scalar());
    result = m_tmp * result;
  }
}

template<typename MatrixType, typename PlainObject>
template<typename ResultType>
void MatrixPower<MatrixType,PlainObject>::computeIntPower(ResultType& result)
{
  MatrixType tmp;
  if (m_dimb > m_dimA) {
    tmp = MatrixType::Identity(m_dimA, m_dimA);
    computeChainProduct(tmp);
    result.noalias() = tmp * m_b;
  } else {
    result = m_b;
    computeChainProduct(result);
  }
}

template<typename MatrixType, typename PlainObject>
template<typename ResultType>
void MatrixPower<MatrixType,PlainObject>::computeChainProduct(ResultType& result)
{
  using std::abs;
  using std::fmod;
  using std::ldexp;

  RealScalar p = abs(m_pInt);
  int cost = computeCost(p);

  if (m_pInt < RealScalar(0)) {
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

template<typename MatrixType, typename PlainObject>
int MatrixPower<MatrixType,PlainObject>::computeCost(RealScalar p)
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

template<typename MatrixType, typename PlainObject>
template<typename ResultType>
void MatrixPower<MatrixType,PlainObject>::partialPivLuSolve(ResultType& result, RealScalar p)
{
  const PartialPivLU<MatrixType> Asolver(m_A);
  for (; p >= RealScalar(1); p--)
    result = Asolver.solve(result);
}

template<typename MatrixType, typename PlainObject>
void MatrixPower<MatrixType,PlainObject>::computeSchurDecomposition()
{
  const ComplexSchur<MatrixType> schurOfA(m_A);
  m_T = schurOfA.matrixT();
  m_U = schurOfA.matrixU();
}

template<typename MatrixType, typename PlainObject>
void MatrixPower<MatrixType,PlainObject>::getFractionalExponent()
{
  using std::pow;
  typedef Array<RealScalar, Rows, 1, ColMajor, MaxRows> RealArray;

  const ComplexArray Tdiag = m_T.diagonal();
  const RealArray absTdiag = Tdiag.abs();
  const RealScalar maxAbsEival = absTdiag.maxCoeff();
  const RealScalar minAbsEival = absTdiag.minCoeff();

  m_logTdiag = Tdiag.log();
  if (m_pFrac > RealScalar(0.5) &&  // This is just a shortcut.
      m_pFrac > (RealScalar(1) - m_pFrac) * pow(maxAbsEival/minAbsEival, m_pFrac)) {
    m_pFrac--;
    m_pInt++;
  }
}

template<typename MatrixType, typename PlainObject>
std::complex<typename MatrixType::RealScalar>
MatrixPower<MatrixType,PlainObject>::atanh2(const ComplexScalar& y, const ComplexScalar& x)
{
  using std::abs;
  using std::log;
  using std::sqrt;
  const ComplexScalar z = y / x;

  if (abs(z) > sqrt(NumTraits<RealScalar>::epsilon()))
    return RealScalar(0.5) * log((x + y) / (x - y));
  else
    return z + z*z*z / RealScalar(3);
}

template<typename MatrixType, typename PlainObject>
void MatrixPower<MatrixType,PlainObject>::compute2x2(RealScalar p)
{
  using std::abs;
  using std::ceil;
  using std::exp;
  using std::imag;
  using std::ldexp;
  using std::pow;
  using std::sinh;

  int i, j, unwindingNumber;
  ComplexScalar w;

  m_fT(0,0) = pow(m_T(0,0), p);
  for (j = 1; j < m_dimA; j++) {
    i = j - 1;
    m_fT(j,j) = pow(m_T(j,j), p);

    if (m_T(i,i) == m_T(j,j)) {
      m_fT(i,j) = p * pow(m_T(i,j), p - RealScalar(1));
    } else if (abs(m_T(i,i)) < ldexp(abs(m_T(j,j)), -1) || abs(m_T(j,j)) < ldexp(abs(m_T(i,i)), -1)) {
      m_fT(i,j) = m_T(i,j) * (m_fT(j,j) - m_fT(i,i)) / (m_T(j,j) - m_T(i,i));
    } else {
      // computation in previous branch is inaccurate if abs(m_T(j,j)) \approx abs(m_T(i,i))
      unwindingNumber = ceil((imag(m_logTdiag[j] - m_logTdiag[i]) - M_PI) / (2 * M_PI));
      w = atanh2(m_T(j,j) - m_T(i,i), m_T(j,j) + m_T(i,i)) + ComplexScalar(0, M_PI * unwindingNumber);
      m_fT(i,j) = m_T(i,j) * RealScalar(2) * exp(RealScalar(0.5) * p * (m_logTdiag[j] + m_logTdiag[i])) *
	  sinh(p * w) / (m_T(j,j) - m_T(i,i));
    }
  }
}

template<typename MatrixType, typename PlainObject>
void MatrixPower<MatrixType,PlainObject>::computeBig()
{
  using std::ldexp;
  const int digits = std::numeric_limits<RealScalar>::digits;
  const RealScalar maxNormForPade = digits <=  24? 4.3386528e-1f:                           // sigle precision
                                    digits <=  53? 2.789358995219730e-1:                    // double precision
				    digits <=  64? 2.4471944416607995472e-1L:               // extended precision
				    digits <= 106? 1.1016843812851143391275867258512e-01:   // double-double
				                   9.134603732914548552537150753385375e-02; // quadruple precision
  int degree, degree2, numberOfSquareRoots = 0, numberOfExtraSquareRoots = 0;
  ComplexMatrix IminusT, sqrtT, T = m_T;
  RealScalar normIminusT;

  while (true) {
    IminusT = ComplexMatrix::Identity(m_dimA, m_dimA) - T;
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
    compute2x2(ldexp(m_pFrac, -numberOfSquareRoots));
    m_fT *= m_fT;
  }
  compute2x2(m_pFrac);
}

template<typename MatrixType, typename PlainObject>
inline int MatrixPower<MatrixType,PlainObject>::getPadeDegree(float normIminusT)
{
  const float maxNormForPade[] = { 2.8064004e-1f /* degree = 3 */ , 4.3386528e-1f };
  int degree = 3;
  for (; degree <= 4; degree++)
    if (normIminusT <= maxNormForPade[degree - 3])
      break;
  return degree;
}

template<typename MatrixType, typename PlainObject>
inline int MatrixPower<MatrixType,PlainObject>::getPadeDegree(double normIminusT)
{
  const double maxNormForPade[] = { 1.884160592658218e-2 /* degree = 3 */ , 6.038881904059573e-2,
      1.239917516308172e-1, 1.999045567181744e-1, 2.789358995219730e-1 };
  int degree = 3;
  for (; degree <= 7; degree++)
    if (normIminusT <= maxNormForPade[degree - 3])
      break;
  return degree;
}

template<typename MatrixType, typename PlainObject>
inline int MatrixPower<MatrixType,PlainObject>::getPadeDegree(long double normIminusT)
{
#if LDBL_MANT_DIG == 53
  const int maxPadeDegree = 7;
  const double maxNormForPade[] = { 1.884160592658218e-2L /* degree = 3 */ , 6.038881904059573e-2L,
      1.239917516308172e-1L, 1.999045567181744e-1L, 2.789358995219730e-1L };

#elif LDBL_MANT_DIG <= 64
  const int maxPadeDegree = 8;
  const double maxNormForPade[] = { 6.3854693117491799460e-3L /* degree = 3 */ , 2.6394893435456973676e-2L,
      6.4216043030404063729e-2L, 1.1701165502926694307e-1L, 1.7904284231268670284e-1L, 2.4471944416607995472e-1L };

#elif LDBL_MANT_DIG <= 106
  const int maxPadeDegree = 10;
  const double maxNormForPade[] = { 1.0007161601787493236741409687186e-4L /* degree = 3 */ ,
      1.0007161601787493236741409687186e-3L, 4.7069769360887572939882574746264e-3L, 1.3220386624169159689406653101695e-2L,
      2.8063482381631737920612944054906e-2L, 4.9625993951953473052385361085058e-2L, 7.7367040706027886224557538328171e-2L,
      1.1016843812851143391275867258512e-1L };
#else
  const int maxPadeDegree = 10;
  const double maxNormForPade[] = { 5.524506147036624377378713555116378e-5L /* degree = 3 */ ,
      6.640600568157479679823602193345995e-4L, 3.227716520106894279249709728084626e-3L,
      9.619593944683432960546978734646284e-3L, 2.134595382433742403911124458161147e-2L,
      3.908166513900489428442993794761185e-2L, 6.266780814639442865832535460550138e-2L,
      9.134603732914548552537150753385375e-2L };
#endif
  int degree = 3;
  for (; degree <= maxPadeDegree; degree++)
    if (normIminusT <= maxNormForPade[degree - 3])
      break;
  return degree;
}
template<typename MatrixType, typename PlainObject>
void MatrixPower<MatrixType,PlainObject>::computePade(const int& degree, const ComplexMatrix& IminusT)
{
  int i = degree << 1;
  m_fT = coeff(i) * IminusT;
  for (i--; i; i--) {
    m_fT = (ComplexMatrix::Identity(m_dimA, m_dimA) + m_fT).template triangularView<Upper>()
	.solve(coeff(i) * IminusT).eval();
  }
  m_fT += ComplexMatrix::Identity(m_dimA, m_dimA);
}

template<typename MatrixType, typename PlainObject>
inline typename MatrixType::RealScalar MatrixPower<MatrixType,PlainObject>::coeff(const int& i)
{
  if (i == 1)
    return -m_pFrac;
  else if (i & 1)
    return (-m_pFrac - RealScalar(i >> 1)) / RealScalar(i << 1);
  else
    return (m_pFrac - RealScalar(i >> 1)) / RealScalar((i - 1) << 1);
}

template<typename MatrixType, typename PlainObject>
void MatrixPower<MatrixType,PlainObject>::computeTmp(RealScalar)
{ m_tmp = (m_U * m_fT * m_U.adjoint()).real(); }

template<typename MatrixType, typename PlainObject>
void MatrixPower<MatrixType,PlainObject>::computeTmp(ComplexScalar)
{ m_tmp = m_U * m_fT * m_U.adjoint(); }

/**
 * \ingroup MatrixFunctions_Module
 *
 * \brief Proxy for the matrix power multiplied by other matrix.
 *
 * \tparam Derived     type of the base, a matrix (expression).
 * \tparam RhsDerived  type of the multiplier.
 *
 * This class holds the arguments to the matrix power until it is
 * assigned or evaluated for some other reason (so the argument
 * should not be changed in the meantime). It is the return type of
 * MatrixPowerReturnValue::operator*() and related functions and most
 * of the time this is the only way it is used.
 */
template<typename Derived, typename RhsDerived>
class MatrixPowerProductReturnValue : public ReturnByValue<MatrixPowerProductReturnValue<Derived,RhsDerived> >
{
  private:
    typedef typename Derived::PlainObject MatrixType;
    typedef typename RhsDerived::PlainObject PlainObject;
    typedef typename RhsDerived::RealScalar RealScalar;
    typedef typename RhsDerived::Index Index;

  public:
    /**
     * \brief Constructor.
     *
     * \param[in] A  %Matrix (expression), the base of the matrix power.
     * \param[in] p  scalar, the exponent of the matrix power.
     * \prarm[in] b  %Matrix (expression), the multiplier.
     */
    MatrixPowerProductReturnValue(const Derived& A, RealScalar p, const RhsDerived& b)
    : m_A(A), m_p(p), m_b(b) { }

    /**
     * \brief Compute the expression.
     *
     * \param[out] result  \f$ A^p b \f$ where \p A, \p p and \p bare as
     * in the constructor.
     */
    template<typename ResultType>
    inline void evalTo(ResultType& result) const
    {
      const MatrixType A = m_A;
      const PlainObject b = m_b;
      MatrixPower<MatrixType, PlainObject> mp(A, m_p, b);
      mp.compute(result);
    }

    Index rows() const { return m_b.rows(); }
    Index cols() const { return m_b.cols(); }

  private:
    const Derived& m_A;
    const RealScalar m_p;
    const RhsDerived& m_b;
    MatrixPowerProductReturnValue& operator=(const MatrixPowerProductReturnValue&);
};

/**
 * \ingroup MatrixFunctions_Module
 *
 * \brief Proxy for the matrix power of some matrix (expression).
 *
 * \tparam Derived  type of the base, a matrix (expression).
 *
 * This class holds the arguments to the matrix power until it is
 * assigned or evaluated for some other reason (so the argument
 * should not be changed in the meantime). It is the return type of
 * MatrixBase::pow() and related functions and most of the
 * time this is the only way it is used.
 */
template<typename Derived>
class MatrixPowerReturnValue : public ReturnByValue<MatrixPowerReturnValue<Derived> >
{
  private:
    typedef typename Derived::RealScalar RealScalar;
    typedef typename Derived::Index Index;

  public:
    /**
     * \brief Constructor.
     *
     * \param[in] A  %Matrix (expression), the base of the matrix power.
     * \param[in] p  scalar, the exponent of the matrix power.
     */
    MatrixPowerReturnValue(const Derived& A, RealScalar p)
    : m_A(A), m_p(p) { }

    /**
     * \brief Return the matrix power multiplied by %Matrix \p b.
     *
     * The %MatrixPower class can optimize \f$ A^p b \f$ computing, and
     * this method provides an elegant way to call it.
     *
     * Unlike general matrix-matrix / matrix-vector product, this does
     * \b NOT produce a temporary storage for the result. Therefore,
     * the code below is \a already optimal:
     * \code
     * v = A.pow(p) * b;
     * // v.noalias() = A.pow(p) * b; Won't compile!
     * \endcode
     *
     * \param[in] b  %Matrix (expression), the multiplier.
     */
    template<typename RhsDerived>
    const MatrixPowerProductReturnValue<Derived,RhsDerived> operator*(const MatrixBase<RhsDerived>& b) const
    { return MatrixPowerProductReturnValue<Derived,RhsDerived>(m_A, m_p, b.derived()); }

    /**
     * \brief Compute the matrix power.
     *
     * \param[out] result  \f$ A^p \f$ where \p A and \p p are as in the
     * constructor.
     */
    template<typename ResultType>
    inline void evalTo(ResultType& result) const
    {
      typedef typename Derived::PlainObject PlainObject;
      const PlainObject A = m_A;
      const PlainObject Identity = PlainObject::Identity(m_A.rows(), m_A.cols());
      MatrixPower<PlainObject> mp(A, m_p, Identity);
      mp.compute(result);
    }

    Index rows() const { return m_A.rows(); }
    Index cols() const { return m_A.cols(); }

  private:
    const Derived& m_A;
    const RealScalar m_p;
    MatrixPowerReturnValue& operator=(const MatrixPowerReturnValue&);
};

namespace internal {
  template<typename Derived>
  struct traits<MatrixPowerReturnValue<Derived> >
  {
    typedef typename Derived::PlainObject ReturnType;
  };

  template<typename Derived, typename RhsDerived>
  struct traits<MatrixPowerProductReturnValue<Derived,RhsDerived> >
  {
    typedef typename RhsDerived::PlainObject ReturnType;
  };
}

template<typename Derived>
const MatrixPowerReturnValue<Derived> MatrixBase<Derived>::pow(RealScalar p) const
{
  eigen_assert(rows() == cols());
  return MatrixPowerReturnValue<Derived>(derived(), p);
}

} // end namespace Eigen

#endif // EIGEN_MATRIX_POWER
