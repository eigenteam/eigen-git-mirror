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

namespace Eigen {

namespace internal {

template<int IsComplex>
struct recompose_complex_schur
{
  template<typename ResultType, typename MatrixType>
  static inline void run(ResultType& res, const MatrixType& T, const MatrixType& U)
  { res = U * (T.template triangularView<Upper>() * U.adjoint()); }
};

template<>
struct recompose_complex_schur<0>
{
  template<typename ResultType, typename MatrixType>
  static inline void run(ResultType& res, const MatrixType& T, const MatrixType& U)
  { res = (U * (T.template triangularView<Upper>() * U.adjoint())).real(); }
};

template<typename T>
inline int binary_powering_cost(T p)
{
  int cost, tmp;
  frexp(p, &cost);
  while (std::frexp(p, &tmp), tmp > 0) {
    p -= std::ldexp(static_cast<T>(0.5), tmp);
    ++cost;
  }
  return cost;
}

inline int matrix_power_get_pade_degree(float normIminusT)
{
  const float maxNormForPade[] = { 2.8064004e-1f /* degree = 3 */ , 4.3386528e-1f };
  int degree = 3;
  for (; degree <= 4; ++degree)
    if (normIminusT <= maxNormForPade[degree - 3])
      break;
  return degree;
}

inline int matrix_power_get_pade_degree(double normIminusT)
{
  const double maxNormForPade[] = { 1.884160592658218e-2 /* degree = 3 */ , 6.038881904059573e-2, 1.239917516308172e-1,
      1.999045567181744e-1, 2.789358995219730e-1 };
  int degree = 3;
  for (; degree <= 7; ++degree)
    if (normIminusT <= maxNormForPade[degree - 3])
      break;
  return degree;
}

inline int matrix_power_get_pade_degree(long double normIminusT)
{
#if   LDBL_MANT_DIG == 53
  const int maxPadeDegree = 7;
  const double maxNormForPade[] = { 1.884160592658218e-2L /* degree = 3 */ , 6.038881904059573e-2L, 1.239917516308172e-1L,
      1.999045567181744e-1L, 2.789358995219730e-1L };
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
  for (; degree <= maxPadeDegree; ++degree)
    if (normIminusT <= maxNormForPade[degree - 3])
      break;
  return degree;
}
} // namespace internal

/* (non-doc)
 * \brief Class for computing triangular matrices to fractional power.
 *
 * \tparam MatrixType  type of the base.
 */
template<typename MatrixType, int UpLo = Upper> class MatrixPowerTriangularAtomic
{
  private:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef Array<Scalar,
		  EIGEN_SIZE_MIN_PREFER_FIXED(MatrixType::RowsAtCompileTime,MatrixType::ColsAtCompileTime),
		  1,ColMajor,
		  EIGEN_SIZE_MIN_PREFER_FIXED(MatrixType::MaxRowsAtCompileTime,MatrixType::MaxColsAtCompileTime)> ArrayType;
    const MatrixType& m_T;

    void computePade(int degree, const MatrixType& IminusT, MatrixType& res, RealScalar p) const;
    void compute2x2(MatrixType& res, RealScalar p) const;
    void computeBig(MatrixType& res, RealScalar p) const;

  public:
    explicit MatrixPowerTriangularAtomic(const MatrixType& T);
    void compute(MatrixType& res, RealScalar p) const;
};

template<typename MatrixType, int UpLo>
MatrixPowerTriangularAtomic<MatrixType,UpLo>::MatrixPowerTriangularAtomic(const MatrixType& T) :
  m_T(T)
{ eigen_assert(T.rows() == T.cols()); }

template<typename MatrixType, int UpLo>
void MatrixPowerTriangularAtomic<MatrixType,UpLo>::compute(MatrixType& res, RealScalar p) const
{
  switch (m_T.rows()) {
    case 0:
      break;
    case 1:
      res(0,0) = std::pow(m_T(0,0), p);
      break;
    case 2:
      compute2x2(res, p);
      break;
    default:
      computeBig(res, p);
  }
}

template<typename MatrixType, int UpLo>
void MatrixPowerTriangularAtomic<MatrixType,UpLo>::computePade(int degree, const MatrixType& IminusT, MatrixType& res,
  RealScalar p) const
{
  int i = degree<<1;
  res = (p-(i>>1)) / ((i-1)<<1) * IminusT;
  for (--i; i; --i) {
    res = (MatrixType::Identity(m_T.rows(), m_T.cols()) + res).template triangularView<UpLo>()
	.solve((i==1 ? -p : i&1 ? (-p-(i>>1))/(i<<1) : (p-(i>>1))/((i-1)<<1)) * IminusT).eval();
  }
  res += MatrixType::Identity(m_T.rows(), m_T.cols());
}

template<typename MatrixType, int UpLo>
void MatrixPowerTriangularAtomic<MatrixType,UpLo>::compute2x2(MatrixType& res, RealScalar p) const
{
  using std::abs;
  using std::pow;
  
  ArrayType logTdiag = m_T.diagonal().array().log();
  res(0,0) = pow(m_T(0,0), p);

  for (int i=1; i < m_T.cols(); ++i) {
    res(i,i) = pow(m_T(i,i), p);
    if (m_T(i-1,i-1) == m_T(i,i)) {
      res(i-1,i) = p * pow(m_T(i-1,i), p-1);
    } else if (2*abs(m_T(i-1,i-1)) < abs(m_T(i,i)) || 2*abs(m_T(i,i)) < abs(m_T(i-1,i-1))) {
      res(i-1,i) = m_T(i-1,i) * (res(i,i)-res(i-1,i-1)) / (m_T(i,i)-m_T(i-1,i-1));
    } else {
      // computation in previous branch is inaccurate if abs(m_T(i,i)) \approx abs(m_T(i-1,i-1))
      int unwindingNumber = std::ceil(((logTdiag[i]-logTdiag[i-1]).imag() - M_PI) / (2*M_PI));
      Scalar w = internal::atanh2(m_T(i,i)-m_T(i-1,i-1), m_T(i,i)+m_T(i-1,i-1)) + Scalar(0, M_PI*unwindingNumber);
      res(i-1,i) = m_T(i-1,i) * RealScalar(2) * std::exp(RealScalar(0.5) * p * (logTdiag[i]+logTdiag[i-1])) *
	  std::sinh(p * w) / (m_T(i,i) - m_T(i-1,i-1));
    }
  }
}

template<typename MatrixType, int UpLo>
void MatrixPowerTriangularAtomic<MatrixType,UpLo>::computeBig(MatrixType& res, RealScalar p) const
{
  const int digits = std::numeric_limits<RealScalar>::digits;
  const RealScalar maxNormForPade = digits <=  24? 4.3386528e-1f:                           // sigle precision
				    digits <=  53? 2.789358995219730e-1:                    // double precision
				    digits <=  64? 2.4471944416607995472e-1L:               // extended precision
				    digits <= 106? 1.1016843812851143391275867258512e-01:   // double-double
						   9.134603732914548552537150753385375e-02; // quadruple precision
  int degree, degree2, numberOfSquareRoots=0, numberOfExtraSquareRoots=0;
  MatrixType IminusT, sqrtT, T=m_T;
  RealScalar normIminusT;

  while (true) {
    IminusT = MatrixType::Identity(m_T.rows(), m_T.cols()) - T;
    normIminusT = IminusT.cwiseAbs().colwise().sum().maxCoeff();
    if (normIminusT < maxNormForPade) {
      degree = internal::matrix_power_get_pade_degree(normIminusT);
      degree2 = internal::matrix_power_get_pade_degree(normIminusT/2);
      if (degree - degree2 <= 1 || numberOfExtraSquareRoots)
	break;
      ++numberOfExtraSquareRoots;
    }
    MatrixSquareRootTriangular<MatrixType>(T).compute(sqrtT);
    T = sqrtT;
    ++numberOfSquareRoots;
  }
  computePade(degree, IminusT, res, p);

  for (; numberOfSquareRoots; --numberOfSquareRoots) {
    compute2x2(res, std::ldexp(p,-numberOfSquareRoots));
    res *= res;
  }
  compute2x2(res, p);
}

/**
 * \ingroup MatrixFunctions_Module
 *
 * \brief Class for computing matrix powers.
 *
 * \tparam MatrixType  type of the base, expected to be an instantiation
 * of the Matrix class template.
 *
 * This class is capable of computing real/complex matrices raised to
 * an arbitrary real power. Meanwhile, it saves the result of Schur
 * decomposition if an non-integral power has even been calculated.
 * Therefore, if you want to compute multiple (>= 2) matrix powers
 * for the same matrix, using the class directly is more efficient than
 * calling MatrixBase::pow().
 *
 * Example:
 * \include MatrixPower_optimal.cpp
 * Output: \verbinclude MatrixPower_optimal.out
 */
template<typename MatrixType> class MatrixPower
{
  private:
    static const int Rows    = MatrixType::RowsAtCompileTime;
    static const int Cols    = MatrixType::ColsAtCompileTime;
    static const int Options = MatrixType::Options;
    static const int MaxRows = MatrixType::MaxRowsAtCompileTime;
    static const int MaxCols = MatrixType::MaxColsAtCompileTime;

    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::Index Index;
    typedef Matrix<std::complex<RealScalar>,Rows,Cols,Options,MaxRows,MaxCols> ComplexMatrix;

    const MatrixType& m_A;
    MatrixType m_tmp1, m_tmp2;
    ComplexMatrix m_T, m_U, m_fT;
    bool m_init;

    RealScalar modfAndInit(RealScalar, RealScalar*);

    template<typename PlainObject, typename ResultType>
    void apply(const PlainObject&, ResultType&, bool&);

    template<typename ResultType>
    void computeIntPower(ResultType&, RealScalar);

    template<typename PlainObject, typename ResultType>
    void computeIntPower(const PlainObject&, ResultType&, RealScalar);

    template<typename ResultType>
    void computeFracPower(ResultType&, RealScalar);

  public:
    /**
     * \brief Constructor.
     *
     * \param[in] A  the base of the matrix power.
     */
    explicit MatrixPower(const MatrixType& A);

    /**
     * \brief Return the expression \f$ A^p \f$.
     *
     * \param[in] p  exponent, a real scalar.
     */
    const MatrixPowerReturnValue<MatrixPower<MatrixType> > operator()(RealScalar p)
    { return MatrixPowerReturnValue<MatrixPower<MatrixType> >(*this, p); }

    /**
     * \brief Compute the matrix power.
     *
     * \param[in]  p    exponent, a real scalar.
     * \param[out] res  \f$ A^p \f$ where A is specified in the
     * constructor.
     */
    void compute(MatrixType& res, RealScalar p);

    /**
     * \brief Compute the matrix power multiplied by another matrix.
     *
     * \param[in]  b    a matrix with the same rows as A.
     * \param[in]  p    exponent, a real scalar.
     * \param[in]  noalias
     * \param[out] res  \f$ A^p b \f$, where A is specified in the
     * constructor.
     */
    template<typename PlainObject, typename ResultType>
    void compute(const PlainObject& b, ResultType& res, RealScalar p);
    
    Index rows() const { return m_A.rows(); }
    Index cols() const { return m_A.cols(); }
};

template<typename MatrixType>
MatrixPower<MatrixType>::MatrixPower(const MatrixType& A) :
  m_A(A),
  m_init(false)
{ /* empty body */ }

template<typename MatrixType>
void MatrixPower<MatrixType>::compute(MatrixType& res, RealScalar p)
{
  switch (m_A.cols()) {
    case 0:
      break;
    case 1:
      res(0,0) = std::pow(m_A(0,0), p);
      break;
    default:
      RealScalar intpart;
      RealScalar x = modfAndInit(p, &intpart);
      res = MatrixType::Identity(m_A.rows(),m_A.cols());
      computeIntPower(res, intpart);
      computeFracPower(res, x);
  }
}

template<typename MatrixType>
template<typename PlainObject, typename ResultType>
void MatrixPower<MatrixType>::compute(const PlainObject& b, ResultType& res, RealScalar p)
{
  switch (m_A.cols()) {
    case 0:
      break;
    case 1:
      res = std::pow(m_A(0,0), p) * b;
      break;
    default:
      RealScalar intpart;
      RealScalar x = modfAndInit(p, &intpart);
      computeIntPower(b, res, intpart);
      computeFracPower(res, x);
  }
}

template<typename MatrixType>
typename MatrixType::RealScalar MatrixPower<MatrixType>::modfAndInit(RealScalar x, RealScalar* intpart)
{
  static RealScalar maxAbsEival, minAbsEival;
  *intpart = std::floor(x);
  RealScalar res = x - *intpart;

  if (!m_init && res) { // !init && res
    const ComplexSchur<MatrixType> schurOfA(m_A);
    m_T = schurOfA.matrixT();
    m_U = schurOfA.matrixU();
    m_init = true;

    const Array<RealScalar,EIGEN_SIZE_MIN_PREFER_FIXED(Rows,Cols),1,ColMajor,EIGEN_SIZE_MIN_PREFER_FIXED(MaxRows,MaxCols)>
      absTdiag = m_T.diagonal().array().abs();
    maxAbsEival = absTdiag.maxCoeff();
    minAbsEival = absTdiag.minCoeff();
  }

  if (res > RealScalar(0.5) && res > (1-res) * std::pow(maxAbsEival/minAbsEival, res)) {
    --res;
    ++*intpart;
  }
  return res;
}

template<typename MatrixType>
template<typename PlainObject, typename ResultType>
void MatrixPower<MatrixType>::apply(const PlainObject& b, ResultType& res, bool& init)
{
  if (init)
    res = m_tmp1 * res;
  else {
    init = true;
    res.noalias() = m_tmp1 * b;
  }
}

template<typename MatrixType>
template<typename ResultType>
void MatrixPower<MatrixType>::computeIntPower(ResultType& res, RealScalar p)
{
  RealScalar pp = std::abs(p);

  if (p<0)  m_tmp1 = m_A.inverse();
  else      m_tmp1 = m_A;

  while (pp >= 1) {
    if (std::fmod(pp, 2) >= 1)
      res = m_tmp1 * res;
    m_tmp1 *= m_tmp1;
    pp /= 2;
  }
}

template<typename MatrixType>
template<typename PlainObject, typename ResultType>
void MatrixPower<MatrixType>::computeIntPower(const PlainObject& b, ResultType& res, RealScalar p)
{
  if (b.cols() > m_A.cols()) {
    m_tmp2 = MatrixType::Identity(m_A.rows(),m_A.cols());
    computeIntPower(m_tmp2, p);
    res.noalias() = m_tmp2 * b;
  } else {
    RealScalar pp = std::abs(p);
    int cost = internal::binary_powering_cost(pp);
    bool init = false;

    if (p==0) {
      res = b;
      return;
    }
    if (p<0)  m_tmp1 = m_A.inverse();
    else      m_tmp1 = m_A;

    while (b.cols()*pp > m_A.cols()*cost) {
      if (std::fmod(pp, 2) >= 1) {
	apply(b, res, init);
	--cost;
      }
      m_tmp1 *= m_tmp1;
      --cost;
      pp /= 2;
    }
    for (; pp >= 1; --pp)
      apply(b, res, init);
  }
}

template<typename MatrixType>
template<typename ResultType>
void MatrixPower<MatrixType>::computeFracPower(ResultType& res, RealScalar p)
{
  if (p) {
    MatrixPowerTriangularAtomic<ComplexMatrix>(m_T).compute(m_fT, p);
    internal::recompose_complex_schur<NumTraits<Scalar>::IsComplex>::run(m_tmp1, m_fT, m_U);
    res = m_tmp1 * res;
  }
}

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
  public:
    typedef typename Derived::RealScalar RealScalar;
    typedef typename Derived::Index Index;

    /**
     * \brief Constructor.
     *
     * \param[in] A  %Matrix (expression), the base of the matrix power.
     * \param[in] p  scalar, the exponent of the matrix power.
     */
    MatrixPowerReturnValue(const Derived& A, RealScalar p)
    : m_A(A), m_p(p) { }

    /**
     * \brief Compute the matrix power.
     *
     * \param[out] result  \f$ A^p \f$ where \p A and \p p are as in the
     * constructor.
     */
    template<typename ResultType>
    inline void evalTo(ResultType& res) const
    { MatrixPower<typename Derived::PlainObject>(m_A).compute(res, m_p); }

    Index rows() const { return m_A.rows(); }
    Index cols() const { return m_A.cols(); }

  private:
    const Derived& m_A;
    const RealScalar m_p;
    MatrixPowerReturnValue& operator=(const MatrixPowerReturnValue&);
};

template<typename MatrixType>
class MatrixPowerReturnValue<MatrixPower<MatrixType> >
: public ReturnByValue<MatrixPowerReturnValue<MatrixPower<MatrixType> > >
{
  public:
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::Index Index;

    MatrixPowerReturnValue(MatrixPower<MatrixType>& ref, RealScalar p)
    : m_pow(ref), m_p(p) { }

    template<typename ResultType>
    inline void evalTo(ResultType& res) const
    { m_pow.compute(res, m_p); }

    Index rows() const { return m_pow.rows(); }
    Index cols() const { return m_pow.cols(); }

  private:
    MatrixPower<MatrixType>& m_pow;
    const RealScalar m_p;
    MatrixPowerReturnValue& operator=(const MatrixPowerReturnValue&);
};

namespace internal {
template<typename Derived>
struct traits<MatrixPowerReturnValue<Derived> >
{ typedef typename Derived::PlainObject ReturnType; };

template<typename MatrixType>
struct traits<MatrixPowerReturnValue<MatrixPower<MatrixType> > >
{ typedef MatrixType ReturnType; };

template<typename Derived>
struct traits<MatrixPowerProductBase<Derived> >
{ typedef typename traits<Derived>::ReturnType ReturnType; };
}

template<typename Derived>
const MatrixPowerReturnValue<Derived> MatrixBase<Derived>::pow(RealScalar p) const
{
  eigen_assert(rows() == cols());
  return MatrixPowerReturnValue<Derived>(derived(), p);
}

} // end namespace Eigen

#endif // EIGEN_MATRIX_POWER
