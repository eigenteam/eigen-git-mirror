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
template<typename MatrixType>
class MatrixPower : public MatrixPowerBase<MatrixPower<MatrixType>,MatrixType>
{
  public:
    EIGEN_MATRIX_POWER_PUBLIC_INTERFACE(MatrixPower)

    /**
     * \brief Constructor.
     *
     * \param[in] A  the base of the matrix power.
     */
    template<typename MatrixExpression>
    explicit MatrixPower(const MatrixExpression& A);

    /**
     * \brief Return the expression \f$ A^p \f$.
     *
     * \param[in] p  exponent, a real scalar.
     */
    const MatrixPowerReturnValue<MatrixType> operator()(RealScalar p)
    { return MatrixPowerReturnValue<MatrixType>(*this, p); }

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
    template<typename Derived, typename ResultType>
    void compute(const Derived& b, ResultType& res, RealScalar p);

  private:
    using Base::m_A;
    MatrixType m_tmp1, m_tmp2;
    ComplexMatrix m_T, m_U, m_fT;
    bool m_init;

    RealScalar modfAndInit(RealScalar, RealScalar*);

    template<typename Derived, typename ResultType>
    void apply(const Derived&, ResultType&, bool&);

    template<typename ResultType>
    void computeIntPower(ResultType&, RealScalar);

    template<typename Derived, typename ResultType>
    void computeIntPower(const Derived&, ResultType&, RealScalar);

    template<typename ResultType>
    void computeFracPower(ResultType&, RealScalar);
};

template<typename MatrixType>
template<typename MatrixExpression>
MatrixPower<MatrixType>::MatrixPower(const MatrixExpression& A) :
  Base(A),
  m_init(false)
{ /* empty body */ }

template<typename MatrixType>
void MatrixPower<MatrixType>::compute(MatrixType& res, RealScalar p)
{
  switch (m_A.cols()) {
    case 0:
      break;
    case 1:
      res(0,0) = std::pow(m_A.coeff(0,0), p);
      break;
    default:
      RealScalar intpart, x = modfAndInit(p, &intpart);
      res = MatrixType::Identity(m_A.rows(), m_A.cols());
      computeIntPower(res, intpart);
      computeFracPower(res, x);
  }
}

template<typename MatrixType>
template<typename Derived, typename ResultType>
void MatrixPower<MatrixType>::compute(const Derived& b, ResultType& res, RealScalar p)
{
  switch (m_A.cols()) {
    case 0:
      break;
    case 1:
      res = std::pow(m_A.coeff(0,0), p) * b;
      break;
    default:
      RealScalar intpart, x = modfAndInit(p, &intpart);
      computeIntPower(b, res, intpart);
      computeFracPower(res, x);
  }
}

template<typename MatrixType>
typename MatrixPower<MatrixType>::Base::RealScalar MatrixPower<MatrixType>::modfAndInit(RealScalar x, RealScalar* intpart)
{
  static RealScalar maxAbsEival, minAbsEival;
  *intpart = std::floor(x);
  RealScalar res = x - *intpart;

  if (!m_init && res) {
    const ComplexSchur<MatrixType> schurOfA(m_A);
    m_T = schurOfA.matrixT();
    m_U = schurOfA.matrixU();
    m_init = true;

    const RealArray absTdiag = m_T.diagonal().array().abs();
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
template<typename Derived, typename ResultType>
void MatrixPower<MatrixType>::apply(const Derived& b, ResultType& res, bool& init)
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
template<typename Derived, typename ResultType>
void MatrixPower<MatrixType>::computeIntPower(const Derived& b, ResultType& res, RealScalar p)
{
  if (b.cols() >= m_A.cols()) {
    m_tmp2 = MatrixType::Identity(m_A.rows(), m_A.cols());
    computeIntPower(m_tmp2, p);
    res.noalias() = m_tmp2 * b;
  }
  else {
    RealScalar pp = std::abs(p);
    int squarings, applyings = internal::binary_powering_cost(pp, &squarings);
    bool init = false;

    if (p==0) {
      res = b;
      return;
    }
    else if (p>0) {
      m_tmp1 = m_A;
    }
    else if (m_A.cols() > 2 && b.cols()*(pp-applyings) <= m_A.cols()*squarings) {
      PartialPivLU<MatrixType> A(m_A);
      res = A.solve(b);
      for (--pp; pp >= 1; --pp)
	res = A.solve(res);
      return;
    }
    else {
      m_tmp1 = m_A.inverse();
    }

    while (b.cols()*(pp-applyings) > m_A.cols()*squarings) {
      if (std::fmod(pp, 2) >= 1) {
	apply(b, res, init);
	--applyings;
      }
      m_tmp1 *= m_tmp1;
      --squarings;
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

template<typename Lhs, typename Rhs>
class MatrixPowerMatrixProduct : public MatrixPowerProductBase<MatrixPowerMatrixProduct<Lhs,Rhs>,Lhs,Rhs>
{
  public:
    EIGEN_MATRIX_POWER_PRODUCT_PUBLIC_INTERFACE(MatrixPowerMatrixProduct)

    MatrixPowerMatrixProduct(MatrixPower<Lhs>& pow, const Rhs& b, RealScalar p)
    : m_pow(pow), m_b(b), m_p(p) { }

    template<typename ResultType>
    inline void evalTo(ResultType& res) const
    { m_pow.compute(m_b, res, m_p); }

    Index rows() const { return m_b.rows(); }
    Index cols() const { return m_b.cols(); }

  private:
    MatrixPower<Lhs>& m_pow;
    const Rhs& m_b;
    const RealScalar m_p;
    MatrixPowerMatrixProduct& operator=(const MatrixPowerMatrixProduct&);
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
  public:
    typedef typename Derived::PlainObject PlainObject;
    typedef typename Derived::RealScalar RealScalar;
    typedef typename Derived::Index Index;

    /**
     * \brief Constructor.
     *
     * \param[in] A  %Matrix (expression), the base of the matrix power.
     * \param[in] p  scalar, the exponent of the matrix power.
     */
    MatrixPowerReturnValue(const Derived& A, RealScalar p)
    : m_pow(*new MatrixPower<PlainObject>(A)), m_p(p), m_del(true) { }

    MatrixPowerReturnValue(MatrixPower<PlainObject>& pow, RealScalar p)
    : m_pow(pow), m_p(p), m_del(false) { }

    ~MatrixPowerReturnValue()
    { if (m_del)  delete &m_pow; }

    /**
     * \brief Compute the matrix power.
     *
     * \param[out] result  \f$ A^p \f$ where \p A and \p p are as in the
     * constructor.
     */
    template<typename ResultType>
    inline void evalTo(ResultType& res) const
    { m_pow.compute(res, m_p); }

    template<typename OtherDerived>
    const MatrixPowerMatrixProduct<PlainObject,OtherDerived> operator*(const MatrixBase<OtherDerived>& b) const
    { return MatrixPowerMatrixProduct<PlainObject,OtherDerived>(m_pow, b.derived(), m_p); }

    Index rows() const { return m_pow.rows(); }
    Index cols() const { return m_pow.cols(); }

  private:
    MatrixPower<PlainObject>& m_pow;
    const RealScalar m_p;
    const bool m_del;  // whether to delete the pointer at destruction
    MatrixPowerReturnValue& operator=(const MatrixPowerReturnValue&);
};

namespace internal {
template<typename MatrixType, typename Derived>
struct nested<MatrixPowerMatrixProduct<MatrixType,Derived> >
{ typedef typename MatrixPowerMatrixProduct<MatrixType,Derived>::PlainObject const& type; };

template<typename Derived>
struct traits<MatrixPowerReturnValue<Derived> >
{ typedef typename Derived::PlainObject ReturnType; };

template<typename Lhs, typename Rhs>
struct traits<MatrixPowerMatrixProduct<Lhs,Rhs> >
: traits<MatrixPowerProductBase<MatrixPowerMatrixProduct<Lhs,Rhs>,Lhs,Rhs> >
{ };
}

template<typename Derived>
const MatrixPowerReturnValue<Derived> MatrixBase<Derived>::pow(RealScalar p) const
{
  eigen_assert(rows() == cols());
  return MatrixPowerReturnValue<Derived>(derived(), p);
}

} // namespace Eigen

#endif // EIGEN_MATRIX_POWER
