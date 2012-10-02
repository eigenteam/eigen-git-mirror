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
 * This class is capable of computing upper triangular matrices raised
 * to an arbitrary real power.
 */
template<typename MatrixType>
class MatrixPowerTriangular : public MatrixPowerBase<MatrixPowerTriangular<MatrixType>,MatrixType>
{
  public:
    EIGEN_MATRIX_POWER_PUBLIC_INTERFACE(MatrixPowerTriangular)

    /**
     * \brief Constructor.
     *
     * \param[in] A  the base of the matrix power.
     *
     * The class stores a reference to A, so it should not be changed
     * (or destroyed) before evaluation.
     */
    explicit MatrixPowerTriangular(const MatrixType& A) : Base(A), m_T(Base::m_A)
    { }

  #ifdef EIGEN_PARSED_BY_DOXYGEN
    /**
     * \brief Returns the matrix power.
     *
     * \param[in] p  exponent, a real scalar.
     * \return The expression \f$ A^p \f$, where A is specified in the
     * constructor.
     */
    const MatrixPowerBaseReturnValue<MatrixPowerTriangular<MatrixType>,MatrixType> operator()(RealScalar p);
  #endif

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
     * \param[out] res  \f$ A^p b \f$, where A is specified in the
     * constructor.
     */
    template<typename Derived, typename ResultType>
    void compute(const Derived& b, ResultType& res, RealScalar p);

  private:
    EIGEN_MATRIX_POWER_PROTECTED_MEMBERS(MatrixPowerTriangular)

    const TriangularView<MatrixType,Upper> m_T;

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
void MatrixPowerTriangular<MatrixType>::compute(MatrixType& res, RealScalar p)
{
  switch (m_A.cols()) {
    case 0:
      break;
    case 1:
      res(0,0) = std::pow(m_T.coeff(0,0), p);
      break;
    default:
      RealScalar intpart, x = modfAndInit(p, &intpart);
      res = m_Id;
      computeIntPower(res, intpart);
      computeFracPower(res, x);
  }
}

template<typename MatrixType>
template<typename Derived, typename ResultType>
void MatrixPowerTriangular<MatrixType>::compute(const Derived& b, ResultType& res, RealScalar p)
{
  switch (m_A.cols()) {
    case 0:
      break;
    case 1:
      res = std::pow(m_T.coeff(0,0), p) * b;
      break;
    default:
      RealScalar intpart, x = modfAndInit(p, &intpart);
      computeIntPower(b, res, intpart);
      computeFracPower(res, x);
  }
}

template<typename MatrixType>
typename MatrixPowerTriangular<MatrixType>::Base::RealScalar
MatrixPowerTriangular<MatrixType>::modfAndInit(RealScalar x, RealScalar* intpart)
{
  *intpart = std::floor(x);
  RealScalar res = x - *intpart;

  if (!m_conditionNumber && res) {
    const RealArray absTdiag = m_A.diagonal().array().abs();
    m_conditionNumber = absTdiag.maxCoeff() / absTdiag.minCoeff();
  }

  if (res>RealScalar(0.5) && res>(1-res)*std::pow(m_conditionNumber,res)) {
    --res;
    ++*intpart;
  }
  return res;
}

template<typename MatrixType>
template<typename Derived, typename ResultType>
void MatrixPowerTriangular<MatrixType>::apply(const Derived& b, ResultType& res, bool& init)
{
  if (init)
    res = m_tmp1.template triangularView<Upper>() * res;
  else {
    init = true;
    res.noalias() = m_tmp1.template triangularView<Upper>() * b;
  }
}

template<typename MatrixType>
template<typename ResultType>
void MatrixPowerTriangular<MatrixType>::computeIntPower(ResultType& res, RealScalar p)
{
  RealScalar pp = std::abs(p);

  if (p<0)  m_tmp1 = m_T.solve(m_Id);
  else      m_tmp1 = m_T;

  while (pp >= 1) {
    if (std::fmod(pp, 2) >= 1)
      res = m_tmp1.template triangularView<Upper>() * res;
    m_tmp1 = m_tmp1.template triangularView<Upper>() * m_tmp1;
    pp /= 2;
  }
}

template<typename MatrixType>
template<typename Derived, typename ResultType>
void MatrixPowerTriangular<MatrixType>::computeIntPower(const Derived& b, ResultType& res, RealScalar p)
{
  if (b.cols() >= m_A.cols()) {
    m_tmp2 = m_Id;
    computeIntPower(m_tmp2, p);
    res.noalias() = m_tmp2.template triangularView<Upper>() * b;
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
      m_tmp1 = m_T;
    }
    else if (b.cols()*(pp-applyings) <= m_A.cols()*squarings) {
      res = m_T.solve(b);
      for (--pp; pp >= 1; --pp)
	res = m_T.solve(res);
      return;
    }
    else {
      m_tmp1 = m_T.solve(m_Id);
    }

    while (b.cols()*(pp-applyings) > m_A.cols()*squarings) {
      if (std::fmod(pp, 2) >= 1) {
	apply(b, res, init);
	--applyings;
      }
      m_tmp1 = m_tmp1.template triangularView<Upper>() * m_tmp1;
      --squarings;
      pp /= 2;
    }
    for (; pp >= 1; --pp)
      apply(b, res, init);
  }
}

template<typename MatrixType>
template<typename ResultType>
void MatrixPowerTriangular<MatrixType>::computeFracPower(ResultType& res, RealScalar p)
{
  if (p) {
    eigen_assert(m_conditionNumber);
    MatrixPowerTriangularAtomic<MatrixType>(m_A).compute(m_tmp1, p);
    res = m_tmp1.template triangularView<Upper>() * res;
  }
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
template<typename MatrixType>
class MatrixPower : public MatrixPowerBase<MatrixPower<MatrixType>,MatrixType>
{
  public:
    EIGEN_MATRIX_POWER_PUBLIC_INTERFACE(MatrixPower)

    /**
     * \brief Constructor.
     *
     * \param[in] A  the base of the matrix power.
     *
     * The class stores a reference to A, so it should not be changed
     * (or destroyed) before evaluation.
     */
    explicit MatrixPower(const MatrixType& A) : Base(A)
    { }

  #ifdef EIGEN_PARSED_BY_DOXYGEN
    /**
     * \brief Returns the matrix power.
     *
     * \param[in] p  exponent, a real scalar.
     * \return The expression \f$ A^p \f$, where A is specified in the
     * constructor.
     */
    const MatrixPowerBaseReturnValue<MatrixPower<MatrixType>,MatrixType> operator()(RealScalar p);
  #endif

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
     * \param[out] res  \f$ A^p b \f$, where A is specified in the
     * constructor.
     */
    template<typename Derived, typename ResultType>
    void compute(const Derived& b, ResultType& res, RealScalar p);

  private:
    EIGEN_MATRIX_POWER_PROTECTED_MEMBERS(MatrixPower)

    typedef Matrix<std::complex<RealScalar>,   RowsAtCompileTime,   ColsAtCompileTime,
				    Options,MaxRowsAtCompileTime,MaxColsAtCompileTime> ComplexMatrix;
    static const bool m_OKforLU = RowsAtCompileTime == Dynamic || RowsAtCompileTime > 4;    
    ComplexMatrix m_T, m_U, m_fT;

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
      res = m_Id;
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
typename MatrixPower<MatrixType>::RealScalar MatrixPower<MatrixType>::modfAndInit(RealScalar x, RealScalar* intpart)
{
  *intpart = std::floor(x);
  RealScalar res = x - *intpart;

  if (!m_conditionNumber && res) {
    const ComplexSchur<MatrixType> schurOfA(m_A);
    m_T = schurOfA.matrixT();
    m_U = schurOfA.matrixU();
    
    const RealArray absTdiag = m_T.diagonal().array().abs();
    m_conditionNumber = absTdiag.maxCoeff() / absTdiag.minCoeff();
  }

  if (res>RealScalar(0.5) && res>(1-res)*std::pow(m_conditionNumber, res)) {
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
    m_tmp2 = m_Id;
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
    else if (m_OKforLU && b.cols()*(pp-applyings) <= m_A.cols()*squarings) {
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
    eigen_assert(m_conditionNumber);
    MatrixPowerTriangularAtomic<ComplexMatrix>(m_T).compute(m_fT, p);
    internal::recompose_complex_schur<NumTraits<Scalar>::IsComplex>::run(m_tmp1, m_fT, m_U);
    res = m_tmp1 * res;
  }
}

namespace internal {

template<typename Derived>
struct traits<MatrixPowerReturnValue<Derived> >
{ typedef typename Derived::PlainObject ReturnType; };

} // namespace internal

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
    MatrixPowerReturnValue(const Derived& A, RealScalar p) : m_A(A), m_p(p)
    { }

    /**
     * \brief Compute the matrix power.
     *
     * \param[out] result  \f$ A^p \f$ where \p A and \p p are as in the
     * constructor.
     */
    template<typename ResultType>
    inline void evalTo(ResultType& res) const
    { MatrixPower<PlainObject>(m_A.eval()).compute(res, m_p); }

    /**
     * \brief Return the expression \f$ A^p b \f$.
     *
     * \p A and \p p are specified in the constructor.
     *
     * \param[in] b  the matrix (expression) to be applied.
     */
    template<typename OtherDerived>
    const MatrixPowerProduct<MatrixPower<PlainObject>,PlainObject,OtherDerived>
    operator*(const MatrixBase<OtherDerived>& b) const
    {
      MatrixPower<PlainObject> Apow(m_A.eval());
      return MatrixPowerProduct<MatrixPower<PlainObject>,PlainObject,OtherDerived>(Apow, b.derived(), m_p);
    }

    Index rows() const { return m_A.rows(); }
    Index cols() const { return m_A.cols(); }

  private:
    const Derived& m_A;
    const RealScalar m_p;
    MatrixPowerReturnValue& operator=(const MatrixPowerReturnValue&);
};

template<typename Derived>
const MatrixPowerReturnValue<Derived> MatrixBase<Derived>::pow(RealScalar p) const
{ return MatrixPowerReturnValue<Derived>(derived(), p); }

} // namespace Eigen

#endif // EIGEN_MATRIX_POWER
