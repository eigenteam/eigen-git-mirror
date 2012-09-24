// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_POWER_BASE
#define EIGEN_MATRIX_POWER_BASE

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

template<typename Derived, typename _Lhs, typename _Rhs>
struct traits<MatrixPowerProductBase<Derived,_Lhs,_Rhs> >
{
  typedef MatrixXpr XprKind;
  typedef typename remove_all<_Lhs>::type Lhs;
  typedef typename remove_all<_Rhs>::type Rhs;
  typedef typename remove_all<Derived>::type PlainObject;
  typedef typename scalar_product_traits<typename Lhs::Scalar, typename Rhs::Scalar>::ReturnType Scalar;
  typedef typename promote_storage_type<typename traits<Lhs>::StorageKind,
					typename traits<Rhs>::StorageKind>::ret StorageKind;
  typedef typename promote_index_type<typename traits<Lhs>::Index,
				      typename traits<Rhs>::Index>::type Index;

  enum {
    RowsAtCompileTime = traits<Lhs>::RowsAtCompileTime,
    ColsAtCompileTime = traits<Rhs>::ColsAtCompileTime,
    MaxRowsAtCompileTime = traits<Lhs>::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = traits<Rhs>::MaxColsAtCompileTime,
    Flags = (MaxRowsAtCompileTime==1 ? RowMajorBit : 0)
	  | EvalBeforeNestingBit | EvalBeforeAssigningBit | NestByRefBit,
    CoeffReadCost = 0
  };
};

template<typename T>
inline int binary_powering_cost(T p, int* squarings)
{
  int applyings=0, tmp;

  frexp(p, squarings);
  --*squarings;

  while (std::frexp(p, &tmp), tmp > 0) {
    p -= std::ldexp(static_cast<T>(0.5), tmp);
    ++applyings;
  }
  return applyings;
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

template<typename MatrixType, int UpLo=Upper> class MatrixPowerTriangularAtomic
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
  res = (p-degree) / ((i-1)<<1) * IminusT;
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
    }
    else if (2*abs(m_T(i-1,i-1)) < abs(m_T(i,i)) || 2*abs(m_T(i,i)) < abs(m_T(i-1,i-1))) {
      res(i-1,i) = m_T(i-1,i) * (res(i,i)-res(i-1,i-1)) / (m_T(i,i)-m_T(i-1,i-1));
    }
    else {
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
  MatrixType IminusT, sqrtT, T=m_T;
  RealScalar normIminusT;
  int degree, degree2, numberOfSquareRoots=0, numberOfExtraSquareRoots=0;

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

#define	EIGEN_MATRIX_POWER_PRODUCT_PUBLIC_INTERFACE(Derived) \
  typedef MatrixPowerProductBase<Derived, Lhs, Rhs > Base; \
  EIGEN_DENSE_PUBLIC_INTERFACE(Derived)

template<typename Derived, typename Lhs, typename Rhs>
class MatrixPowerProductBase : public MatrixBase<Derived>
{
  public:
    typedef MatrixBase<Derived> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(MatrixPowerProductBase)

    inline Index rows() const { return derived().rows(); }
    inline Index cols() const { return derived().cols(); }

    template<typename ResultType>
    inline void evalTo(ResultType& res) const { derived().evalTo(res); }
};

template<typename Derived>
template<typename ProductDerived, typename Lhs, typename Rhs>
Derived& MatrixBase<Derived>::lazyAssign(const MatrixPowerProductBase<ProductDerived,Lhs,Rhs>& other)
{
  other.derived().evalTo(derived());
  return derived();
}

} // namespace Eigen

#endif // EIGEN_MATRIX_POWER
