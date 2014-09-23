

typedef CwiseUnaryOp<internal::scalar_abs_op<Scalar>, const Derived> AbsReturnType;
typedef CwiseUnaryOp<internal::scalar_abs2_op<Scalar>, const Derived> Abs2ReturnType;
typedef CwiseUnaryOp<internal::scalar_sqrt_op<Scalar>, const Derived> SqrtReturnType;
typedef CwiseUnaryOp<internal::scalar_inverse_op<Scalar>, const Derived> InverseReturnType;

typedef CwiseUnaryOp<internal::scalar_exp_op<Scalar>, const Derived> ExpReturnType;
typedef CwiseUnaryOp<internal::scalar_log_op<Scalar>, const Derived> LogReturnType;
typedef CwiseUnaryOp<internal::scalar_cos_op<Scalar>, const Derived> CosReturnType;
typedef CwiseUnaryOp<internal::scalar_sin_op<Scalar>, const Derived> SinReturnType;
typedef CwiseUnaryOp<internal::scalar_acos_op<Scalar>, const Derived> AcosReturnType;
typedef CwiseUnaryOp<internal::scalar_asin_op<Scalar>, const Derived> AsinReturnType;
typedef CwiseUnaryOp<internal::scalar_tan_op<Scalar>, const Derived> TanReturnType;
typedef CwiseUnaryOp<internal::scalar_atan_op<Scalar>, const Derived> AtanReturnType;
typedef CwiseUnaryOp<internal::scalar_pow_op<Scalar>, const Derived> PowReturnType;
typedef CwiseUnaryOp<internal::scalar_square_op<Scalar>, const Derived> SquareReturnType;
typedef CwiseUnaryOp<internal::scalar_cube_op<Scalar>, const Derived> CubeReturnType;

/** \returns an expression of the coefficient-wise absolute value of \c *this
  *
  * Example: \include Cwise_abs.cpp
  * Output: \verbinclude Cwise_abs.out
  *
  * \sa abs2()
  */
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE const AbsReturnType
abs() const
{
  return AbsReturnType(derived());
}

/** \returns an expression of the coefficient-wise squared absolute value of \c *this
  *
  * Example: \include Cwise_abs2.cpp
  * Output: \verbinclude Cwise_abs2.out
  *
  * \sa abs(), square()
  */
EIGEN_DEVICE_FUNC
EIGEN_STRONG_INLINE const Abs2ReturnType
abs2() const
{
  return Abs2ReturnType(derived());
}

/** \returns an expression of the coefficient-wise exponential of *this.
  *
  * This function computes the coefficient-wise exponential. The function MatrixBase::exp() in the
  * unsupported module MatrixFunctions computes the matrix exponential.
  *
  * Example: \include Cwise_exp.cpp
  * Output: \verbinclude Cwise_exp.out
  *
  * \sa pow(), log(), sin(), cos()
  */
EIGEN_DEVICE_FUNC
inline const ExpReturnType
exp() const
{
  return ExpReturnType(derived());
}

/** \returns an expression of the coefficient-wise logarithm of *this.
  *
  * This function computes the coefficient-wise logarithm. The function MatrixBase::log() in the
  * unsupported module MatrixFunctions computes the matrix logarithm.
  *
  * Example: \include Cwise_log.cpp
  * Output: \verbinclude Cwise_log.out
  *
  * \sa exp()
  */
EIGEN_DEVICE_FUNC
inline const LogReturnType
log() const
{
  return LogReturnType(derived());
}

/** \returns an expression of the coefficient-wise square root of *this.
  *
  * This function computes the coefficient-wise square root. The function MatrixBase::sqrt() in the
  * unsupported module MatrixFunctions computes the matrix square root.
  *
  * Example: \include Cwise_sqrt.cpp
  * Output: \verbinclude Cwise_sqrt.out
  *
  * \sa pow(), square()
  */
EIGEN_DEVICE_FUNC
inline const SqrtReturnType
sqrt() const
{
  return SqrtReturnType(derived());
}

/** \returns an expression of the coefficient-wise cosine of *this.
  *
  * This function computes the coefficient-wise cosine. The function MatrixBase::cos() in the
  * unsupported module MatrixFunctions computes the matrix cosine.
  *
  * Example: \include Cwise_cos.cpp
  * Output: \verbinclude Cwise_cos.out
  *
  * \sa sin(), acos()
  */
EIGEN_DEVICE_FUNC
inline const CosReturnType
cos() const
{
  return CosReturnType(derived());
}


/** \returns an expression of the coefficient-wise sine of *this.
  *
  * This function computes the coefficient-wise sine. The function MatrixBase::sin() in the
  * unsupported module MatrixFunctions computes the matrix sine.
  *
  * Example: \include Cwise_sin.cpp
  * Output: \verbinclude Cwise_sin.out
  *
  * \sa cos(), asin()
  */
EIGEN_DEVICE_FUNC
inline const SinReturnType
sin() const
{
  return SinReturnType(derived());
}

/** \returns an expression of the coefficient-wise arc cosine of *this.
  *
  * Example: \include Cwise_acos.cpp
  * Output: \verbinclude Cwise_acos.out
  *
  * \sa cos(), asin()
  */
EIGEN_DEVICE_FUNC
inline const AcosReturnType
acos() const
{
  return AcosReturnType(derived());
}

/** \returns an expression of the coefficient-wise arc sine of *this.
  *
  * Example: \include Cwise_asin.cpp
  * Output: \verbinclude Cwise_asin.out
  *
  * \sa sin(), acos()
  */
EIGEN_DEVICE_FUNC
inline const AsinReturnType
asin() const
{
  return AsinReturnType(derived());
}

/** \returns an expression of the coefficient-wise tan of *this.
  *
  * Example: \include Cwise_tan.cpp
  * Output: \verbinclude Cwise_tan.out
  *
  * \sa cos(), sin()
  */
EIGEN_DEVICE_FUNC
inline const TanReturnType
tan() const
{
  return TanReturnType(derived());
}

/** \returns an expression of the coefficient-wise arc tan of *this.
  *
  * Example: \include Cwise_atan.cpp
  * Output: \verbinclude Cwise_atan.out
  *
  * \sa cos(), sin(), tan()
  */
inline const AtanReturnType
atan() const
{
  return AtanReturnType(derived());
}

/** \returns an expression of the coefficient-wise power of *this to the given exponent.
  *
  * This function computes the coefficient-wise power. The function MatrixBase::pow() in the
  * unsupported module MatrixFunctions computes the matrix power.
  *
  * Example: \include Cwise_pow.cpp
  * Output: \verbinclude Cwise_pow.out
  *
  * \sa exp(), log()
  */
EIGEN_DEVICE_FUNC
inline const PowReturnType
pow(const Scalar& exponent) const
{
  return PowReturnType(derived(), internal::scalar_pow_op<Scalar>(exponent));
}


/** \returns an expression of the coefficient-wise inverse of *this.
  *
  * Example: \include Cwise_inverse.cpp
  * Output: \verbinclude Cwise_inverse.out
  *
  * \sa operator/(), operator*()
  */
EIGEN_DEVICE_FUNC
inline const InverseReturnType
inverse() const
{
  return InverseReturnType(derived());
}

/** \returns an expression of the coefficient-wise square of *this.
  *
  * Example: \include Cwise_square.cpp
  * Output: \verbinclude Cwise_square.out
  *
  * \sa operator/(), operator*(), abs2()
  */
EIGEN_DEVICE_FUNC
inline const SquareReturnType
square() const
{
  return SquareReturnType(derived());
}

/** \returns an expression of the coefficient-wise cube of *this.
  *
  * Example: \include Cwise_cube.cpp
  * Output: \verbinclude Cwise_cube.out
  *
  * \sa square(), pow()
  */
EIGEN_DEVICE_FUNC
inline const CubeReturnType
cube() const
{
  return CubeReturnType(derived());
}

#define EIGEN_MAKE_SCALAR_CWISE_UNARY_OP(METHOD_NAME,FUNCTOR) \
  EIGEN_DEVICE_FUNC \
  inline const CwiseUnaryOp<std::binder2nd<FUNCTOR<Scalar> >, const Derived> \
  METHOD_NAME(const Scalar& s) const { \
    return CwiseUnaryOp<std::binder2nd<FUNCTOR<Scalar> >, const Derived> \
            (derived(), std::bind2nd(FUNCTOR<Scalar>(), s)); \
  } \
  friend inline const CwiseUnaryOp<std::binder1st<FUNCTOR<Scalar> >, const Derived> \
  METHOD_NAME(const Scalar& s, const Derived& d) { \
	  return CwiseUnaryOp<std::binder1st<FUNCTOR<Scalar> >, const Derived> \
			  (d, std::bind1st(FUNCTOR<Scalar>(), s)); \
  }

EIGEN_MAKE_SCALAR_CWISE_UNARY_OP(operator==,  std::equal_to)
EIGEN_MAKE_SCALAR_CWISE_UNARY_OP(operator!=,  std::not_equal_to)
EIGEN_MAKE_SCALAR_CWISE_UNARY_OP(operator<,   std::less)
EIGEN_MAKE_SCALAR_CWISE_UNARY_OP(operator<=,  std::less_equal)
EIGEN_MAKE_SCALAR_CWISE_UNARY_OP(operator>,   std::greater)
EIGEN_MAKE_SCALAR_CWISE_UNARY_OP(operator>=,  std::greater_equal)


