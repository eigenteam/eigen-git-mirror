
/** \returns an expression of the coefficient-wise \< operator of *this and \a other
  *
  * Example: \include Cwise_less.cpp
  * Output: \verbinclude Cwise_less.out
  *
  * \sa all(), any(), operator>(), operator<=()
  */
EIGEN_MAKE_CWISE_BINARY_OP(operator<,std::less)

/** \returns an expression of the coefficient-wise \<= operator of *this and \a other
  *
  * Example: \include Cwise_less_equal.cpp
  * Output: \verbinclude Cwise_less_equal.out
  *
  * \sa all(), any(), operator>=(), operator<()
  */
EIGEN_MAKE_CWISE_BINARY_OP(operator<=,std::less_equal)
// template<typename ExpressionType>
// template<typename OtherDerived>
// inline const EIGEN_CWISE_BINOP_RETURN_TYPE(std::less_equal)
// operator<=(const MatrixBase<OtherDerived> &other) const
// {
//   return EIGEN_CWISE_BINOP_RETURN_TYPE(std::less_equal)(_expression(), other.derived());
// }

/** \returns an expression of the coefficient-wise \> operator of *this and \a other
  *
  * Example: \include Cwise_greater.cpp
  * Output: \verbinclude Cwise_greater.out
  *
  * \sa all(), any(), operator>=(), operator<()
  */
EIGEN_MAKE_CWISE_BINARY_OP(operator>,std::greater)
// template<typename ExpressionType>
// template<typename OtherDerived>
// inline const EIGEN_CWISE_BINOP_RETURN_TYPE(std::greater)
// operator>(const MatrixBase<OtherDerived> &other) const
// {
//   return EIGEN_CWISE_BINOP_RETURN_TYPE(std::greater)(_expression(), other.derived());
// }

/** \returns an expression of the coefficient-wise \>= operator of *this and \a other
  *
  * Example: \include Cwise_greater_equal.cpp
  * Output: \verbinclude Cwise_greater_equal.out
  *
  * \sa all(), any(), operator>(), operator<=()
  */
EIGEN_MAKE_CWISE_BINARY_OP(operator>=,std::greater_equal)
// template<typename ExpressionType>
// template<typename OtherDerived>
// inline const EIGEN_CWISE_BINOP_RETURN_TYPE(std::greater_equal)
// operator>=(const MatrixBase<OtherDerived> &other) const
// {
//   return EIGEN_CWISE_BINOP_RETURN_TYPE(std::greater_equal)(_expression(), other.derived());
// }

/** \returns an expression of the coefficient-wise == operator of *this and \a other
  *
  * \warning this performs an exact comparison, which is generally a bad idea with floating-point types.
  * In order to check for equality between two vectors or matrices with floating-point coefficients, it is
  * generally a far better idea to use a fuzzy comparison as provided by isApprox() and
  * isMuchSmallerThan().
  *
  * Example: \include Cwise_equal_equal.cpp
  * Output: \verbinclude Cwise_equal_equal.out
  *
  * \sa all(), any(), isApprox(), isMuchSmallerThan()
  */
EIGEN_MAKE_CWISE_BINARY_OP(operator==,std::equal_to)
// template<typename ExpressionType>
// template<typename OtherDerived>
// inline const EIGEN_CWISE_BINOP_RETURN_TYPE(std::equal_to)
// operator==(const MatrixBase<OtherDerived> &other) const
// {
//   return EIGEN_CWISE_BINOP_RETURN_TYPE(std::equal_to)(_expression(), other.derived());
// }

/** \returns an expression of the coefficient-wise != operator of *this and \a other
  *
  * \warning this performs an exact comparison, which is generally a bad idea with floating-point types.
  * In order to check for equality between two vectors or matrices with floating-point coefficients, it is
  * generally a far better idea to use a fuzzy comparison as provided by isApprox() and
  * isMuchSmallerThan().
  *
  * Example: \include Cwise_not_equal.cpp
  * Output: \verbinclude Cwise_not_equal.out
  *
  * \sa all(), any(), isApprox(), isMuchSmallerThan()
  */
EIGEN_MAKE_CWISE_BINARY_OP(operator!=,std::not_equal_to)
// template<typename ExpressionType>
// template<typename OtherDerived>
// inline const EIGEN_CWISE_BINOP_RETURN_TYPE(std::not_equal_to)
// operator!=(const MatrixBase<OtherDerived> &other) const
// {
//   return EIGEN_CWISE_BINOP_RETURN_TYPE(std::not_equal_to)(_expression(), other.derived());
// }

// comparisons to scalar value

#if 0

/** \returns an expression of the coefficient-wise \< operator of *this and a scalar \a s
  *
  * \sa operator<(const MatrixBase<OtherDerived> &) const
  */
template<typename ExpressionType>
inline const EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::less)
operator<(Scalar s) const
{
  return EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::less)(_expression(),
            typename ExpressionType::ConstantReturnType(_expression().rows(), _expression().cols(), s));
}

/** \returns an expression of the coefficient-wise \<= operator of *this and a scalar \a s
  *
  * \sa operator<=(const MatrixBase<OtherDerived> &) const
  */
template<typename ExpressionType>
inline const EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::less_equal)
operator<=(Scalar s) const
{
  return EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::less_equal)(_expression(),
            typename ExpressionType::ConstantReturnType(_expression().rows(), _expression().cols(), s));
}

/** \returns an expression of the coefficient-wise \> operator of *this and a scalar \a s
  *
  * \sa operator>(const MatrixBase<OtherDerived> &) const
  */
template<typename ExpressionType>
inline const EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::greater)
operator>(Scalar s) const
{
  return EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::greater)(_expression(),
            typename ExpressionType::ConstantReturnType(_expression().rows(), _expression().cols(), s));
}

/** \returns an expression of the coefficient-wise \>= operator of *this and a scalar \a s
  *
  * \sa operator>=(const MatrixBase<OtherDerived> &) const
  */
template<typename ExpressionType>
inline const EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::greater_equal)
operator>=(Scalar s) const
{
  return EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::greater_equal)(_expression(),
            typename ExpressionType::ConstantReturnType(_expression().rows(), _expression().cols(), s));
}

/** \returns an expression of the coefficient-wise == operator of *this and a scalar \a s
  *
  * \warning this performs an exact comparison, which is generally a bad idea with floating-point types.
  * In order to check for equality between two vectors or matrices with floating-point coefficients, it is
  * generally a far better idea to use a fuzzy comparison as provided by isApprox() and
  * isMuchSmallerThan().
  *
  * \sa operator==(const MatrixBase<OtherDerived> &) const
  */
template<typename ExpressionType>
inline const EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::equal_to)
operator==(Scalar s) const
{
  return EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::equal_to)(_expression(),
            typename ExpressionType::ConstantReturnType(_expression().rows(), _expression().cols(), s));
}

/** \returns an expression of the coefficient-wise != operator of *this and a scalar \a s
  *
  * \warning this performs an exact comparison, which is generally a bad idea with floating-point types.
  * In order to check for equality between two vectors or matrices with floating-point coefficients, it is
  * generally a far better idea to use a fuzzy comparison as provided by isApprox() and
  * isMuchSmallerThan().
  *
  * \sa operator!=(const MatrixBase<OtherDerived> &) const
  */
template<typename ExpressionType>
inline const EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::not_equal_to)
operator!=(Scalar s) const
{
  return EIGEN_CWISE_COMP_TO_SCALAR_RETURN_TYPE(std::not_equal_to)(_expression(),
            typename ExpressionType::ConstantReturnType(_expression().rows(), _expression().cols(), s));
}

#endif

// scalar addition

/** \returns an expression of \c *this with each coeff incremented by the constant \a scalar
  *
  * Example: \include Cwise_plus.cpp
  * Output: \verbinclude Cwise_plus.out
  *
  * \sa operator+=(), operator-()
  */
inline const CwiseUnaryOp<ei_scalar_add_op<Scalar>, Derived>
operator+(const Scalar& scalar) const
{
  return CwiseUnaryOp<ei_scalar_add_op<Scalar>, Derived>(derived(), ei_scalar_add_op<Scalar>(scalar));
}

/** Adds the given \a scalar to each coeff of this expression.
  *
  * Example: \include Cwise_plus_equal.cpp
  * Output: \verbinclude Cwise_plus_equal.out
  *
  * \sa operator+(), operator-=()
  */
// template<typename ExpressionType>
// inline ExpressionType& operator+=(const Scalar& scalar)
// {
//   return m_matrix.const_cast_derived() = *this + scalar;
// }

/** \returns an expression of \c *this with each coeff decremented by the constant \a scalar
  *
  * Example: \include Cwise_minus.cpp
  * Output: \verbinclude Cwise_minus.out
  *
  * \sa operator+(), operator-=()
  */
// template<typename ExpressionType>
// inline const typename ScalarAddReturnType
// operator-(const Scalar& scalar) const
// {
//   return *this + (-scalar);
// }

/** Substracts the given \a scalar from each coeff of this expression.
  *
  * Example: \include Cwise_minus_equal.cpp
  * Output: \verbinclude Cwise_minus_equal.out
  *
  * \sa operator+=(), operator-()
  */

// template<typename ExpressionType>
// inline ExpressionType& operator-=(const Scalar& scalar)
// {
//   return m_matrix.const_cast_derived() = *this - scalar;
// }
