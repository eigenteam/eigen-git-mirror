/** \returns an expression of the difference of \c *this and \a other
  *
  * \note If you want to substract a given scalar from all coefficients, see Cwise::operator-().
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator-=()
  */
template<typename OtherDerived>
EIGEN_STRONG_INLINE const CwiseBinaryOp<ei_scalar_difference_op<typename ei_traits<Derived>::Scalar>,
                                 Derived, OtherDerived>
operator-(const EIGEN_CURRENT_STORAGE_BASE_CLASS<OtherDerived> &other) const
{
  return CwiseBinaryOp<ei_scalar_difference_op<Scalar>,
                       Derived, OtherDerived>(derived(), other.derived());
}

/** \returns an expression of the sum of \c *this and \a other
  *
  * \note If you want to add a given scalar to all coefficients, see Cwise::operator+().
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator+=()
  */
template<typename OtherDerived>
EIGEN_STRONG_INLINE const CwiseBinaryOp<ei_scalar_sum_op<typename ei_traits<Derived>::Scalar>, Derived, OtherDerived>
operator+(const EIGEN_CURRENT_STORAGE_BASE_CLASS<OtherDerived> &other) const
{
  return CwiseBinaryOp<ei_scalar_sum_op<Scalar>, Derived, OtherDerived>(derived(), other.derived());
}

/** \returns an expression of a custom coefficient-wise operator \a func of *this and \a other
  *
  * The template parameter \a CustomBinaryOp is the type of the functor
  * of the custom operator (see class CwiseBinaryOp for an example)
  *
  * Here is an example illustrating the use of custom functors:
  * \include class_CwiseBinaryOp.cpp
  * Output: \verbinclude class_CwiseBinaryOp.out
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator+, MatrixBase::operator-, MatrixBase::cwiseProduct
  */
template<typename CustomBinaryOp, typename OtherDerived>
EIGEN_STRONG_INLINE const CwiseBinaryOp<CustomBinaryOp, Derived, OtherDerived>
binaryExpr(const EIGEN_CURRENT_STORAGE_BASE_CLASS<OtherDerived> &other, const CustomBinaryOp& func = CustomBinaryOp()) const
{
  return CwiseBinaryOp<CustomBinaryOp, Derived, OtherDerived>(derived(), other.derived(), func);
}

/** \returns an expression of the Schur product (coefficient wise product) of *this and \a other
  *
  * Example: \include MatrixBase_cwiseProduct.cpp
  * Output: \verbinclude MatrixBase_cwiseProduct.out
  *
  * \sa class CwiseBinaryOp, cwiseAbs2
  */
template<typename OtherDerived>
EIGEN_STRONG_INLINE const CwiseBinaryOp<ei_scalar_product_op<Scalar>, Derived, OtherDerived>
cwiseProduct(const EIGEN_CURRENT_STORAGE_BASE_CLASS<OtherDerived> &other) const
{
  return CwiseBinaryOp<ei_scalar_product_op<Scalar>, Derived, OtherDerived>(derived(), other.derived());
}

/** \returns an expression of the coefficient-wise == operator of *this and \a other
  *
  * \warning this performs an exact comparison, which is generally a bad idea with floating-point types.
  * In order to check for equality between two vectors or matrices with floating-point coefficients, it is
  * generally a far better idea to use a fuzzy comparison as provided by MatrixBase::isApprox() and
  * MatrixBase::isMuchSmallerThan().
  *
  * Example: \include MatrixBase_cwiseEqual.cpp
  * Output: \verbinclude MatrixBase_cwiseEqual.out
  *
  * \sa MatrixBase::cwiseNotEqual(), MatrixBase::isApprox(), MatrixBase::isMuchSmallerThan()
  */
template<typename OtherDerived>
inline const CwiseBinaryOp<std::equal_to<Scalar>, Derived, OtherDerived>
cwiseEqual(const EIGEN_CURRENT_STORAGE_BASE_CLASS<OtherDerived> &other) const
{
  return CwiseBinaryOp<std::equal_to<Scalar>, Derived, OtherDerived>(derived(), other.derived());
}

/** \returns an expression of the coefficient-wise != operator of *this and \a other
  *
  * \warning this performs an exact comparison, which is generally a bad idea with floating-point types.
  * In order to check for equality between two vectors or matrices with floating-point coefficients, it is
  * generally a far better idea to use a fuzzy comparison as provided by MatrixBase::isApprox() and
  * MatrixBase::isMuchSmallerThan().
  *
  * Example: \include MatrixBase_cwiseNotEqual.cpp
  * Output: \verbinclude MatrixBase_cwiseNotEqual.out
  *
  * \sa MatrixBase::cwiseEqual(), MatrixBase::isApprox(), MatrixBase::isMuchSmallerThan()
  */
template<typename OtherDerived>
inline const CwiseBinaryOp<std::not_equal_to<Scalar>, Derived, OtherDerived>
cwiseNotEqual(const EIGEN_CURRENT_STORAGE_BASE_CLASS<OtherDerived> &other) const
{
  return CwiseBinaryOp<std::not_equal_to<Scalar>, Derived, OtherDerived>(derived(), other.derived());
}

/** \returns an expression of the coefficient-wise min of *this and \a other
  *
  * Example: \include MatrixBase_cwiseMin.cpp
  * Output: \verbinclude MatrixBase_cwiseMin.out
  *
  * \sa class CwiseBinaryOp, max()
  */
template<typename OtherDerived>
EIGEN_STRONG_INLINE const CwiseBinaryOp<ei_scalar_min_op<Scalar>, Derived, OtherDerived>
cwiseMin(const EIGEN_CURRENT_STORAGE_BASE_CLASS<OtherDerived> &other) const
{
  return CwiseBinaryOp<ei_scalar_min_op<Scalar>, Derived, OtherDerived>(derived(), other.derived());
}

/** \returns an expression of the coefficient-wise max of *this and \a other
  *
  * Example: \include MatrixBase_cwiseMax.cpp
  * Output: \verbinclude MatrixBase_cwiseMax.out
  *
  * \sa class CwiseBinaryOp, min()
  */
template<typename OtherDerived>
EIGEN_STRONG_INLINE const CwiseBinaryOp<ei_scalar_max_op<Scalar>, Derived, OtherDerived>
cwiseMax(const EIGEN_CURRENT_STORAGE_BASE_CLASS<OtherDerived> &other) const
{
  return CwiseBinaryOp<ei_scalar_max_op<Scalar>, Derived, OtherDerived>(derived(), other.derived());
}

/** \returns an expression of the coefficient-wise quotient of *this and \a other
  *
  * Example: \include MatrixBase_cwiseQuotient.cpp
  * Output: \verbinclude MatrixBase_cwiseQuotient.out
  *
  * \sa class CwiseBinaryOp, cwiseProduct(), cwiseInverse()
  */
template<typename OtherDerived>
EIGEN_STRONG_INLINE const CwiseBinaryOp<ei_scalar_quotient_op<Scalar>, Derived, OtherDerived>
cwiseQuotient(const EIGEN_CURRENT_STORAGE_BASE_CLASS<OtherDerived> &other) const
{
  return CwiseBinaryOp<ei_scalar_quotient_op<Scalar>, Derived, OtherDerived>(derived(), other.derived());
}
