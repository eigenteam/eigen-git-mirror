
/// \returns as expression of \c *this with reshaped sizes.
///
/// \param nRows the number of rows in the reshaped expression, specified at either run-time or compile-time
/// \param nCols the number of columns in the reshaped expression, specified at either run-time or compile-time
/// \tparam NRowsType the type of the value handling the number of rows, typically Index.
/// \tparam NColsType the type of the value handling the number of columns, typically Index.
///
/// Dynamic size example: \include MatrixBase_reshaped_int_int.cpp
/// Output: \verbinclude MatrixBase_reshaped_int_int.out
///
/// The number of rows \a nRows and columns \a nCols can also be specified at compile-time by passing Eigen::fix<N>,
/// or Eigen::fix<N>(n) as arguments. In the later case, \c n plays the role of a runtime fallback value in case \c N equals Eigen::Dynamic.
/// Here is an example with a fixed number of rows and columns:
/// \include MatrixBase_reshaped_fixed.cpp
/// Output: \verbinclude MatrixBase_reshaped_fixed.out
///
/// \sa class Reshaped, fix, fix<N>(int)
///
#ifdef EIGEN_PARSED_BY_DOXYGEN
template<typename NRowsType, typename NColsType, typename OrderType>
EIGEN_DEVICE_FUNC
inline Reshaped<Derived,...>
reshaped(NRowsType nRows, NColsType nCols, OrderType = ColOrder);

/** This is the const version of reshaped(NRowsType,NColsType). */
template<typename NRowsType, typename NColsType, typename OrderType>
EIGEN_DEVICE_FUNC
inline const Reshaped<const Derived,...>
reshaped(NRowsType nRows, NColsType nCols, OrderType = ColOrder) const;
#else
template<typename NRowsType, typename NColsType>
EIGEN_DEVICE_FUNC
inline Reshaped<Derived,internal::get_fixed_value<NRowsType>::value,internal::get_fixed_value<NColsType>::value>
reshaped(NRowsType nRows, NColsType nCols)
{
  return Reshaped<Derived,internal::get_fixed_value<NRowsType>::value,internal::get_fixed_value<NColsType>::value>(
            derived(), internal::get_runtime_value(nRows), internal::get_runtime_value(nCols));
}

template<typename NRowsType, typename NColsType, typename OrderType>
EIGEN_DEVICE_FUNC
inline Reshaped<Derived,internal::get_fixed_value<NRowsType>::value,internal::get_fixed_value<NColsType>::value,
                OrderType::value==AutoOrderValue?Flags&RowMajorBit:OrderType::value>
reshaped(NRowsType nRows, NColsType nCols, OrderType)
{
  return Reshaped<Derived,internal::get_fixed_value<NRowsType>::value,internal::get_fixed_value<NColsType>::value,
                  OrderType::value==AutoOrderValue?Flags&RowMajorBit:OrderType::value>(
            derived(), internal::get_runtime_value(nRows), internal::get_runtime_value(nCols));
}


template<typename NRowsType, typename NColsType>
EIGEN_DEVICE_FUNC
inline const Reshaped<const Derived,internal::get_fixed_value<NRowsType>::value,internal::get_fixed_value<NColsType>::value>
reshaped(NRowsType nRows, NColsType nCols) const
{
  return Reshaped<const Derived,internal::get_fixed_value<NRowsType>::value,internal::get_fixed_value<NColsType>::value>(
            derived(), internal::get_runtime_value(nRows), internal::get_runtime_value(nCols));
}

template<typename NRowsType, typename NColsType, typename OrderType>
EIGEN_DEVICE_FUNC
inline const Reshaped<const Derived,internal::get_fixed_value<NRowsType>::value,internal::get_fixed_value<NColsType>::value,
                      OrderType::value==AutoOrderValue?Flags&RowMajorBit:OrderType::value>
reshaped(NRowsType nRows, NColsType nCols, OrderType) const
{
  return Reshaped<const Derived,internal::get_fixed_value<NRowsType>::value,internal::get_fixed_value<NColsType>::value,
                  OrderType::value==AutoOrderValue?Flags&RowMajorBit:OrderType::value>(
            derived(), internal::get_runtime_value(nRows), internal::get_runtime_value(nCols));
}

#endif // EIGEN_PARSED_BY_DOXYGEN
