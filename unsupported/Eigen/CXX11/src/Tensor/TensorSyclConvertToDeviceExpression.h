// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*****************************************************************
 * TensorSyclConvertToDeviceExpression.h
 *
 * \brief:
 *  Conversion from host pointer to device pointer
 *  inside leaf nodes of the expression.
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_TENSORSYCL_CONVERT_TO_DEVICE_EXPRESSION_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_TENSORSYCL_CONVERT_TO_DEVICE_EXPRESSION_HPP

namespace Eigen {
namespace TensorSycl {
namespace internal {
/// \struct ConvertToDeviceExpression
/// \brief This struct is used to convert the MakePointer in the host expression
/// to the MakeGlobalPointer for the device expression. For the leafNodes
/// containing the pointer. This is due to the fact that the address space of
/// the pointer T* is different on the host and the device.
template <typename Expr>
struct ConvertToDeviceExpression;

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is TensorMap
template <typename Scalar_, int Options_, int Options2_, int NumIndices_,
          typename IndexType_, template <class> class MakePointer_>
struct ConvertToDeviceExpression<
    TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>, Options2_,
              MakePointer_>> {
  using Type = TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>,
                         Options2_, MakeGlobalPointer>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is const TensorMap
template <typename Scalar_, int Options_, int Options2_, int NumIndices_,
          typename IndexType_, template <class> class MakePointer_>
struct ConvertToDeviceExpression<
    const TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>,
                    Options2_, MakePointer_>> {
  using Type =
      const TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>,
                      Options2_, MakeGlobalPointer>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is const TensorCwiseNullaryOp
template <typename OP, typename RHSExpr>
struct ConvertToDeviceExpression<const TensorCwiseNullaryOp<OP, RHSExpr>> {
  using RHSPlaceHolderType = typename ConvertToDeviceExpression<RHSExpr>::Type;
  using Type = const TensorCwiseNullaryOp<OP, RHSPlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is TensorCwiseNullaryOp
template <typename OP, typename RHSExpr>
struct ConvertToDeviceExpression<TensorCwiseNullaryOp<OP, RHSExpr>> {
  using RHSPlaceHolderType = typename ConvertToDeviceExpression<RHSExpr>::Type;
  using Type = TensorCwiseNullaryOp<OP, RHSPlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is const TensorBroadcastingOp
template <typename OP, typename RHSExpr>
struct ConvertToDeviceExpression<const TensorBroadcastingOp<OP, RHSExpr>> {
  using RHSPlaceHolderType = typename ConvertToDeviceExpression<RHSExpr>::Type;
  using Type = const TensorBroadcastingOp<OP, RHSPlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is TensorBroadcastingOp
template <typename OP, typename RHSExpr>
struct ConvertToDeviceExpression<TensorBroadcastingOp<OP, RHSExpr>> {
  using RHSPlaceHolderType = typename ConvertToDeviceExpression<RHSExpr>::Type;
  using Type = TensorBroadcastingOp<OP, RHSPlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is const TensorCwiseUnaryOp
template <typename OP, typename RHSExpr>
struct ConvertToDeviceExpression<const TensorCwiseUnaryOp<OP, RHSExpr>> {
  using RHSPlaceHolderType = typename ConvertToDeviceExpression<RHSExpr>::Type;
  using Type = const TensorCwiseUnaryOp<OP, RHSPlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is TensorCwiseUnaryOp
template <typename OP, typename RHSExpr>
struct ConvertToDeviceExpression<TensorCwiseUnaryOp<OP, RHSExpr>> {
  using RHSPlaceHolderType = typename ConvertToDeviceExpression<RHSExpr>::Type;
  using Type = TensorCwiseUnaryOp<OP, RHSPlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is const TensorCwiseBinaryOp
template <typename OP, typename LHSExpr, typename RHSExpr>
struct ConvertToDeviceExpression<
    const TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>> {
  using LHSPlaceHolderType = typename ConvertToDeviceExpression<LHSExpr>::Type;
  using RHSPlaceHolderType = typename ConvertToDeviceExpression<RHSExpr>::Type;
  using Type =
      const TensorCwiseBinaryOp<OP, LHSPlaceHolderType, RHSPlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is TensorCwiseBinaryOp
template <typename OP, typename LHSExpr, typename RHSExpr>
struct ConvertToDeviceExpression<TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>> {
  using LHSPlaceHolderType = typename ConvertToDeviceExpression<LHSExpr>::Type;
  using RHSPlaceHolderType = typename ConvertToDeviceExpression<RHSExpr>::Type;
  using Type = TensorCwiseBinaryOp<OP, LHSPlaceHolderType, RHSPlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is const TensorCwiseTernaryOp
template <typename OP, typename Arg1Impl, typename Arg2Impl, typename Arg3Impl>
struct ConvertToDeviceExpression<
    const TensorCwiseTernaryOp<OP, Arg1Impl, Arg2Impl, Arg3Impl>> {
  using Arg1PlaceHolderType =
      typename ConvertToDeviceExpression<Arg1Impl>::Type;
  using Arg2PlaceHolderType =
      typename ConvertToDeviceExpression<Arg2Impl>::Type;
  using Arg3PlaceHolderType =
      typename ConvertToDeviceExpression<Arg3Impl>::Type;
  using Type =
      const TensorCwiseTernaryOp<OP, Arg1PlaceHolderType, Arg2PlaceHolderType,
                                 Arg3PlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is TensorCwiseTernaryOp
template <typename OP, typename Arg1Impl, typename Arg2Impl, typename Arg3Impl>
struct ConvertToDeviceExpression<
    TensorCwiseTernaryOp<OP, Arg1Impl, Arg2Impl, Arg3Impl>> {
  using Arg1PlaceHolderType =
      typename ConvertToDeviceExpression<Arg1Impl>::Type;
  using Arg2PlaceHolderType =
      typename ConvertToDeviceExpression<Arg2Impl>::Type;
  using Arg3PlaceHolderType =
      typename ConvertToDeviceExpression<Arg3Impl>::Type;
  using Type = TensorCwiseTernaryOp<OP, Arg1PlaceHolderType,
                                    Arg2PlaceHolderType, Arg3PlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is const TensorCwiseSelectOp
template <typename IfExpr, typename ThenExpr, typename ElseExpr>
struct ConvertToDeviceExpression<
    const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>> {
  using IfPlaceHolderType = typename ConvertToDeviceExpression<IfExpr>::Type;
  using ThenPlaceHolderType =
      typename ConvertToDeviceExpression<ThenExpr>::Type;
  using ElsePlaceHolderType =
      typename ConvertToDeviceExpression<ElseExpr>::Type;
  using Type = const TensorSelectOp<IfPlaceHolderType, ThenPlaceHolderType,
                                    ElsePlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is TensorCwiseSelectOp
template <typename IfExpr, typename ThenExpr, typename ElseExpr>
struct ConvertToDeviceExpression<TensorSelectOp<IfExpr, ThenExpr, ElseExpr>> {
  using IfPlaceHolderType = typename ConvertToDeviceExpression<IfExpr>::Type;
  using ThenPlaceHolderType =
      typename ConvertToDeviceExpression<ThenExpr>::Type;
  using ElsePlaceHolderType =
      typename ConvertToDeviceExpression<ElseExpr>::Type;
  using Type = TensorSelectOp<IfPlaceHolderType, ThenPlaceHolderType,
                              ElsePlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is const AssingOP
template <typename LHSExpr, typename RHSExpr>
struct ConvertToDeviceExpression<const TensorAssignOp<LHSExpr, RHSExpr>> {
  using LHSPlaceHolderType = typename ConvertToDeviceExpression<LHSExpr>::Type;
  using RHSPlaceHolderType = typename ConvertToDeviceExpression<RHSExpr>::Type;
  using Type = const TensorAssignOp<LHSPlaceHolderType, RHSPlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is AssingOP
template <typename LHSExpr, typename RHSExpr>
struct ConvertToDeviceExpression<TensorAssignOp<LHSExpr, RHSExpr>> {
  using LHSPlaceHolderType = typename ConvertToDeviceExpression<LHSExpr>::Type;
  using RHSPlaceHolderType = typename ConvertToDeviceExpression<RHSExpr>::Type;
  using Type = TensorAssignOp<LHSPlaceHolderType, RHSPlaceHolderType>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is const TensorForcedEvalOp
template <typename Expr>
struct ConvertToDeviceExpression<const TensorForcedEvalOp<Expr>> {
  using PlaceHolderType = typename ConvertToDeviceExpression<Expr>::Type;
  using Type = const TensorForcedEvalOp<PlaceHolderType, MakeGlobalPointer>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is const TensorForcedEvalOp
template <typename Expr>
struct ConvertToDeviceExpression<TensorForcedEvalOp<Expr>> {
  using PlaceHolderType = typename ConvertToDeviceExpression<Expr>::Type;
  using Type = TensorForcedEvalOp<PlaceHolderType, MakeGlobalPointer>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is const TensorEvalToOp
template <typename Expr>
struct ConvertToDeviceExpression<const TensorEvalToOp<Expr>> {
  using PlaceHolderType = typename ConvertToDeviceExpression<Expr>::Type;
  using Type = const TensorEvalToOp<PlaceHolderType, MakeGlobalPointer>;
};

/// specialisation of the \ref ConvertToDeviceExpression struct when the node
/// type is TensorEvalToOp
template <typename Expr>
struct ConvertToDeviceExpression<TensorEvalToOp<Expr>> {
  using PlaceHolderType = typename ConvertToDeviceExpression<Expr>::Type;
  using Type = TensorEvalToOp<PlaceHolderType, MakeGlobalPointer>;
};
}  // namespace internal
}  // namespace TensorSycl
}  // namespace Eigen

#endif  // UNSUPPORTED_EIGEN_CXX1
