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
 * TensorSyclPlaceHolderExpr.h
 *
 * \brief:
 *  This is the specialisation of the placeholder expression based on the
 * operation type
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_PLACEHOLDER_EXPR_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_PLACEHOLDER_EXPR_HPP

namespace Eigen {
namespace TensorSycl {
namespace internal {
/// \sttruct PlaceHolderExpression
/// \brief it is used to create the PlaceHolder expression. The PlaceHolder
/// expression is a copy of expression type in which the TensorMap of the has
/// been replaced with PlaceHolder.
template <typename Expr, size_t N>
struct PlaceHolderExpression;

/// specialisation of the \ref PlaceHolderExpression when the node is TensorMap
template <typename Scalar_, int Options_, int Options2_, int NumIndices_,
          typename IndexType_, template <class> class MakePointer_, size_t N>
struct PlaceHolderExpression<
    Eigen::TensorMap<Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType_>,
                     Options2_, MakePointer_>,
    N> {
  using Type = Eigen::internal::PlaceHolder<
      Eigen::TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>,
                       Options2_, MakePointer_>,
      N>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is const
/// TensorMap
template <typename Scalar_, int Options_, int Options2_, int NumIndices_,
          typename IndexType_, template <class> class MakePointer_, size_t N>
struct PlaceHolderExpression<
    const Eigen::TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>,
                           Options2_, MakePointer_>,
    N> {
  using Type = const Eigen::internal::PlaceHolder<
      const TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>,
                      Options2_, MakePointer_>,
      N>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorCwiseNullaryOp
template <typename OP, typename RHSExpr, size_t N>
struct PlaceHolderExpression<TensorCwiseNullaryOp<OP, RHSExpr>, N> {
  using RHSPlaceHolderType = typename PlaceHolderExpression<RHSExpr, N>::Type;
  using Type = TensorCwiseNullaryOp<OP, RHSPlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is const
/// TensorCwiseNullaryOp
template <typename OP, typename RHSExpr, size_t N>
struct PlaceHolderExpression<const TensorCwiseNullaryOp<OP, RHSExpr>, N> {
  using RHSPlaceHolderType = typename PlaceHolderExpression<RHSExpr, N>::Type;
  using Type = const TensorCwiseNullaryOp<OP, RHSPlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorBroadcastingOp
template <typename OP, typename RHSExpr, size_t N>
struct PlaceHolderExpression<TensorBroadcastingOp<OP, RHSExpr>, N> {
  using RHSPlaceHolderType = typename PlaceHolderExpression<RHSExpr, N>::Type;
  using Type = TensorBroadcastingOp<OP, RHSPlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is const
/// TensorBroadcastingOp
template <typename OP, typename RHSExpr, size_t N>
struct PlaceHolderExpression<const TensorBroadcastingOp<OP, RHSExpr>, N> {
  using RHSPlaceHolderType = typename PlaceHolderExpression<RHSExpr, N>::Type;
  using Type = const TensorBroadcastingOp<OP, RHSPlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorCwiseUnaryOp
template <typename OP, typename RHSExpr, size_t N>
struct PlaceHolderExpression<TensorCwiseUnaryOp<OP, RHSExpr>, N> {
  using RHSPlaceHolderType = typename PlaceHolderExpression<RHSExpr, N>::Type;
  using Type = TensorCwiseUnaryOp<OP, RHSPlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is const
/// TensorCwiseUnaryOp
template <typename OP, typename RHSExpr, size_t N>
struct PlaceHolderExpression<const TensorCwiseUnaryOp<OP, RHSExpr>, N> {
  using RHSPlaceHolderType = typename PlaceHolderExpression<RHSExpr, N>::Type;
  using Type = const TensorCwiseUnaryOp<OP, RHSPlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorCwiseBinaryOp
template <typename OP, typename LHSExpr, typename RHSExpr, size_t N>
struct PlaceHolderExpression<TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>, N> {
  static const size_t RHSLeafCount = LeafCount<RHSExpr>::Count;

  using LHSPlaceHolderType =
      typename PlaceHolderExpression<LHSExpr, N - RHSLeafCount>::Type;
  using RHSPlaceHolderType = typename PlaceHolderExpression<RHSExpr, N>::Type;

  using Type = TensorCwiseBinaryOp<OP, LHSPlaceHolderType, RHSPlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is const
/// TensorCwiseBinaryOp
template <typename OP, typename LHSExpr, typename RHSExpr, size_t N>
struct PlaceHolderExpression<const TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>,
                             N> {
  static const size_t RHSLeafCount = LeafCount<RHSExpr>::Count;

  using LHSPlaceHolderType =
      typename PlaceHolderExpression<LHSExpr, N - RHSLeafCount>::Type;
  using RHSPlaceHolderType = typename PlaceHolderExpression<RHSExpr, N>::Type;

  using Type =
      const TensorCwiseBinaryOp<OP, LHSPlaceHolderType, RHSPlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is const
/// TensorCwiseSelectOp
template <typename OP, typename Arg1Expr, typename Arg2Expr, typename Arg3Expr,
          size_t N>
struct PlaceHolderExpression<
    const TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>, N> {
  static const size_t Arg3LeafCount = LeafCount<Arg3Expr>::Count;
  static const size_t Arg2LeafCount = LeafCount<Arg2Expr>::Count;

  using Arg1PlaceHolderType =
      typename PlaceHolderExpression<Arg1Expr,
                                     N - Arg3LeafCount - Arg2LeafCount>::Type;
  using Arg2PlaceHolderType =
      typename PlaceHolderExpression<Arg2Expr, N - Arg3LeafCount>::Type;

  using Arg3PlaceHolderType = typename PlaceHolderExpression<Arg3Expr, N>::Type;

  using Type =
      const TensorCwiseTernaryOp<OP, Arg1PlaceHolderType, Arg2PlaceHolderType,
                                 Arg3PlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorCwiseSelectOp
template <typename OP, typename Arg1Expr, typename Arg2Expr, typename Arg3Expr,
          size_t N>
struct PlaceHolderExpression<
    TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>, N> {
  static const size_t Arg3LeafCount = LeafCount<Arg3Expr>::Count;
  static const size_t Arg2LeafCount = LeafCount<Arg2Expr>::Count;

  using Arg1PlaceHolderType =
      typename PlaceHolderExpression<Arg1Expr,
                                     N - Arg3LeafCount - Arg2LeafCount>::Type;
  using Arg2PlaceHolderType =
      typename PlaceHolderExpression<Arg2Expr, N - Arg3LeafCount>::Type;

  using Arg3PlaceHolderType = typename PlaceHolderExpression<Arg3Expr, N>::Type;

  using Type = TensorCwiseTernaryOp<OP, Arg1PlaceHolderType,
                                    Arg2PlaceHolderType, Arg3PlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is const
/// TensorCwiseSelectOp
template <typename IfExpr, typename ThenExpr, typename ElseExpr, size_t N>
struct PlaceHolderExpression<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>,
                             N> {
  static const size_t ElseLeafCount = LeafCount<ElseExpr>::Count;
  static const size_t ThenLeafCount = LeafCount<ThenExpr>::Count;

  using IfPlaceHolderType =
      typename PlaceHolderExpression<IfExpr,
                                     N - ElseLeafCount - ThenLeafCount>::Type;
  using ThenPlaceHolderType =
      typename PlaceHolderExpression<ThenExpr, N - ElseLeafCount>::Type;

  using ElsePlaceHolderType = typename PlaceHolderExpression<ElseExpr, N>::Type;

  using Type = const TensorSelectOp<IfPlaceHolderType, ThenPlaceHolderType,
                                    ElsePlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorCwiseSelectOp
template <typename IfExpr, typename ThenExpr, typename ElseExpr, size_t N>
struct PlaceHolderExpression<TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, N> {
  static const size_t ElseLeafCount = LeafCount<ElseExpr>::Count;
  static const size_t ThenLeafCount = LeafCount<ThenExpr>::Count;

  using IfPlaceHolderType =
      typename PlaceHolderExpression<IfExpr,
                                     N - ElseLeafCount - ThenLeafCount>::Type;
  using ThenPlaceHolderType =
      typename PlaceHolderExpression<ThenExpr, N - ElseLeafCount>::Type;

  using ElsePlaceHolderType = typename PlaceHolderExpression<ElseExpr, N>::Type;

  using Type = TensorSelectOp<IfPlaceHolderType, ThenPlaceHolderType,
                              ElsePlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorAssignOp
template <typename LHSExpr, typename RHSExpr, size_t N>
struct PlaceHolderExpression<TensorAssignOp<LHSExpr, RHSExpr>, N> {
  static const size_t RHSLeafCount = LeafCount<RHSExpr>::Count;

  using LHSPlaceHolderType =
      typename PlaceHolderExpression<LHSExpr, N - RHSLeafCount>::Type;
  using RHSPlaceHolderType = typename PlaceHolderExpression<RHSExpr, N>::Type;

  using Type = TensorAssignOp<LHSPlaceHolderType, RHSPlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is const
/// TensorAssignOp
template <typename LHSExpr, typename RHSExpr, size_t N>
struct PlaceHolderExpression<const TensorAssignOp<LHSExpr, RHSExpr>, N> {
  static const size_t RHSLeafCount = LeafCount<RHSExpr>::Count;

  using LHSPlaceHolderType =
      typename PlaceHolderExpression<LHSExpr, N - RHSLeafCount>::Type;
  using RHSPlaceHolderType = typename PlaceHolderExpression<RHSExpr, N>::Type;

  using Type = const TensorAssignOp<LHSPlaceHolderType, RHSPlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is const
/// TensorForcedEvalOp
template <typename Expr, size_t N>
struct PlaceHolderExpression<const TensorForcedEvalOp<Expr>, N> {
  using Type =
      const Eigen::internal::PlaceHolder<const TensorForcedEvalOp<Expr>, N>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorForcedEvalOp
template <typename Expr, size_t N>
struct PlaceHolderExpression<TensorForcedEvalOp<Expr>, N> {
  using Type = Eigen::internal::PlaceHolder<TensorForcedEvalOp<Expr>, N>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is const
/// TensorEvalToOp
template <typename Expr, size_t N>
struct PlaceHolderExpression<const TensorEvalToOp<Expr>, N> {
  static const size_t RHSLeafCount = LeafCount<Expr>::Count;

  using RHSPlaceHolderType = typename PlaceHolderExpression<Expr, N>::Type;

  using Type = const TensorEvalToOp<RHSPlaceHolderType>;
};

/// specialisation of the \ref PlaceHolderExpression when the node is
/// TensorEvalToOp
template <typename Expr, size_t N>
struct PlaceHolderExpression<TensorEvalToOp<Expr>, N> {
  static const size_t RHSLeafCount = LeafCount<Expr>::Count;

  using RHSPlaceHolderType = typename PlaceHolderExpression<Expr, N>::Type;

  using Type = TensorEvalToOp<RHSPlaceHolderType>;
};

/// template deduction for \ref PlaceHolderExpression struct
template <typename Expr>
struct createPlaceHolderExpression {
  static const size_t TotalLeaves = LeafCount<Expr>::Count;
  using Type = typename PlaceHolderExpression<Expr, TotalLeaves - 1>::Type;
};
}
}
}  // namespace Eigen

#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_PLACEHOLDER_EXPR_HPP
