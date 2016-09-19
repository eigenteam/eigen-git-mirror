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
 * TensorSyclLeafCount.h
 *
 * \brief:
 *  The leaf count used the pre-order expression tree traverse in order to name
 *  count the number of leaf nodes in the expression
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_LEAF_COUNT_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_LEAF_COUNT_HPP

namespace Eigen {
namespace TensorSycl {
namespace internal {
/// \brief LeafCount used to counting terminal nodes. The total number of
/// leaf nodes is used by MakePlaceHolderExprHelper to find the order
/// of the leaf node in a expression tree at compile time.
template <typename Expr>
struct LeafCount;

/// specialisation of the \ref LeafCount struct when the node type is const
/// TensorMap
template <typename PlainObjectType, int Options_,
          template <class> class MakePointer_>
struct LeafCount<const TensorMap<PlainObjectType, Options_, MakePointer_>> {
  static const size_t Count = 1;
};

/// specialisation of the \ref LeafCount struct when the node type is TensorMap
template <typename PlainObjectType, int Options_,
          template <class> class MakePointer_>
struct LeafCount<TensorMap<PlainObjectType, Options_, MakePointer_>> {
  static const size_t Count = 1;
};

/// specialisation of the \ref LeafCount struct when the node type is const
/// TensorCwiseNullaryOp
template <typename OP, typename RHSExpr>
struct LeafCount<const TensorCwiseNullaryOp<OP, RHSExpr>> {
  static const size_t Count = LeafCount<RHSExpr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is
/// TensorCwiseNullaryOp
template <typename OP, typename RHSExpr>
struct LeafCount<TensorCwiseNullaryOp<OP, RHSExpr>> {
  static const size_t Count = LeafCount<RHSExpr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is const
/// TensorBroadcastingOp
template <typename OP, typename RHSExpr>
struct LeafCount<const TensorBroadcastingOp<OP, RHSExpr>> {
  static const size_t Count = LeafCount<RHSExpr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is
/// TensorCwiseNullaryOp
template <typename OP, typename RHSExpr>
struct LeafCount<TensorBroadcastingOp<OP, RHSExpr>> {
  static const size_t Count = LeafCount<RHSExpr>::Count;
};

// TensorCwiseUnaryOp
template <typename OP, typename RHSExpr>
struct LeafCount<const TensorCwiseUnaryOp<OP, RHSExpr>> {
  static const size_t Count = LeafCount<RHSExpr>::Count;
};

// TensorCwiseUnaryOp
template <typename OP, typename RHSExpr>
struct LeafCount<TensorCwiseUnaryOp<OP, RHSExpr>> {
  static const size_t Count = LeafCount<RHSExpr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is const
/// TensorCwiseBinaryOp
template <typename OP, typename LHSExpr, typename RHSExpr>
struct LeafCount<const TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>> {
  static const size_t Count =
      LeafCount<LHSExpr>::Count + LeafCount<RHSExpr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is
/// TensorCwiseBinaryOp
template <typename OP, typename LHSExpr, typename RHSExpr>
struct LeafCount<TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>> {
  static const size_t Count =
      LeafCount<LHSExpr>::Count + LeafCount<RHSExpr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is
/// TensorCwiseTernaryOp
template <typename OP, typename Arg1Expr, typename Arg2Expr, typename Arg3Expr>
struct LeafCount<TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>> {
  static const size_t Count = LeafCount<Arg1Expr>::Count +
                              LeafCount<Arg2Expr>::Count +
                              LeafCount<Arg3Expr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is const
/// TensorCwiseTernaryOp
template <typename OP, typename Arg1Expr, typename Arg2Expr, typename Arg3Expr>
struct LeafCount<const TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>> {
  static const size_t Count = LeafCount<Arg1Expr>::Count +
                              LeafCount<Arg2Expr>::Count +
                              LeafCount<Arg3Expr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is
/// TensorCwiseSelectOp
template <typename IfExpr, typename ThenExpr, typename ElseExpr>
struct LeafCount<TensorSelectOp<IfExpr, ThenExpr, ElseExpr>> {
  static const size_t Count = LeafCount<IfExpr>::Count +
                              LeafCount<ThenExpr>::Count +
                              LeafCount<ElseExpr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is const
/// TensorCwiseSelectOp
template <typename IfExpr, typename ThenExpr, typename ElseExpr>
struct LeafCount<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>> {
  static const size_t Count = LeafCount<IfExpr>::Count +
                              LeafCount<ThenExpr>::Count +
                              LeafCount<ElseExpr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is
/// TensorAssignOp
template <typename LHSExpr, typename RHSExpr>
struct LeafCount<TensorAssignOp<LHSExpr, RHSExpr>> {
  static const size_t Count =
      LeafCount<LHSExpr>::Count + LeafCount<RHSExpr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is const
/// TensorAssignOp
template <typename LHSExpr, typename RHSExpr>
struct LeafCount<const TensorAssignOp<LHSExpr, RHSExpr>> {
  static const size_t Count =
      LeafCount<LHSExpr>::Count + LeafCount<RHSExpr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is const
/// TensorForcedEvalOp
template <typename Expr>
struct LeafCount<const TensorForcedEvalOp<Expr>> {
  static const size_t Count = 1;
};

/// specialisation of the \ref LeafCount struct when the node type is
/// TensorForcedEvalOp
template <typename Expr>
struct LeafCount<TensorForcedEvalOp<Expr>> {
  static const size_t Count = 1;
};

/// specialisation of the \ref LeafCount struct when the node type is const
/// TensorEvalToOp
template <typename Expr>
struct LeafCount<const TensorEvalToOp<Expr>> {
  static const size_t Count = 1 + LeafCount<Expr>::Count;
};

/// specialisation of the \ref LeafCount struct when the node type is
/// TensorEvalToOp
template <typename Expr>
struct LeafCount<TensorEvalToOp<Expr>> {
  static const size_t Count = 1 + LeafCount<Expr>::Count;
};
}
}
}  // namespace Eigen

#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_LEAF_COUNT_HPP
