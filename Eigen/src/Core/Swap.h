// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SWAP_H
#define EIGEN_SWAP_H

namespace Eigen { 

namespace internal {

// Overload default assignPacket behavior for swapping them
template<typename DstEvaluatorTypeT, typename SrcEvaluatorTypeT>
class generic_dense_assignment_kernel<DstEvaluatorTypeT, SrcEvaluatorTypeT, swap_assign_op<typename DstEvaluatorTypeT::Scalar>, Specialized>
 : public generic_dense_assignment_kernel<DstEvaluatorTypeT, SrcEvaluatorTypeT, swap_assign_op<typename DstEvaluatorTypeT::Scalar>, BuiltIn>
{
protected:
  typedef generic_dense_assignment_kernel<DstEvaluatorTypeT, SrcEvaluatorTypeT, swap_assign_op<typename DstEvaluatorTypeT::Scalar>, BuiltIn> Base;
  typedef typename DstEvaluatorTypeT::PacketScalar PacketScalar;
  using Base::m_dst;
  using Base::m_src;
  using Base::m_functor;
  
public:
  typedef typename Base::Scalar Scalar;
  typedef typename Base::Index Index;
  typedef typename Base::DstXprType DstXprType;
  typedef swap_assign_op<Scalar> Functor;
  
  EIGEN_DEVICE_FUNC generic_dense_assignment_kernel(DstEvaluatorTypeT &dst, const SrcEvaluatorTypeT &src, const Functor &func, DstXprType& dstExpr)
    : Base(dst, src, func, dstExpr)
  {}
  
  template<int StoreMode, int LoadMode>
  void assignPacket(Index row, Index col)
  {
    m_functor.template swapPacket<StoreMode,LoadMode,PacketScalar>(&m_dst.coeffRef(row,col), &const_cast<SrcEvaluatorTypeT&>(m_src).coeffRef(row,col));
  }
  
  template<int StoreMode, int LoadMode>
  void assignPacket(Index index)
  {
    m_functor.template swapPacket<StoreMode,LoadMode,PacketScalar>(&m_dst.coeffRef(index), &const_cast<SrcEvaluatorTypeT&>(m_src).coeffRef(index));
  }
  
  // TODO find a simple way not to have to copy/paste this function from generic_dense_assignment_kernel, by simple I mean no CRTP (Gael)
  template<int StoreMode, int LoadMode>
  void assignPacketByOuterInner(Index outer, Index inner)
  {
    Index row = Base::rowIndexByOuterInner(outer, inner); 
    Index col = Base::colIndexByOuterInner(outer, inner);
    assignPacket<StoreMode,LoadMode>(row, col);
  }
};

} // namespace internal

} // end namespace Eigen

#endif // EIGEN_SWAP_H
