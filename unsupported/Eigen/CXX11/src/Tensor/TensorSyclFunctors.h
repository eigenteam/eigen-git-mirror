// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: eigen@codeplay.com
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// General include header of SYCL target for Tensor Module
#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCLFUNCTORS_H
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCLFUNCTORS_H

namespace Eigen {
namespace TensorSycl {
namespace internal {

  template<typename CoeffReturnType, typename OutputAccessor, typename InputAccessor, typename LocalAccessor> struct GenericKernelReducer{
    OutputAccessor aOut;
    InputAccessor aI;
    LocalAccessor scratch;
    size_t length, local;
    GenericKernelReducer(OutputAccessor aOut_, InputAccessor aI_, LocalAccessor scratch_, size_t length_, size_t local_)
    : aOut(aOut_), aI(aI_), scratch(scratch_), length(length_), local(local_){}
    void operator()(cl::sycl::nd_item<1> itemID) {
      size_t globalid = itemID.get_global(0);
      size_t localid = itemID.get_local(0);
      /* All threads collectively read from global memory into local.
       * The barrier ensures all threads' IO is resolved before
       * execution continues (strictly speaking, all threads within
       * a single work-group - there is no co-ordination between
       * work-groups, only work-items). */
      if (globalid < length) {
        scratch[localid] = aI[globalid];
      }
      itemID.barrier(cl::sycl::access::fence_space::local_space);

      /* Apply the reduction operation between the current local
       * id and the one on the other half of the vector. */
      if (globalid < length) {
        auto min = (length < local) ? length : local;
        for (size_t offset = min / 2; offset > 0; offset /= 2) {
          if (localid < offset) {
            scratch[localid] += scratch[localid + offset];
          }
          itemID.barrier(cl::sycl::access::fence_space::local_space);
        }
        /* The final result will be stored in local id 0. */
        if (localid == 0) {
          aI[itemID.get_group(0)] = scratch[localid];
          if((length<=local) && globalid ==0){
            auto aOutPtr = ConvertToActualTypeSycl(CoeffReturnType, aOut);
            aOutPtr[0]=scratch[0];
          }
        }
      }
    }

  };

/// ReductionFunctor
template < typename HostExpr, typename FunctorExpr, typename Tuple_of_Acc, typename Dims, typename Op, typename Index> class ReductionFunctor {
 public:
  typedef  typename TensorSycl::internal::createPlaceHolderExpression<HostExpr>::Type PlaceHolderExpr;
  typedef cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer> write_accessor;
  ReductionFunctor(write_accessor output_accessor_, FunctorExpr functors_, Tuple_of_Acc tuple_of_accessors_,Dims dims_, Op functor_, Index range_)
  :output_accessor(output_accessor_), functors(functors_), tuple_of_accessors(tuple_of_accessors_), dims(dims_), functor(functor_), range(range_) {}
  void operator()(cl::sycl::nd_item<1> itemID) {

    typedef typename ConvertToDeviceExpression<const HostExpr>::Type DevExpr;
    auto device_expr = createDeviceExpression<DevExpr, PlaceHolderExpr>(functors, tuple_of_accessors);
    /// reduction cannot be captured automatically through our device conversion recursion. The reason is that reduction has two behaviour
    /// the first behaviour is when it is used as a root to lauch the sub-kernel. The second one is when it is treated as a leafnode to pass the
    /// calculated result to its parent kernel. While the latter is automatically detected through our device expression generator. The former is created here.
    const auto device_self_expr= Eigen::TensorReductionOp<Op, Dims, decltype(device_expr.expr) ,MakeGlobalPointer>(device_expr.expr, dims, functor);
    /// This is the evaluator for device_self_expr. This is exactly similar to the self which has been passed to run function. The difference is
    /// the device_evaluator is detectable and recognisable on the device.
    typedef Eigen::TensorEvaluator<decltype(device_self_expr), Eigen::DefaultDevice> DeviceSelf;
    auto device_self_evaluator = Eigen::TensorEvaluator<decltype(device_self_expr), Eigen::DefaultDevice>(device_self_expr, Eigen::DefaultDevice());
    auto output_accessor_ptr =ConvertToActualTypeSycl(typename DeviceSelf::CoeffReturnType, output_accessor);
    /// const cast added as a naive solution to solve the qualifier drop error
    auto globalid=static_cast<Index>(itemID.get_global_linear_id());
    if (globalid< range) {
      typename DeviceSelf::CoeffReturnType accum = functor.initialize();
      Eigen::internal::GenericDimReducer<DeviceSelf::NumReducedDims-1, DeviceSelf, Op>::reduce(device_self_evaluator, device_self_evaluator.firstInput(static_cast<typename DevExpr::Index>(globalid)),const_cast<Op&>(functor), &accum);
      functor.finalize(accum);
      output_accessor_ptr[globalid]= accum;
    }
  }
 private:
  write_accessor output_accessor;
  FunctorExpr functors;
  Tuple_of_Acc tuple_of_accessors;
  Dims dims;
  Op functor;
  Index range;
};


template<typename CoeffReturnType ,typename OutAccessor, typename HostExpr, typename FunctorExpr, typename Op, typename Dims, typename Index, typename TupleType>
class FullReductionKernelFunctor{
public:
  typedef  typename TensorSycl::internal::createPlaceHolderExpression<HostExpr>::Type PlaceHolderExpr;
  OutAccessor tmp_global_accessor;
  Index rng , remaining, red_factor;
  Op op;
  Dims dims;
  FunctorExpr functors;
  TupleType tuple_of_accessors;

  FullReductionKernelFunctor(OutAccessor acc,   Index rng_, Index remaining_, Index red_factor_, Op op_, Dims dims_, FunctorExpr functors_, TupleType t_acc)
  :tmp_global_accessor(acc), rng(rng_), remaining(remaining_), red_factor(red_factor_),op(op_), dims(dims_), functors(functors_), tuple_of_accessors(t_acc){}

  void operator()(cl::sycl::nd_item<1> itemID) {

    typedef typename TensorSycl::internal::ConvertToDeviceExpression<const HostExpr>::Type DevExpr;
    auto device_expr = TensorSycl::internal::createDeviceExpression<DevExpr, PlaceHolderExpr>(functors, tuple_of_accessors);
    /// reduction cannot be captured automatically through our device conversion recursion. The reason is that reduction has two behaviour
    /// the first behaviour is when it is used as a root to lauch the sub-kernel. The second one is when it is treated as a leafnode to pass the
    /// calculated result to its parent kernel. While the latter is automatically detected through our device expression generator. The former is created here.
    const auto device_self_expr= Eigen::TensorReductionOp<Op, Dims, decltype(device_expr.expr) ,MakeGlobalPointer>(device_expr.expr, dims, op);
    /// This is the evaluator for device_self_expr. This is exactly similar to the self which has been passed to run function. The difference is
    /// the device_evaluator is detectable and recognisable on the device.
    auto device_self_evaluator = Eigen::TensorEvaluator<decltype(device_self_expr), Eigen::DefaultDevice>(device_self_expr, Eigen::DefaultDevice());
    /// const cast added as a naive solution to solve the qualifier drop error
    auto globalid=itemID.get_global_linear_id();

    if(globalid<rng)
      tmp_global_accessor.get_pointer()[globalid]=Eigen::internal::InnerMostDimReducer<decltype(device_self_evaluator), Op, false>::reduce(device_self_evaluator, static_cast<typename DevExpr::Index>(red_factor*globalid), red_factor, const_cast<Op&>(op));
    else
      tmp_global_accessor.get_pointer()[globalid]=static_cast<CoeffReturnType>(0);

    if(remaining!=0 && globalid==0 )
      // this will add the rest of input buffer when the input size is not devidable to red_factor.
      tmp_global_accessor.get_pointer()[0]+=Eigen::internal::InnerMostDimReducer<decltype(device_self_evaluator), Op, false>::reduce(device_self_evaluator, static_cast<typename DevExpr::Index>(red_factor*(rng)), static_cast<typename DevExpr::Index>(remaining), const_cast<Op&>(op));
  }
};



}
}
}
#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSOR_TENSORSYCLFUNCTORS_H
