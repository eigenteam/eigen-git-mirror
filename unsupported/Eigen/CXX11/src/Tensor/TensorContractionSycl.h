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
 *  TensorContractionsycl
 *
*****************************************************************/

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_SYCL_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_SYCL_H
namespace Eigen {

template <typename LhsScalar, typename RhsScalar,bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered> struct LaunchSyclKernels;
template<typename Indices, typename LeftArgType, typename RightArgType>
struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, const Eigen::SyclDevice> :
    public TensorContractionEvaluatorBase<TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, const Eigen::SyclDevice> > {

  typedef const Eigen::SyclDevice Device;

  typedef TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, Device> Self;
  typedef TensorContractionEvaluatorBase<Self> Base;
  typedef TensorContractionOp<Indices, LeftArgType, RightArgType> XprType;
  typedef typename internal::remove_const<typename XprType::Scalar>::type Scalar;
  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;

  enum {
    Layout = TensorEvaluator<LeftArgType, Device>::Layout,
  };

  // Most of the code is assuming that both input tensors are ColMajor. If the
  // inputs are RowMajor, we will "cheat" by swapping the LHS and RHS:
  // If we want to compute A * B = C, where A is LHS and B is RHS, the code
  // will pretend B is LHS and A is RHS.
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), LeftArgType, RightArgType>::type EvalLeftArgType;
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), RightArgType, LeftArgType>::type EvalRightArgType;

  static const int LDims =
      internal::array_size<typename TensorEvaluator<EvalLeftArgType, Device>::Dimensions>::value;
  static const int RDims =
      internal::array_size<typename TensorEvaluator<EvalRightArgType, Device>::Dimensions>::value;
  static const int ContractDims = internal::array_size<Indices>::value;

  typedef array<Index, LDims> left_dim_mapper_t;
  typedef array<Index, RDims> right_dim_mapper_t;

  typedef array<Index, ContractDims> contract_t;
  typedef array<Index, LDims - ContractDims> left_nocontract_t;
  typedef array<Index, RDims - ContractDims> right_nocontract_t;

  static const int NumDims = LDims + RDims - 2 * ContractDims;

  typedef DSizes<Index, NumDims> Dimensions;

  // typedefs needed in evalTo
  typedef typename internal::remove_const<typename EvalLeftArgType::Scalar>::type LhsScalar;
  typedef typename internal::remove_const<typename EvalRightArgType::Scalar>::type RhsScalar;

  typedef TensorEvaluator<EvalLeftArgType, Device> LeftEvaluator;
  typedef TensorEvaluator<EvalRightArgType, Device> RightEvaluator;

  typedef typename LeftEvaluator::Dimensions LeftDimensions;
  typedef typename RightEvaluator::Dimensions RightDimensions;

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device) :
      Base(op, device) {}

  // We need to redefine this method to make nvcc happy
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* data) {
    this->m_leftImpl.evalSubExprsIfNeeded(NULL);
    this->m_rightImpl.evalSubExprsIfNeeded(NULL);
    if (data) {
      evalTo(data);
      return false;
    } else {
      this->m_result = static_cast<Scalar*>(this->m_device.allocate(this->dimensions().TotalSize() * sizeof(Scalar)));
      evalTo(this->m_result);
      return true;
    }
  }
  const Eigen::SyclDevice& device() const {return this->m_device;}
  void evalTo(Scalar* buffer) const {
    // Here is the result
    if (this->m_lhs_inner_dim_contiguous) {
      if (this->m_rhs_inner_dim_contiguous) {
        if (this->m_rhs_inner_dim_reordered) {
          evalTyped<true, true, true, Unaligned>(buffer);
        }
        else {
          evalTyped<true, true, false, Unaligned>(buffer);
        }
      }
      else {
       if (this->m_rhs_inner_dim_reordered) {
          evalTyped<true, false, true, Unaligned>(buffer);
        }
        else {
          evalTyped<true, false, false, Unaligned>(buffer);
        }
      }
    }
    else {
      if (this->m_rhs_inner_dim_contiguous) {
        if (this->m_rhs_inner_dim_reordered) {
          evalTyped<false, true, true, Unaligned>(buffer);
        }
        else {
          evalTyped<false, true, false, Unaligned>(buffer);
        }
      }
      else {
       if (this->m_rhs_inner_dim_reordered) {
          evalTyped<false, false, true, Unaligned>(buffer);
        }
        else {
          evalTyped<false, false, false, Unaligned>(buffer);
        }
      }
    }
  }

  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment>
  void evalTyped(Scalar* buffer) const {
    // columns in left side, rows in right side
    const Index k = this->m_k_size;
    EIGEN_UNUSED_VARIABLE(k)
    // rows in left side
    const Index m = this->m_i_size;
    // columns in right side
    const Index n = this->m_j_size;

    // zero out the result buffer (which must be of size at least m * n * sizeof(Scalar)
    this->m_device.memset(buffer, 0, m * n * sizeof(Scalar));
  LaunchSyclKernels<LhsScalar, RhsScalar,lhs_inner_dim_contiguous, rhs_inner_dim_contiguous, rhs_inner_dim_reordered>::Run(*this, buffer, m, n, k,
   this->m_k_strides, this->m_left_contracting_strides, this->m_right_contracting_strides,
   this->m_i_strides, this->m_j_strides, this->m_left_nocontract_strides, this->m_right_nocontract_strides);
  }
  // required by sycl to construct the expr on the device. Returns original left_impl
  const TensorEvaluator<LeftArgType, Device>& left_impl() const {
    return choose(Cond<static_cast<int>(Layout) == static_cast<int>(ColMajor)>(), this->m_leftImpl, this->m_rightImpl);
  }
  // required by sycl to construct the expr on the device. Returns original right_impl
  const TensorEvaluator<RightArgType, Device>& right_impl() const {
    return choose(Cond<static_cast<int>(Layout) == static_cast<int>(ColMajor)>(), this->m_rightImpl, this->m_leftImpl);
  }
  // required by sycl to construct the expr on the device
  const Indices& indices() const {return this->m_expr_indices;}
};

/// Dummy container on the device. This is used to avoid calling the constructor of TensorEvaluator for TensorContractionOp. This makes the code much faster.
template<typename Expr> struct TensorEvaluatorContainer;
template<typename Indices, typename LeftArgType, typename RightArgType>
struct TensorEvaluatorContainer<TensorContractionOp<Indices, LeftArgType, RightArgType>>{
  typedef Eigen::DefaultDevice Device;
  typedef TensorContractionOp<Indices, LeftArgType, RightArgType> XprType;
  typedef typename internal::remove_const<typename XprType::Scalar>::type Scalar;
  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Eigen::DefaultDevice>::type PacketReturnType;
  enum {
    Layout = TensorEvaluator<LeftArgType, Device>::Layout,
  };

  typedef typename internal::conditional<static_cast<int>(Layout) == static_cast<int>(ColMajor), LeftArgType, RightArgType>::type EvalLeftArgType;
  typedef typename internal::conditional<static_cast<int>(Layout) == static_cast<int>(ColMajor), RightArgType, LeftArgType>::type EvalRightArgType;
  typedef TensorEvaluator<EvalLeftArgType, Device> LeftEvaluator;
  typedef TensorEvaluator<EvalRightArgType, Device> RightEvaluator;

  TensorEvaluatorContainer(const XprType& op, const Eigen::DefaultDevice& device)
  : m_leftImpl(choose(Cond<static_cast<int>(Layout) == static_cast<int>(ColMajor)>(),
                        op.lhsExpression(), op.rhsExpression()), device),
  m_rightImpl(choose(Cond<static_cast<int>(Layout) == static_cast<int>(ColMajor)>(),
                        op.rhsExpression(), op.lhsExpression()), device){}
LeftEvaluator  m_leftImpl;
RightEvaluator m_rightImpl;
};


template <typename HostExpr, typename OutScalar, typename LhsScalar, typename RhsScalar,  typename FunctorExpr, typename LhsLocalAcc, typename RhsLocalAcc, typename OutAccessor, typename Index, typename ContractT, typename LeftNocontractT,
typename RightNocontractT, bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered,
int TileSizeDimM, int TileSizeDimN,int TileSizeDimK, int WorkLoadPerThreadM,int WorkLoadPerThreadN,
int LocalThreadSizeM, int LocalThreadSizeN, int LoadPerThreadLhs, int LoadPerThreadRhs, typename TupleType> struct KernelConstructor{

  typedef  typename Eigen::TensorSycl::internal::createPlaceHolderExpression<HostExpr>::Type PlaceHolderExpr;

  FunctorExpr functors;
  LhsLocalAcc localLhs;
  RhsLocalAcc localRhs;
  OutAccessor out_res;
  Index roundUpK, M, N, K;
  ContractT m_k_strides, m_left_contracting_strides, m_right_contracting_strides;
  LeftNocontractT m_i_strides, m_left_nocontract_strides;
  RightNocontractT m_j_strides,  m_right_nocontract_strides;
  TupleType tuple_of_accessors;

  KernelConstructor(FunctorExpr functors_, LhsLocalAcc localLhs_, RhsLocalAcc localRhs_, OutAccessor out_res_,
    Index roundUpK_, Index M_, Index N_, Index K_, ContractT m_k_strides_, ContractT m_left_contracting_strides_,
    ContractT m_right_contracting_strides_, LeftNocontractT m_i_strides_, RightNocontractT m_j_strides_,
    LeftNocontractT m_left_nocontract_strides_, RightNocontractT m_right_nocontract_strides_, TupleType tuple_of_accessors_)
    :functors(functors_), localLhs(localLhs_), localRhs(localRhs_), out_res(out_res_), roundUpK(roundUpK_), M(M_), N(N_), K(K_),
    m_k_strides(m_k_strides_), m_left_contracting_strides(m_left_contracting_strides_),
    m_right_contracting_strides(m_right_contracting_strides_),
    m_i_strides(m_i_strides_), m_left_nocontract_strides(m_left_nocontract_strides_),
    m_j_strides(m_j_strides_),  m_right_nocontract_strides(m_right_nocontract_strides_),
    tuple_of_accessors(tuple_of_accessors_){}

    void operator()(cl::sycl::nd_item<1> itemID) {
      typedef  typename Eigen::TensorSycl::internal::ConvertToDeviceExpression<HostExpr>::Type DevExpr;
      auto device_expr =Eigen::TensorSycl::internal::createDeviceExpression<DevExpr, PlaceHolderExpr>(functors, tuple_of_accessors);
      auto device_evaluator = TensorEvaluatorContainer<DevExpr>(device_expr.expr, Eigen::DefaultDevice());
      typedef TensorEvaluatorContainer<DevExpr> DevEvaluator;
      typedef internal::TensorContractionInputMapper<LhsScalar, Index, internal::Lhs,
                                                     typename DevEvaluator::LeftEvaluator, LeftNocontractT,
                                                     ContractT, 1,
                                                     lhs_inner_dim_contiguous,
                                                     false, Unaligned, MakeGlobalPointer> LhsMapper;

      typedef internal::TensorContractionInputMapper<RhsScalar, Index, internal::Rhs,
                                                     typename DevEvaluator::RightEvaluator, RightNocontractT,
                                                     ContractT, 1,
                                                     rhs_inner_dim_contiguous,
                                                     rhs_inner_dim_reordered, Unaligned, MakeGlobalPointer> RhsMapper;
      // initialize data mappers must happen inside the kernel for device eval
      LhsMapper lhs(device_evaluator.m_leftImpl, m_left_nocontract_strides, m_i_strides, m_left_contracting_strides, m_k_strides);
      RhsMapper rhs(device_evaluator.m_rightImpl, m_right_nocontract_strides, m_j_strides, m_right_contracting_strides, m_k_strides);
      auto out_ptr = ConvertToActualTypeSycl(OutScalar, out_res);
      // Matmul Kernel
      // Thread identifiers
      const int mLocalThreadId = itemID.get_local(0); // Local ID row
      const int nLocalThreadId = itemID.get_local(1); // Local ID col
      const int mGroupId = itemID.get_group(0); // Work-group ID row
      const int nGroupId = itemID.get_group(1); // Work-group ID localCol
      const int linearLocalThreadId = nLocalThreadId*LocalThreadSizeM + mLocalThreadId; // linear local thread ID
      // Allocate register space
      float privateLhs;
      float privateRhs[WorkLoadPerThreadN];
      float privateRes[WorkLoadPerThreadM][WorkLoadPerThreadN];
      // Initialise the privateResumulation registers
      for (int wLPTM=0; wLPTM<WorkLoadPerThreadM; wLPTM++) {
          for (int wLPTN=0; wLPTN<WorkLoadPerThreadN; wLPTN++) {
              privateRes[wLPTM][wLPTN] = 0.0f;
          }
      }

      // Tile Lhs
      for (int lPTL=0; lPTL<LoadPerThreadLhs; lPTL++) {
          int
          localLhsLinearId = lPTL*LocalThreadSizeN*LocalThreadSizeM + linearLocalThreadId;
          int localLhsRow =  localLhsLinearId% TileSizeDimM;
          int localLhsCol = localLhsLinearId/TileSizeDimM;
          // Load the value (wide vector load)
          int GlobalLhsColId = TileSizeDimK*0 + localLhsCol;
          localLhs[0 + ((localLhsCol*TileSizeDimM + localLhsRow)*2)] =((GlobalLhsColId < K)&& (mGroupId*(TileSizeDimM)+ localLhsRow <M))? lhs(mGroupId*(TileSizeDimM) + localLhsRow, GlobalLhsColId):static_cast<OutScalar>(0);
      }
      // Tile Rhs
      for (int lPTR=0; lPTR<LoadPerThreadRhs; lPTR++) {
          int localRhsLinearId = lPTR*LocalThreadSizeN*LocalThreadSizeM + linearLocalThreadId;
          int localRhsRow =  localRhsLinearId% TileSizeDimN;
          int localRhsCol = localRhsLinearId/TileSizeDimN;
          // Load the value (wide vector load)
          int GlobalRhsRowId = TileSizeDimK*0 + localRhsCol;
          localRhs[0 + ((localRhsCol*TileSizeDimN + localRhsRow) *2)] = ((GlobalRhsRowId < K)&& ((nGroupId*(TileSizeDimN) + localRhsRow)< N))? rhs(GlobalRhsRowId, nGroupId*(TileSizeDimN) + localRhsRow): static_cast<OutScalar>(0);

      }
      // Loop over all tiles
      const int numTiles = roundUpK/TileSizeDimK;
      int firstHalf=0;
      do {
          // Synchronise
          itemID.barrier(cl::sycl::access::fence_space::local_space);
          // Load the next tile of Lhs and Rhs into local memory
          int nextHalf = firstHalf + 1;
          if (nextHalf < numTiles) {
              // Tile A
              for (int lPTL=0; lPTL<LoadPerThreadLhs; lPTL++) {
                  int localLhsLinearId = lPTL*LocalThreadSizeN*LocalThreadSizeM + linearLocalThreadId;
                  int localLhsRow =  localLhsLinearId% TileSizeDimM;
                  int localLhsCol = localLhsLinearId/TileSizeDimM;
                  // global K id
                  int GlobalLhsColId = TileSizeDimK*nextHalf + localLhsCol;
                  // Store the loaded value into local memory
                  localLhs[(nextHalf%2) + ((localLhsCol*TileSizeDimM + localLhsRow) *2)] = ((GlobalLhsColId < K)&& (mGroupId*(TileSizeDimM)+ localLhsRow <M))? lhs(mGroupId*(TileSizeDimM) + localLhsRow, GlobalLhsColId): static_cast<OutScalar>(0);
              }
              // Tile B
              for (int lPTR=0; lPTR<LoadPerThreadRhs; lPTR++) {
                  int localRhsLinearId = lPTR*LocalThreadSizeN*LocalThreadSizeM + linearLocalThreadId;
                  int localRhsRow =  localRhsLinearId% TileSizeDimN;
                  int localRhsCol = localRhsLinearId/TileSizeDimN;
                  // Load the value (wide vector load)
                  int GlobalRhsRowId = TileSizeDimK*nextHalf + localRhsCol;
                  // Store the loaded vector into local memory
                  localRhs[(nextHalf%2) +((localRhsCol*TileSizeDimN + localRhsRow)*2)] = ((GlobalRhsRowId < K)&& ((nGroupId*(TileSizeDimN) + localRhsRow)< N))? rhs(GlobalRhsRowId, nGroupId*(TileSizeDimN) + localRhsRow):static_cast<OutScalar>(0);
              }
          }
          // Loop over the values of a single tile
          for (int k=0; k<TileSizeDimK; k++) {
              // Cache the values of localRhs in registers
              for (int wLPTN=0; wLPTN<WorkLoadPerThreadN; wLPTN++) {
                  int localRhsCol = nLocalThreadId + wLPTN*LocalThreadSizeN;
                  privateRhs[wLPTN] = localRhs[(firstHalf%2) +((k*TileSizeDimN + localRhsCol)*2)];
              }
              // Perform the computation
              for (int wLPTM=0; wLPTM<WorkLoadPerThreadM; wLPTM++) {
                  int localLhsRow = mLocalThreadId + wLPTM*LocalThreadSizeM;
                  privateLhs = localLhs[(firstHalf%2)+ ((k*TileSizeDimM + localLhsRow)*2)];
                  for (int wLPTN=0; wLPTN<WorkLoadPerThreadN; wLPTN++) {
                      privateRes[wLPTM][wLPTN] += privateLhs * privateRhs[wLPTN];
                  }
              }
          }
          // Next tile
          firstHalf++;
      } while (firstHalf<numTiles);


      // Store the final results in C
      for (int wLPTM=0; wLPTM<WorkLoadPerThreadM; wLPTM++) {
          int globalRow = mGroupId*TileSizeDimM + mLocalThreadId + wLPTM*LocalThreadSizeM;
          if (globalRow< M){
            for (int wLPTN=0; wLPTN<WorkLoadPerThreadN; wLPTN++) {
                int globalCol = nGroupId*TileSizeDimN + nLocalThreadId + wLPTN*LocalThreadSizeN;
                if(globalCol<N)
                  out_ptr[globalCol*M + globalRow] = privateRes[wLPTM][wLPTN];
            }
          }
      }

    }

};
template <typename LhsScalar, typename RhsScalar, bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered> struct LaunchSyclKernels {

static const int TileSizeDimM = 32;                                      // Tile size for dimension M
static const int TileSizeDimN = 32;                                      // Tile size for dimension N
static const int TileSizeDimK = 16;                                      // Tile size for dimension K
static const int WorkLoadPerThreadM = 4;                                 // Work load per thread in dimension M
static const int WorkLoadPerThreadN = 4;                                 // work load per thread in dimension N
static const int LocalThreadSizeM = (TileSizeDimM/WorkLoadPerThreadM);   // Local thread size for the first dimension (M here)
static const int LocalThreadSizeN = (TileSizeDimN/WorkLoadPerThreadN);   // Local thread size for the second dimension (N here)
static const int LoadPerThreadLhs = ((TileSizeDimK*WorkLoadPerThreadM*WorkLoadPerThreadN)/(TileSizeDimN));  // workload per thread for Lhs expression
static const int LoadPerThreadRhs = ((TileSizeDimK*WorkLoadPerThreadM*WorkLoadPerThreadN)/(TileSizeDimM));  // workload per thread for Rhs expression

// RoundUp function to make sure that the global threadId is divisable by local threadId
static int RoundUp(int x, int y) {
  return ((((x) + (y) - 1) / (y))*(y));
}

template< typename Self, typename OutScalar, typename Index, typename ContractT, typename LeftNocontractT, typename RightNocontractT>
  static void Run(const Self& self, OutScalar* buffer,  Index M, Index N, Index K,
    ContractT m_k_strides, ContractT m_left_contracting_strides, ContractT m_right_contracting_strides,
    LeftNocontractT m_i_strides, RightNocontractT m_j_strides, LeftNocontractT m_left_nocontract_strides, RightNocontractT m_right_nocontract_strides){
    // create a tuple of accessors from Evaluator
    typedef typename Self::XprType HostExpr;
  //  typedef  typename Eigen::TensorSycl::internal::createPlaceHolderExpression<HostExpr>::Type PlaceHolderExpr;
  //  typedef KernelNameConstructor<PlaceHolderExpr, lhs_inner_dim_contiguous, rhs_inner_dim_contiguous, rhs_inner_dim_reordered> KernelName;
    auto functors = Eigen::TensorSycl::internal::extractFunctors(self);
    typedef decltype(functors) FunctorExpr;
    Index roundUpK = RoundUp(K, TileSizeDimK);
    Index roundUpM = RoundUp(M, TileSizeDimM);
    Index roundUpN = RoundUp(N, TileSizeDimN);
    self.device().sycl_queue().submit([&](cl::sycl::handler &cgh) {
      auto tuple_of_accessors = Eigen::TensorSycl::internal::createTupleOfAccessors<Self>(cgh, self);
      typedef decltype(tuple_of_accessors) TupleType;
      // Local memory for elements of Lhs
      typedef cl::sycl::accessor<LhsScalar, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> LhsLocalAcc;
      LhsLocalAcc localLhs(cl::sycl::range<1>(2* TileSizeDimM * TileSizeDimK), cgh);
      // Local memory for elements of Rhs
      typedef cl::sycl::accessor<RhsScalar, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> RhsLocalAcc;
      RhsLocalAcc localRhs(cl::sycl::range<1>(2* TileSizeDimK * TileSizeDimN), cgh);
      //OutScalar memory
      auto out_res= self.device(). template get_sycl_accessor<cl::sycl::access::mode::write>(cgh, buffer);
      typedef decltype(out_res) OutAccessor;
      // sycl parallel for
      cgh.parallel_for(cl::sycl::nd_range<2>(cl::sycl::range<2>(roundUpM/WorkLoadPerThreadM, roundUpN/WorkLoadPerThreadN),
      cl::sycl::range<2>(LocalThreadSizeM, LocalThreadSizeN)),
       KernelConstructor<HostExpr, OutScalar, LhsScalar, RhsScalar,  FunctorExpr, LhsLocalAcc, RhsLocalAcc, OutAccessor, Index, ContractT, LeftNocontractT,
       RightNocontractT, lhs_inner_dim_contiguous, rhs_inner_dim_contiguous, rhs_inner_dim_reordered, TileSizeDimM, TileSizeDimN, TileSizeDimK,
       WorkLoadPerThreadM, WorkLoadPerThreadN, LocalThreadSizeM, LocalThreadSizeN, LoadPerThreadLhs, LoadPerThreadRhs, TupleType>(functors,
          localLhs, localRhs, out_res, roundUpK, M, N, K, m_k_strides, m_left_contracting_strides, m_right_contracting_strides,m_i_strides, m_j_strides,
          m_left_nocontract_strides,m_right_nocontract_strides, tuple_of_accessors));
    });
    self.device().asynchronousExec();
  }
};

} // end namespace Eigen
#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_SYCL_H
