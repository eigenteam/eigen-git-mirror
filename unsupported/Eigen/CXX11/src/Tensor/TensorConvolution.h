// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONVOLUTION_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONVOLUTION_H

namespace Eigen {

/** \class TensorConvolution
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor convolution class.
  *
  *
  */
namespace internal {
template<typename Dimensions, typename InputXprType, typename KernelXprType>
struct traits<TensorConvolutionOp<Dimensions, InputXprType, KernelXprType> >
{
  // Type promotion to handle the case where the types of the lhs and the rhs are different.
  typedef typename internal::promote_storage_type<typename InputXprType::Scalar,
                                                  typename KernelXprType::Scalar>::ret Scalar;
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename promote_storage_type<typename traits<InputXprType>::StorageKind,
                                        typename traits<KernelXprType>::StorageKind>::ret StorageKind;
  typedef typename promote_index_type<typename traits<InputXprType>::Index,
                                      typename traits<KernelXprType>::Index>::type Index;
  typedef typename InputXprType::Nested LhsNested;
  typedef typename KernelXprType::Nested RhsNested;
  typedef typename remove_reference<LhsNested>::type _LhsNested;
  typedef typename remove_reference<RhsNested>::type _RhsNested;

  enum {
    Flags = 0,
  };
};

template<typename Dimensions, typename InputXprType, typename KernelXprType>
struct eval<TensorConvolutionOp<Dimensions, InputXprType, KernelXprType>, Eigen::Dense>
{
  typedef const TensorConvolutionOp<Dimensions, InputXprType, KernelXprType>& type;
};

template<typename Dimensions, typename InputXprType, typename KernelXprType>
struct nested<TensorConvolutionOp<Dimensions, InputXprType, KernelXprType>, 1, typename eval<TensorConvolutionOp<Dimensions, InputXprType, KernelXprType> >::type>
{
  typedef TensorConvolutionOp<Dimensions, InputXprType, KernelXprType> type;
};

}  // end namespace internal



template<typename Indices, typename InputXprType, typename KernelXprType>
class TensorConvolutionOp : public TensorBase<TensorConvolutionOp<Indices, InputXprType, KernelXprType> >
{
  public:
  typedef typename Eigen::internal::traits<TensorConvolutionOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorConvolutionOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename internal::promote_storage_type<typename InputXprType::CoeffReturnType,
                                                  typename KernelXprType::CoeffReturnType>::ret CoeffReturnType;
  typedef typename internal::promote_storage_type<typename InputXprType::PacketReturnType,
                                                  typename KernelXprType::PacketReturnType>::ret PacketReturnType;
  typedef typename Eigen::internal::nested<TensorConvolutionOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorConvolutionOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorConvolutionOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorConvolutionOp(const InputXprType& input, const KernelXprType& kernel, const Indices& dims)
      : m_input_xpr(input), m_kernel_xpr(kernel), m_indices(dims) {}

    EIGEN_DEVICE_FUNC
    const Indices& indices() const { return m_indices; }

    /** \returns the nested expressions */
    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename InputXprType::Nested>::type&
    inputExpression() const { return m_input_xpr; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename KernelXprType::Nested>::type&
    kernelExpression() const { return m_kernel_xpr; }

  protected:
    typename InputXprType::Nested m_input_xpr;
    typename KernelXprType::Nested m_kernel_xpr;
    const Indices m_indices;
};


template<typename Indices, typename InputArgType, typename KernelArgType, typename Device>
struct TensorEvaluator<const TensorConvolutionOp<Indices, InputArgType, KernelArgType>, Device>
{
  typedef TensorConvolutionOp<Indices, InputArgType, KernelArgType> XprType;

  static const int NumDims = TensorEvaluator<InputArgType, Device>::Dimensions::count;
  static const int KernelDims = internal::array_size<Indices>::value;
  typedef typename XprType::Index Index;
  typedef DSizes<Index, NumDims> Dimensions;

  enum {
    IsAligned = TensorEvaluator<InputArgType, Device>::IsAligned & TensorEvaluator<KernelArgType, Device>::IsAligned,
    PacketAccess = /*TensorEvaluator<InputArgType>::PacketAccess & TensorEvaluator<KernelArgType>::PacketAccess */
                   false,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_inputImpl(op.inputExpression(), device), m_kernelImpl(op.kernelExpression(), device), m_dimensions(op.inputExpression().dimensions())
  {
    const typename TensorEvaluator<InputArgType, Device>::Dimensions& input_dims = m_inputImpl.dimensions();
    const typename TensorEvaluator<KernelArgType, Device>::Dimensions& kernel_dims = m_kernelImpl.dimensions();

    for (int i = 0; i < NumDims; ++i) {
      if (i > 0) {
        m_inputStride[i] = m_inputStride[i-1] * input_dims[i-1];
      } else {
        m_inputStride[0] = 1;
      }
    }

    for (int i = 0; i < KernelDims; ++i) {
      const Index index = op.indices()[i];
      const Index input_dim = input_dims[index];
      const Index kernel_dim = kernel_dims[i];
      const Index result_dim = input_dim - kernel_dim + 1;
      m_dimensions[index] = result_dim;

      if (i > 0) {
        m_kernelStride[i] = m_kernelStride[i-1] * kernel_dims[i-1];
      } else {
        m_kernelStride[0] = 1;
      }
      m_indexStride[i] = m_inputStride[index];
    }

    for (int i = 0; i < NumDims; ++i) {
      if (i > 0) {
        m_outputStride[i] = m_outputStride[i-1] * m_dimensions[i-1];
      } else {
        m_outputStride[0] = 1;
      }
    }
  }

  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalSubExprsIfNeeded(Scalar*) {
    m_inputImpl.evalSubExprsIfNeeded(NULL);
    m_kernelImpl.evalSubExprsIfNeeded(NULL);
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_inputImpl.cleanup();
    m_kernelImpl.cleanup();
  }

  void evalTo(typename XprType::Scalar* buffer) const {
    for (int i = 0; i < dimensions().TotalSize(); ++i) {
      buffer[i] += coeff(i);
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    Index startInput = 0;
    for (int i = NumDims - 1; i >= 0; --i) {
      const Index idx = index / m_outputStride[i];
      startInput += idx * m_inputStride[i];
      index -= idx * m_outputStride[i];
    }

    CoeffReturnType result = CoeffReturnType(0);
    convolve(startInput, 0, 0, result);
    return result;
  }

  /* TODO: vectorization
  template<int LoadMode>
  EIGEN_DEVICE_FUNC PacketReturnType packet(Index index) const
  {
    assert(false);
  }*/

  EIGEN_DEVICE_FUNC void convolve(Index firstIndex, Index firstKernel, int DimIndex, CoeffReturnType& accum) const {
    for (int j = 0; j < m_kernelImpl.dimensions()[DimIndex]; ++j) {
      const Index input = firstIndex + j * m_indexStride[DimIndex];
      const Index kernel = firstKernel + j * m_kernelStride[DimIndex];
      if (DimIndex < KernelDims-1) {
        convolve(input, kernel, DimIndex+1, accum);
      } else {

        accum += m_inputImpl.coeff(input) * m_kernelImpl.coeff(kernel);
      }
    }
  }

 private:
  array<Index, NumDims> m_inputStride;
  array<Index, NumDims> m_outputStride;

  array<Index, KernelDims> m_indexStride;
  array<Index, KernelDims> m_kernelStride;
  TensorEvaluator<InputArgType, Device> m_inputImpl;
  TensorEvaluator<KernelArgType, Device> m_kernelImpl;
  Dimensions m_dimensions;
};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_CONVOLUTION_H
