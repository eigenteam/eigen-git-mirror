// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_REDUX_H
#define EIGEN_REDUX_H

// TODO
//  * implement other kind of vectorization
//  * factorize code

/***************************************************************************
* Part 1 : the logic deciding a strategy for vectorization and unrolling
***************************************************************************/

template<typename Func, typename Derived>
struct ei_redux_traits
{
private:
  enum {
    PacketSize = ei_packet_traits<typename Derived::Scalar>::size,
    InnerMaxSize = int(Derived::Flags)&RowMajorBit
                 ? Derived::MaxColsAtCompileTime
                 : Derived::MaxRowsAtCompileTime
  };

  enum {
    MightVectorize = (int(Derived::Flags)&ActualPacketAccessBit)
                  && (ei_functor_traits<Func>::PacketAccess),
    MayLinearVectorize = MightVectorize && (int(Derived::Flags)&LinearAccessBit),
    MaySliceVectorize  = MightVectorize && int(InnerMaxSize)>=3*PacketSize
  };

public:
  enum {
    Traversal = int(MayLinearVectorize) ? int(LinearVectorizedTraversal)
                  : int(MaySliceVectorize)  ? int(SliceVectorizedTraversal)
                                            : int(DefaultTraversal)
  };

private:
  enum {
    Cost = Derived::SizeAtCompileTime * Derived::CoeffReadCost
           + (Derived::SizeAtCompileTime-1) * NumTraits<typename Derived::Scalar>::AddCost,
    UnrollingLimit = EIGEN_UNROLLING_LIMIT * (int(Traversal) == int(DefaultTraversal) ? 1 : int(PacketSize))
  };

public:
  enum {
    Unrolling = Cost <= UnrollingLimit
              ? CompleteUnrolling
              : NoUnrolling
  };
};

/***************************************************************************
* Part 2 : unrollers
***************************************************************************/

/*** no vectorization ***/

template<typename Func, typename Derived, int Start, int Length>
struct ei_redux_novec_unroller
{
  enum {
    HalfLength = Length/2
  };

  typedef typename Derived::Scalar Scalar;

  EIGEN_STRONG_INLINE static Scalar run(const Derived &mat, const Func& func)
  {
    return func(ei_redux_novec_unroller<Func, Derived, Start, HalfLength>::run(mat,func),
                ei_redux_novec_unroller<Func, Derived, Start+HalfLength, Length-HalfLength>::run(mat,func));
  }
};

template<typename Func, typename Derived, int Start>
struct ei_redux_novec_unroller<Func, Derived, Start, 1>
{
  enum {
    col = Start / Derived::RowsAtCompileTime,
    row = Start % Derived::RowsAtCompileTime
  };

  typedef typename Derived::Scalar Scalar;

  EIGEN_STRONG_INLINE static Scalar run(const Derived &mat, const Func&)
  {
    return mat.coeff(row, col);
  }
};

// This is actually dead code and will never be called. It is required
// to prevent false warnings regarding failed inlining though
// for 0 length run() will never be called at all.
template<typename Func, typename Derived, int Start>
struct ei_redux_novec_unroller<Func, Derived, Start, 0>
{
  typedef typename Derived::Scalar Scalar;
  EIGEN_STRONG_INLINE static Scalar run(const Derived&, const Func&) { return Scalar(); }
};

/*** vectorization ***/

template<typename Func, typename Derived, int Start, int Length>
struct ei_redux_vec_unroller
{
  enum {
    PacketSize = ei_packet_traits<typename Derived::Scalar>::size,
    HalfLength = Length/2
  };

  typedef typename Derived::Scalar Scalar;
  typedef typename ei_packet_traits<Scalar>::type PacketScalar;

  EIGEN_STRONG_INLINE static PacketScalar run(const Derived &mat, const Func& func)
  {
    return func.packetOp(
            ei_redux_vec_unroller<Func, Derived, Start, HalfLength>::run(mat,func),
            ei_redux_vec_unroller<Func, Derived, Start+HalfLength, Length-HalfLength>::run(mat,func) );
  }
};

template<typename Func, typename Derived, int Start>
struct ei_redux_vec_unroller<Func, Derived, Start, 1>
{
  enum {
    index = Start * ei_packet_traits<typename Derived::Scalar>::size,
    row = int(Derived::Flags)&RowMajorBit
        ? index / int(Derived::ColsAtCompileTime)
        : index % Derived::RowsAtCompileTime,
    col = int(Derived::Flags)&RowMajorBit
        ? index % int(Derived::ColsAtCompileTime)
        : index / Derived::RowsAtCompileTime,
    alignment = (Derived::Flags & AlignedBit) ? Aligned : Unaligned
  };

  typedef typename Derived::Scalar Scalar;
  typedef typename ei_packet_traits<Scalar>::type PacketScalar;

  EIGEN_STRONG_INLINE static PacketScalar run(const Derived &mat, const Func&)
  {
    return mat.template packet<alignment>(row, col);
  }
};

/***************************************************************************
* Part 3 : implementation of all cases
***************************************************************************/

template<typename Func, typename Derived,
         int Traversal = ei_redux_traits<Func, Derived>::Traversal,
         int Unrolling = ei_redux_traits<Func, Derived>::Unrolling
>
struct ei_redux_impl;

template<typename Func, typename Derived>
struct ei_redux_impl<Func, Derived, DefaultTraversal, NoUnrolling>
{
  typedef typename Derived::Scalar Scalar;
  static Scalar run(const Derived& mat, const Func& func)
  {
    ei_assert(mat.rows()>0 && mat.cols()>0 && "you are using a non initialized matrix");
    Scalar res;
    res = mat.coeff(0, 0);
    for(int i = 1; i < mat.rows(); ++i)
      res = func(res, mat.coeff(i, 0));
    for(int j = 1; j < mat.cols(); ++j)
      for(int i = 0; i < mat.rows(); ++i)
        res = func(res, mat.coeff(i, j));
    return res;
  }
};

template<typename Func, typename Derived>
struct ei_redux_impl<Func,Derived, DefaultTraversal, CompleteUnrolling>
  : public ei_redux_novec_unroller<Func,Derived, 0, Derived::SizeAtCompileTime>
{};

template<typename Func, typename Derived>
struct ei_redux_impl<Func, Derived, LinearVectorizedTraversal, NoUnrolling>
{
  typedef typename Derived::Scalar Scalar;
  typedef typename ei_packet_traits<Scalar>::type PacketScalar;

  static Scalar run(const Derived& mat, const Func& func)
  {
    const int size = mat.size();
    const int packetSize = ei_packet_traits<Scalar>::size;
    const int alignedStart = ei_first_aligned(mat);
    enum {
      alignment = (Derived::Flags & DirectAccessBit) || (Derived::Flags & AlignedBit)
                ? Aligned : Unaligned
    };
    const int alignedSize = ((size-alignedStart)/packetSize)*packetSize;
    const int alignedEnd = alignedStart + alignedSize;
    Scalar res;
    if(alignedSize)
    {
      PacketScalar packet_res = mat.template packet<alignment>(alignedStart);
      for(int index = alignedStart + packetSize; index < alignedEnd; index += packetSize)
        packet_res = func.packetOp(packet_res, mat.template packet<alignment>(index));
      res = func.predux(packet_res);

      for(int index = 0; index < alignedStart; ++index)
        res = func(res,mat.coeff(index));

      for(int index = alignedEnd; index < size; ++index)
        res = func(res,mat.coeff(index));
    }
    else // too small to vectorize anything.
         // since this is dynamic-size hence inefficient anyway for such small sizes, don't try to optimize.
    {
      res = mat.coeff(0);
      for(int index = 1; index < size; ++index)
        res = func(res,mat.coeff(index));
    }

    return res;
  }
};

template<typename Func, typename Derived>
struct ei_redux_impl<Func, Derived, SliceVectorizedTraversal, NoUnrolling>
{
  typedef typename Derived::Scalar Scalar;
  typedef typename ei_packet_traits<Scalar>::type PacketScalar;

  static Scalar run(const Derived& mat, const Func& func)
  {
    const int innerSize = mat.innerSize();
    const int outerSize = mat.outerSize();
    enum {
      packetSize = ei_packet_traits<Scalar>::size,
      isRowMajor = Derived::Flags&RowMajorBit?1:0
    };
    const int packetedInnerSize = ((innerSize)/packetSize)*packetSize;
    Scalar res;
    if(packetedInnerSize)
    {
      PacketScalar packet_res = mat.template packet<Unaligned>(0,0);
      for(int j=0; j<outerSize; ++j)
        for(int i=0; i<packetedInnerSize; i+=int(packetSize))
          packet_res = func.packetOp(packet_res, mat.template packet<Unaligned>
                                                 (isRowMajor?j:i, isRowMajor?i:j));

      res = func.predux(packet_res);
      for(int j=0; j<outerSize; ++j)
        for(int i=packetedInnerSize; i<innerSize; ++i)
          res = func(res, mat.coeff(isRowMajor?j:i, isRowMajor?i:j));
    }
    else // too small to vectorize anything.
         // since this is dynamic-size hence inefficient anyway for such small sizes, don't try to optimize.
    {
      res = ei_redux_impl<Func, Derived, DefaultTraversal, NoUnrolling>::run(mat, func);
    }

    return res;
  }
};

template<typename Func, typename Derived>
struct ei_redux_impl<Func, Derived, LinearVectorizedTraversal, CompleteUnrolling>
{
  typedef typename Derived::Scalar Scalar;
  typedef typename ei_packet_traits<Scalar>::type PacketScalar;
  enum {
    PacketSize = ei_packet_traits<Scalar>::size,
    Size = Derived::SizeAtCompileTime,
    VectorizedSize = (Size / PacketSize) * PacketSize
  };
  EIGEN_STRONG_INLINE static Scalar run(const Derived& mat, const Func& func)
  {
    Scalar res = func.predux(ei_redux_vec_unroller<Func, Derived, 0, Size / PacketSize>::run(mat,func));
    if (VectorizedSize != Size)
      res = func(res,ei_redux_novec_unroller<Func, Derived, VectorizedSize, Size-VectorizedSize>::run(mat,func));
    return res;
  }
};


/** \returns the result of a full redux operation on the whole matrix or vector using \a func
  *
  * The template parameter \a BinaryOp is the type of the functor \a func which must be
  * an associative operator. Both current STL and TR1 functor styles are handled.
  *
  * \sa DenseBase::sum(), DenseBase::minCoeff(), DenseBase::maxCoeff(), MatrixBase::colwise(), MatrixBase::rowwise()
  */
template<typename Derived>
template<typename Func>
inline typename ei_result_of<Func(typename ei_traits<Derived>::Scalar)>::type
DenseBase<Derived>::redux(const Func& func) const
{
  typedef typename ei_cleantype<typename Derived::Nested>::type ThisNested;
  return ei_redux_impl<Func, ThisNested>
            ::run(derived(), func);
}

/** \returns the minimum of all coefficients of *this
  */
template<typename Derived>
EIGEN_STRONG_INLINE typename ei_traits<Derived>::Scalar
DenseBase<Derived>::minCoeff() const
{
  return this->redux(Eigen::ei_scalar_min_op<Scalar>());
}

/** \returns the maximum of all coefficients of *this
  */
template<typename Derived>
EIGEN_STRONG_INLINE typename ei_traits<Derived>::Scalar
DenseBase<Derived>::maxCoeff() const
{
  return this->redux(Eigen::ei_scalar_max_op<Scalar>());
}

/** \returns the sum of all coefficients of *this
  *
  * \sa trace(), prod(), mean()
  */
template<typename Derived>
EIGEN_STRONG_INLINE typename ei_traits<Derived>::Scalar
DenseBase<Derived>::sum() const
{
  return this->redux(Eigen::ei_scalar_sum_op<Scalar>());
}

/** \returns the mean of all coefficients of *this
*
* \sa trace(), prod(), sum()
*/
template<typename Derived>
EIGEN_STRONG_INLINE typename ei_traits<Derived>::Scalar
DenseBase<Derived>::mean() const
{
  return this->redux(Eigen::ei_scalar_sum_op<Scalar>()) / Scalar(this->size());
}

/** \returns the product of all coefficients of *this
  *
  * Example: \include MatrixBase_prod.cpp
  * Output: \verbinclude MatrixBase_prod.out
  *
  * \sa sum(), mean(), trace()
  */
template<typename Derived>
EIGEN_STRONG_INLINE typename ei_traits<Derived>::Scalar
DenseBase<Derived>::prod() const
{
  return this->redux(Eigen::ei_scalar_product_op<Scalar>());
}

/** \returns the trace of \c *this, i.e. the sum of the coefficients on the main diagonal.
  *
  * \c *this can be any matrix, not necessarily square.
  *
  * \sa diagonal(), sum()
  */
template<typename Derived>
EIGEN_STRONG_INLINE typename ei_traits<Derived>::Scalar
MatrixBase<Derived>::trace() const
{
  return derived().diagonal().sum();
}

#endif // EIGEN_REDUX_H
