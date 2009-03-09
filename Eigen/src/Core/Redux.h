// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
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
    Vectorization = int(MayLinearVectorize) ? int(LinearVectorization)
                  : int(MaySliceVectorize)  ? int(SliceVectorization)
                                            : int(NoVectorization)
  };
  
private:
  enum {
    Cost = Derived::SizeAtCompileTime * Derived::CoeffReadCost
           + (Derived::SizeAtCompileTime-1) * NumTraits<typename Derived::Scalar>::AddCost,
    UnrollingLimit = EIGEN_UNROLLING_LIMIT * (int(Vectorization) == int(NoVectorization) ? 1 : int(PacketSize))
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
         int Vectorization = ei_redux_traits<Func, Derived>::Vectorization,
         int Unrolling = ei_redux_traits<Func, Derived>::Unrolling
>
struct ei_redux_impl;

template<typename Func, typename Derived>
struct ei_redux_impl<Func, Derived, NoVectorization, NoUnrolling>
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
struct ei_redux_impl<Func,Derived, NoVectorization, CompleteUnrolling>
  : public ei_redux_novec_unroller<Func,Derived, 0, Derived::SizeAtCompileTime>
{};

template<typename Func, typename Derived>
struct ei_redux_impl<Func, Derived, LinearVectorization, NoUnrolling>
{
  typedef typename Derived::Scalar Scalar;
  typedef typename ei_packet_traits<Scalar>::type PacketScalar;

  static Scalar run(const Derived& mat, const Func& func)
  {
    const int size = mat.size();
    const int packetSize = ei_packet_traits<Scalar>::size;
    const int alignedStart =  (Derived::Flags & AlignedBit)
                           || !(Derived::Flags & DirectAccessBit)
                           ? 0
                           : ei_alignmentOffset(&mat.const_cast_derived().coeffRef(0), size);
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
struct ei_redux_impl<Func, Derived, SliceVectorization, NoUnrolling>
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
      res = ei_redux_impl<Func, Derived, NoVectorization, NoUnrolling>::run(mat, func);
    }

    return res;
  }
};

template<typename Func, typename Derived>
struct ei_redux_impl<Func, Derived, LinearVectorization, CompleteUnrolling>
{
  typedef typename Derived::Scalar Scalar;
  typedef typename ei_packet_traits<Scalar>::type PacketScalar;
  enum {
    PacketSize = ei_packet_traits<Scalar>::size,
    Size = Derived::SizeAtCompileTime,
    VectorizationSize = (Size / PacketSize) * PacketSize
  };
  EIGEN_STRONG_INLINE static Scalar run(const Derived& mat, const Func& func)
  {
    Scalar res = func.predux(ei_redux_vec_unroller<Func, Derived, 0, Size / PacketSize>::run(mat,func));
    if (VectorizationSize != Size)
      res = func(res,ei_redux_novec_unroller<Func, Derived, VectorizationSize, Size-VectorizationSize>::run(mat,func));
    return res;
  }
};


/** \returns the result of a full redux operation on the whole matrix or vector using \a func
  *
  * The template parameter \a BinaryOp is the type of the functor \a func which must be
  * an assiociative operator. Both current STL and TR1 functor styles are handled.
  *
  * \sa MatrixBase::sum(), MatrixBase::minCoeff(), MatrixBase::maxCoeff(), MatrixBase::colwise(), MatrixBase::rowwise()
  */
template<typename Derived>
template<typename Func>
inline typename ei_result_of<Func(typename ei_traits<Derived>::Scalar)>::type
MatrixBase<Derived>::redux(const Func& func) const
{
  typename Derived::Nested nested(derived());
  typedef typename ei_cleantype<typename Derived::Nested>::type ThisNested;
  return ei_redux_impl<Func, ThisNested>
            ::run(nested, func);
}

/** \returns the minimum of all coefficients of *this
  */
template<typename Derived>
EIGEN_STRONG_INLINE typename ei_traits<Derived>::Scalar
MatrixBase<Derived>::minCoeff() const
{
  return this->redux(Eigen::ei_scalar_min_op<Scalar>());
}

/** \returns the maximum of all coefficients of *this
  */
template<typename Derived>
EIGEN_STRONG_INLINE typename ei_traits<Derived>::Scalar
MatrixBase<Derived>::maxCoeff() const
{
  return this->redux(Eigen::ei_scalar_max_op<Scalar>());
}

/** \returns the sum of all coefficients of *this
  *
  * \sa trace(), prod()
  */
template<typename Derived>
EIGEN_STRONG_INLINE typename ei_traits<Derived>::Scalar
MatrixBase<Derived>::sum() const
{
  return this->redux(Eigen::ei_scalar_sum_op<Scalar>());
}

/** \returns the product of all coefficients of *this
  *
  * Example: \include MatrixBase_prod.cpp
  * Output: \verbinclude MatrixBase_prod.out
  *
  * \sa sum()
  */
template<typename Derived>
EIGEN_STRONG_INLINE typename ei_traits<Derived>::Scalar
MatrixBase<Derived>::prod() const
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
  return diagonal().sum();
}

#endif // EIGEN_REDUX_H
