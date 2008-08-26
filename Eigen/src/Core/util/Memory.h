// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_MEMORY_H
#define EIGEN_MEMORY_H

#ifdef EIGEN_VECTORIZE
// it seems we cannot assume posix_memalign is defined in the stdlib header
extern "C" int posix_memalign (void **, size_t, size_t) throw ();
#endif

/** \internal
  * Static array automatically aligned if the total byte size is a multiple of 16
  */
template <typename T, int Size, bool Align> struct ei_aligned_array
{
  EIGEN_ALIGN_128 T array[Size];
};

template <typename T, int Size> struct ei_aligned_array<T,Size,false>
{
  T array[Size];
};

/** \internal allocates \a size * sizeof(\a T) bytes with a 16 bytes based alignement */
template<typename T>
inline T* ei_aligned_malloc(size_t size)
{
  #ifdef EIGEN_VECTORIZE
  if (ei_packet_traits<T>::size>1)
  {
    void* ptr;
    if (posix_memalign(&ptr, 16, size*sizeof(T))==0)
      return static_cast<T*>(ptr);
    else
      return 0;
  }
  else
  #endif
    return new T[size];
}

/** \internal free memory allocated with ei_aligned_malloc */
template<typename T>
inline void ei_aligned_free(T* ptr)
{
  #ifdef EIGEN_VECTORIZE
  if (ei_packet_traits<T>::size>1)
    free(ptr);
  else
  #endif
    delete[] ptr;
}

/** \internal \returns the number of elements which have to be skipped such that data are 16 bytes aligned */
template<typename Scalar>
inline static int ei_alignmentOffset(const Scalar* ptr, int maxOffset)
{
  typedef typename ei_packet_traits<Scalar>::type Packet;
  const int PacketSize = ei_packet_traits<Scalar>::size;
  const int PacketAlignedMask = PacketSize-1;
  const bool Vectorized = PacketSize>1;
  return Vectorized
          ? std::min<int>( (PacketSize - ((size_t(ptr)/sizeof(Scalar)) & PacketAlignedMask))
                           & PacketAlignedMask, maxOffset)
          : 0;
}

/** \internal
  * ei_alloc_stack(TYPE,SIZE) allocates sizeof(TYPE)*SIZE bytes on the stack if sizeof(TYPE)*SIZE is
  * smaller than EIGEN_STACK_ALLOCATION_LIMIT. Otherwise the memory is allocated using the operator new.
  * Data allocated with ei_alloc_stack \b must be freed calling ei_free_stack(PTR,TYPE,SIZE).
  * \code
  * float * data = ei_alloc_stack(float,array.size());
  * // ...
  * ei_free_stack(data,float,array.size());
  * \endcode
  */
#ifdef __linux__
# define ei_alloc_stack(TYPE,SIZE) ((sizeof(TYPE)*(SIZE)>16000000) ? new TYPE[SIZE] : (TYPE*)alloca(sizeof(TYPE)*(SIZE)))
# define ei_free_stack(PTR,TYPE,SIZE) if (sizeof(TYPE)*SIZE>16000000) delete[] PTR
#else
# define ei_alloc_stack(TYPE,SIZE) new TYPE[SIZE]
# define ei_free_stack(PTR,TYPE,SIZE) delete[] PTR
#endif

#endif // EIGEN_MEMORY_H
