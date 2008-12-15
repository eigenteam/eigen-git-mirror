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

#ifndef EIGEN_MEMORY_H
#define EIGEN_MEMORY_H

#ifdef EIGEN_VECTORIZE
#ifndef _MSC_VER
// it seems we cannot assume posix_memalign is defined in the stdlib header
extern "C" int posix_memalign (void **, size_t, size_t) throw ();
#endif
#endif

/** \internal
  * Static array automatically aligned if the total byte size is a multiple of 16
  */
template <typename T, int Size, bool Align> struct ei_aligned_array
{
  EIGEN_ALIGN_128 T array[Size];

  ei_aligned_array()
  {
    ei_assert((reinterpret_cast<size_t>(array) & 0xf) == 0
              && "this assertion is explained here: http://eigen.tuxfamily.org/api/UnalignedArrayAssert.html  **** READ THIS WEB PAGE !!! ****");
  }
};

template <typename T, int Size> struct ei_aligned_array<T,Size,false>
{
  T array[Size];
};

struct ei_byte_forcing_aligned_malloc
{
  unsigned char c; // sizeof must be 1.
};

template<typename T> struct ei_force_aligned_malloc { enum { ret = 0 }; };
template<> struct ei_force_aligned_malloc<ei_byte_forcing_aligned_malloc> { enum { ret = 1 }; };

/** \internal allocates \a size * sizeof(\a T) bytes with 16 bytes alignment.
  * */
template<typename T>
inline T* ei_aligned_malloc(size_t size)
{
  #ifdef EIGEN_VECTORIZE
  if(ei_packet_traits<T>::size>1 || ei_force_aligned_malloc<T>::ret)
  {
    #ifdef _MSC_VER
      return static_cast<T*>(_aligned_malloc(size*sizeof(T), 16));
    #else
      void* ptr;
      if(posix_memalign(&ptr, 16, size*sizeof(T))==0)
        return static_cast<T*>(ptr);
      else
        return 0;
    #endif
  }
  else
  #endif
    return new T[size]; // here we really want a new, not a malloc. Justification: if the user uses Eigen on
      // some fancy scalar type such as multiple-precision numbers, and this type has a custom operator new,
      // then we want to honor this operator new! Anyway this type won't have vectorization so the vectorizing path
      // is irrelevant here. Yes, we should say somewhere in the docs that if the user uses a custom scalar type then
      // he can't have both vectorization and a custom operator new on his scalar type.
}

/** \internal free memory allocated with ei_aligned_malloc */
template<typename T>
inline void ei_aligned_free(T* ptr)
{
  #ifdef EIGEN_VECTORIZE
  if (ei_packet_traits<T>::size>1 || ei_force_aligned_malloc<T>::ret)
  #ifdef _MSC_VER
    _aligned_free(ptr);
  #else
    free(ptr);
  #endif
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
          ? std::min<int>( (PacketSize - (int((size_t(ptr)/sizeof(Scalar))) & PacketAlignedMask))
                           & PacketAlignedMask, maxOffset)
          : 0;
}

/** \internal
  * ei_alloc_stack(TYPE,SIZE) allocates sizeof(TYPE)*SIZE bytes on the stack if sizeof(TYPE)*SIZE is
  * smaller than EIGEN_STACK_ALLOCATION_LIMIT. Otherwise the memory is allocated using the operator new.
  * Data allocated with ei_alloc_stack \b must be freed by calling ei_free_stack(PTR,TYPE,SIZE).
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

/** \class WithAlignedOperatorNew
  *
  * \brief Enforces instances of inherited classes to be 16 bytes aligned when allocated with operator new
  *
  * When Eigen's explicit vectorization is enabled, Eigen assumes that some fixed sizes types are aligned
  * on a 16 bytes boundary. Those include all Matrix types having a sizeof multiple of 16 bytes, e.g.:
  *  - Vector2d, Vector4f, Vector4i, Vector4d,
  *  - Matrix2d, Matrix4f, Matrix4i, Matrix4d,
  *  - etc.
  * When an object is statically allocated, the compiler will automatically and always enforces 16 bytes
  * alignment of the data when needed. However some troubles might appear when data are dynamically allocated.
  * Let's pick an example:
  * \code
  * struct Foo {
  *   char dummy;
  *   Vector4f some_vector;
  * };
  * Foo obj1;                           // static allocation
  * obj1.some_vector = Vector4f(..);    // =>   OK
  *
  * Foo *pObj2 = new Foo;               // dynamic allocation
  * pObj2->some_vector = Vector4f(..);  // =>  !! might segfault !!
  * \endcode
  * Here, the problem is that operator new is not aware of the compile time alignment requirement of the
  * type Vector4f (and hence of the type Foo). Therefore "new Foo" does not necessarily returns a 16 bytes
  * aligned pointer. The purpose of the class WithAlignedOperatorNew is exactly to overcome this issue by
  * overloading the operator new to return aligned data when the vectorization is enabled.
  * Here is a similar safe example:
  * \code
  * struct Foo : WithAlignedOperatorNew {
  *   char dummy;
  *   Vector4f some_vector;
  * };
  * Foo *pObj2 = new Foo;               // dynamic allocation
  * pObj2->some_vector = Vector4f(..);  // =>  SAFE !
  * \endcode
  *
  * \sa class ei_new_allocator
  */
struct WithAlignedOperatorNew
{
  #ifdef EIGEN_VECTORIZE

  void *operator new(size_t size) throw()
  {
    return ei_aligned_malloc<ei_byte_forcing_aligned_malloc>(size);
  }

  void *operator new[](size_t size) throw()
  {
    return ei_aligned_malloc<ei_byte_forcing_aligned_malloc>(size);
  }

  void operator delete(void * ptr) { ei_aligned_free(static_cast<ei_byte_forcing_aligned_malloc *>(ptr)); }
  void operator delete[](void * ptr) { ei_aligned_free(static_cast<ei_byte_forcing_aligned_malloc *>(ptr)); }

  #endif
};

template<typename T, int SizeAtCompileTime,
         bool NeedsToAlign = (SizeAtCompileTime!=Dynamic) && ((sizeof(T)*SizeAtCompileTime)%16==0)>
struct ei_with_aligned_operator_new : WithAlignedOperatorNew {};

template<typename T, int SizeAtCompileTime>
struct ei_with_aligned_operator_new<T,SizeAtCompileTime,false> {};

/** \class ei_new_allocator
  *
  * \brief stl compatible allocator to use with with fixed-size vector and matrix types
  *
  * STL allocator simply wrapping operators new[] and delete[]. Unlike GCC's default new_allocator,
  * ei_new_allocator call operator new on the type \a T and not the general new operator ignoring
  * overloaded version of operator new.
  *
  * Example:
  * \code
  * // Vector4f requires 16 bytes alignment:
  * std::vector<Vector4f,ei_new_allocator<Vector4f> > dataVec4;
  * // Vector3f does not require 16 bytes alignment, no need to use Eigen's allocator:
  * std::vector<Vector3f> dataVec3;
  *
  * struct Foo : WithAlignedOperatorNew {
  *   char dummy;
  *   Vector4f some_vector;
  * };
  * std::vector<Foo,ei_new_allocator<Foo> > dataFoo;
  * \endcode
  *
  * \sa class WithAlignedOperatorNew
  */
template<typename T> class ei_new_allocator
{
  public:
    typedef T         value_type;
    typedef T*        pointer;
    typedef const T*  const_pointer;
    typedef T&        reference;
    typedef const T&  const_reference;

    template<typename OtherType>
    struct rebind
    { typedef ei_new_allocator<OtherType> other; };

    T* address(T& ref) const { return &ref; }
    const T* address(const T& ref) const { return &ref; }
    T* allocate(size_t size, const void* = 0) { return new T[size]; }
    void deallocate(T* ptr, size_t) { delete[] ptr; }
    size_t max_size() const { return size_t(-1) / sizeof(T); }
    // FIXME I'm note sure about this construction...
    void construct(T* ptr, const T& refObj) { ::new(ptr) T(refObj); }
    void destroy(T* ptr) { ptr->~T(); }
};

#endif // EIGEN_MEMORY_H
