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

#ifdef __linux
// it seems we cannot assume posix_memalign is defined in the stdlib header
extern "C" int posix_memalign (void **, size_t, size_t) throw ();
#endif

/** \internal allocates \a size bytes. The returned pointer is guaranteed to have 16 bytes alignment.
  * On allocation error, the returned pointer is undefined, but if exceptions are enabled then a std::bad_alloc is thrown.
  */
inline void* ei_aligned_malloc(size_t size)
{
  #ifdef EIGEN_NO_MALLOC
    ei_assert(false && "heap allocation is forbidden (EIGEN_NO_MALLOC is defined)");
  #endif

  void *result;
  #ifdef __linux
    #ifdef EIGEN_EXCEPTIONS
      const int failed =
    #endif
    posix_memalign(&result, 16, size);
  #else
    #ifdef _MSC_VER
      result = _aligned_malloc(size, 16);
    #elif defined(__APPLE__)
      result = malloc(size); // Apple's malloc() already returns 16-byte-aligned ptrs
    #else
      result = _mm_malloc(size, 16);
    #endif
    #ifdef EIGEN_EXCEPTIONS
      const int failed = (result == 0);
    #endif
  #endif
  #ifdef EIGEN_EXCEPTIONS
    if(failed)
      throw std::bad_alloc();
  #endif
  return result;
}

/** allocates \a size bytes. If Align is true, then the returned ptr is 16-byte-aligned.
  * On allocation error, the returned pointer is undefined, but if exceptions are enabled then a std::bad_alloc is thrown.
  */
template<bool Align> inline void* ei_conditional_aligned_malloc(size_t size)
{
  return ei_aligned_malloc(size);
}

template<> inline void* ei_conditional_aligned_malloc<false>(size_t size)
{
  void *void_result = malloc(size);
  #ifdef EIGEN_EXCEPTIONS
    if(!void_result) throw std::bad_alloc();
  #endif
  return void_result;
}

/** allocates \a size objects of type T. The returned pointer is guaranteed to have 16 bytes alignment.
  * On allocation error, the returned pointer is undefined, but if exceptions are enabled then a std::bad_alloc is thrown.
  * The default constructor of T is called.
  */
template<typename T> T* ei_aligned_new(size_t size)
{
  void *void_result = ei_aligned_malloc(sizeof(T)*size);
  return ::new(void_result) T[size];
}

template<typename T, bool Align> T* ei_conditional_aligned_new(size_t size)
{
  void *void_result = ei_conditional_aligned_malloc<Align>(sizeof(T)*size);
  return ::new(void_result) T[size];
}

/** \internal free memory allocated with ei_aligned_malloc
  */
inline void ei_aligned_free(void *ptr)
{
  #if defined(__linux)
    free(ptr);
  #elif defined(__APPLE__)
    free(ptr);
  #elif defined(_MSC_VER)
    _aligned_free(ptr);
  #else
    _mm_free(ptr);
  #endif
}

/** \internal free memory allocated with ei_conditional_aligned_malloc
  */
template<bool Align> inline void ei_conditional_aligned_free(void *ptr)
{
  ei_aligned_free(ptr);
}

template<> void ei_conditional_aligned_free<false>(void *ptr)
{
  free(ptr);
}

/** \internal delete the elements of an array.
  * The \a size parameters tells on how many objects to call the destructor of T.
  */
template<typename T> inline void ei_delete_elements_of_array(T *ptr, size_t size)
{
  // always destruct an array starting from the end.
  while(size) ptr[--size].~T();
}

/** \internal delete objects constructed with ei_aligned_new
  * The \a size parameters tells on how many objects to call the destructor of T.
  */
template<typename T> void ei_aligned_delete(T *ptr, size_t size)
{
  ei_delete_elements_of_array<T>(ptr, size);
  ei_aligned_free(ptr);
}

/** \internal delete objects constructed with ei_conditional_aligned_new
  * The \a size parameters tells on how many objects to call the destructor of T.
  */
template<typename T, bool Align> inline void ei_conditional_aligned_delete(T *ptr, size_t size)
{
  ei_delete_elements_of_array<T>(ptr, size);
  ei_conditional_aligned_free<Align>(ptr);
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
  * ei_aligned_stack_alloc(SIZE) allocates an aligned buffer of SIZE bytes
  * on the stack if SIZE is smaller than EIGEN_STACK_ALLOCATION_LIMIT.
  * Otherwise the memory is allocated on the heap.
  * Data allocated with ei_aligned_stack_alloc \b must be freed by calling ei_aligned_stack_free(PTR,SIZE).
  * \code
  * float * data = ei_aligned_stack_alloc(float,array.size());
  * // ...
  * ei_aligned_stack_free(data,float,array.size());
  * \endcode
  */
#ifdef __linux__
  #define ei_aligned_stack_alloc(SIZE) (SIZE<=EIGEN_STACK_ALLOCATION_LIMIT) \
                                    ? alloca(SIZE) \
                                    : ei_aligned_malloc(SIZE)
  #define ei_aligned_stack_free(PTR,SIZE) if(SIZE>EIGEN_STACK_ALLOCATION_LIMIT) ei_aligned_free(PTR)
#else
  #define ei_aligned_stack_alloc(SIZE) ei_aligned_malloc(SIZE)
  #define ei_aligned_stack_free(PTR,SIZE) ei_aligned_free(PTR)
#endif

#define ei_aligned_stack_new(TYPE,SIZE) ::new(ei_aligned_stack_alloc(sizeof(TYPE)*SIZE)) TYPE[SIZE]
#define ei_aligned_stack_delete(TYPE,PTR,SIZE) ei_delete_elements_of_array<TYPE>(PTR, SIZE); \
                                               ei_aligned_stack_free(PTR,sizeof(TYPE)*SIZE)

/** \brief Overloads the operator new and delete of the class Type with operators that are aligned if NeedsToAlign is true
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
  * struct Foo {
  *   EIGEN_MAKE_ALIGNED_OPERATOR_NEW(Foo)
  *   char dummy;
  *   Vector4f some_vector;
  * };
  * Foo *pObj2 = new Foo;               // dynamic allocation
  * pObj2->some_vector = Vector4f(..);  // =>  SAFE !
  * \endcode
  *
  * \sa class ei_new_allocator
  */
#define EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(Type,NeedsToAlign) \
    void *operator new(size_t size) throw() { \
      return Eigen::ei_conditional_aligned_malloc<NeedsToAlign>(size); \
    } \
    void *operator new[](size_t size) throw() { \
      return Eigen::ei_conditional_aligned_malloc<NeedsToAlign>(size); \
    } \
    void operator delete(void * ptr) { Eigen::ei_aligned_free(ptr); } \
    void operator delete[](void * ptr) { Eigen::ei_aligned_free(ptr); }

#define EIGEN_MAKE_ALIGNED_OPERATOR_NEW(Type) EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(Type,true)
#define EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE(Type,Scalar,Size) \
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(Type,((Size)!=Eigen::Dynamic) && ((sizeof(Scalar)*(Size))%16==0))

/** Deprecated, use the EIGEN_MAKE_ALIGNED_OPERATOR_NEW(Class) macro instead in your own class */
struct WithAlignedOperatorNew
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW(WithAlignedOperatorNew)
};

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
