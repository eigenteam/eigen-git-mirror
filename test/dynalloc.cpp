// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#include "main.h"

// test compilation with both a struct and a class...
struct MyStruct
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW(MyStruct)
  char dummychar;
  Vector4f avec;
};

class MyClassA
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW(MyClassA)
    char dummychar;
    Vector4f avec;
};

template<typename T> void check_dynaligned()
{
  T* obj = new T;
  VERIFY(size_t(obj)%16==0);
  delete obj;
}

void test_dynalloc()
{

#ifdef EIGEN_VECTORIZE
  for (int i=0; i<g_repeat*100; ++i)
  {
    CALL_SUBTEST( check_dynaligned<Vector4f>() );
    CALL_SUBTEST( check_dynaligned<Vector2d>() );
    CALL_SUBTEST( check_dynaligned<Matrix4f>() );
    CALL_SUBTEST( check_dynaligned<Vector4d>() );
    CALL_SUBTEST( check_dynaligned<Vector4i>() );
  }
  
  // check static allocation, who knows ?
  {
    MyStruct foo0;  VERIFY(size_t(foo0.avec.data())%16==0);
    MyClassA fooA;  VERIFY(size_t(fooA.avec.data())%16==0);
  }

  // dynamic allocation, single object
  for (int i=0; i<g_repeat*100; ++i)
  {
    MyStruct *foo0 = new MyStruct();  VERIFY(size_t(foo0->avec.data())%16==0);
    MyClassA *fooA = new MyClassA();  VERIFY(size_t(fooA->avec.data())%16==0);
    delete foo0;
    delete fooA;
  }

  // dynamic allocation, array
  const int N = 10;
  for (int i=0; i<g_repeat*100; ++i)
  {
    MyStruct *foo0 = new MyStruct[N];  VERIFY(size_t(foo0->avec.data())%16==0);
    MyClassA *fooA = new MyClassA[N];  VERIFY(size_t(fooA->avec.data())%16==0);
    delete[] foo0;
    delete[] fooA;
  }

  // std::vector
  for (int i=0; i<g_repeat*100; ++i)
  {
    std::vector<Vector4f, ei_new_allocator<Vector4f> > vecs(N);
    for (int j=0; j<N; ++j)
    {
      VERIFY(size_t(vecs[j].data())%16==0);
    }
    std::vector<MyStruct,ei_new_allocator<MyStruct> > foos(N);
    for (int j=0; j<N; ++j)
    {
      VERIFY(size_t(foos[j].avec.data())%16==0);
    }
  }
  
#endif // EIGEN_VECTORIZE

}
