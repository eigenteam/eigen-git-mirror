// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
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

void test_meta()
{
  typedef float & FloatRef;
  typedef const float & ConstFloatRef;
  
  VERIFY((internal::meta_if<(3<4),internal::meta_true, internal::meta_false>::ret::ret));
  VERIFY(( internal::is_same_type<float,float>::ret));
  VERIFY((!internal::is_same_type<float,double>::ret));
  VERIFY((!internal::is_same_type<float,float&>::ret));
  VERIFY((!internal::is_same_type<float,const float&>::ret));
  
  VERIFY(( internal::is_same_type<float,internal::cleantype<const float&>::type >::ret));
  VERIFY(( internal::is_same_type<float,internal::cleantype<const float*>::type >::ret));
  VERIFY(( internal::is_same_type<float,internal::cleantype<const float*&>::type >::ret));
  VERIFY(( internal::is_same_type<float,internal::cleantype<float**>::type >::ret));
  VERIFY(( internal::is_same_type<float,internal::cleantype<float**&>::type >::ret));
  VERIFY(( internal::is_same_type<float,internal::cleantype<float* const *&>::type >::ret));
  VERIFY(( internal::is_same_type<float,internal::cleantype<float* const>::type >::ret));

  VERIFY(( internal::is_same_type<float*,internal::unconst<const float*>::type >::ret));
  VERIFY(( internal::is_same_type<float&,internal::unconst<const float&>::type >::ret));
  VERIFY(( internal::is_same_type<float&,internal::unconst<ConstFloatRef>::type >::ret));
  
  VERIFY(( internal::is_same_type<float&,internal::unconst<float&>::type >::ret));
  VERIFY(( internal::is_same_type<float,internal::unref<float&>::type >::ret));
  VERIFY(( internal::is_same_type<const float,internal::unref<const float&>::type >::ret));
  VERIFY(( internal::is_same_type<float,internal::unpointer<float*>::type >::ret));
  VERIFY(( internal::is_same_type<const float,internal::unpointer<const float*>::type >::ret));
  VERIFY(( internal::is_same_type<float,internal::unpointer<float* const >::type >::ret));
  
  VERIFY(internal::meta_sqrt<1>::ret == 1);
  #define VERIFY_META_SQRT(X) VERIFY(internal::meta_sqrt<X>::ret == int(internal::sqrt(double(X))))
  VERIFY_META_SQRT(2);
  VERIFY_META_SQRT(3);
  VERIFY_META_SQRT(4);
  VERIFY_META_SQRT(5);
  VERIFY_META_SQRT(6);
  VERIFY_META_SQRT(8);
  VERIFY_META_SQRT(9);
  VERIFY_META_SQRT(15);
  VERIFY_META_SQRT(16);
  VERIFY_META_SQRT(17);
  VERIFY_META_SQRT(255);
  VERIFY_META_SQRT(256);
  VERIFY_META_SQRT(257);
  VERIFY_META_SQRT(1023);
  VERIFY_META_SQRT(1024);
  VERIFY_META_SQRT(1025);
}
