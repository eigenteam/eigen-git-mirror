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

#ifndef EIGEN_STATIC_ASSERT_H
#define EIGEN_STATIC_ASSERT_H

/* Some notes on Eigen's static assertion mechanism:
 *
 *  - in EIGEN_STATIC_ASSERT(CONDITION,MSG) the parameter CONDITION must be a compile time boolean
 *    expression, and MSG an enum listed in struct ei_static_assert<true>
 *
 *  - define EIGEN_NO_STATIC_ASSERT to disable them (and save compilation time)
 *    in that case, the static assertion is converted to the following runtime assert:
 *      ei_assert(CONDITION && "MSG")
 *
 *  - currently EIGEN_STATIC_ASSERT can only be used in function scope
 *
 */

#ifndef EIGEN_NO_STATIC_ASSERT

  #ifdef __GXX_EXPERIMENTAL_CXX0X__

    // if native static_assert is enabled, let's use it
    #define EIGEN_STATIC_ASSERT(X,MSG) static_assert(X,#MSG)

  #else // CXX0X

    template<bool condition>
    struct ei_static_assert {};

    template<>
    struct ei_static_assert<true>
    {
      enum {
        you_tried_calling_a_vector_method_on_a_matrix,
        you_mixed_vectors_of_different_sizes,
        you_mixed_matrices_of_different_sizes,
        this_method_is_only_for_vectors_of_a_specific_size,
        this_method_is_only_for_matrices_of_a_specific_size,
        you_did_a_programming_error,
        you_called_a_fixed_size_method_on_a_dynamic_size_matrix_or_vector,
        unaligned_load_and_store_operations_unimplemented_on_AltiVec,
        scalar_type_must_be_floating_point,
        default_writting_to_selfadjoint_not_supported,
        writting_to_triangular_part_with_unit_diag_is_not_supported,
        this_method_is_only_for_fixed_size
      };
    };

    #define EIGEN_STATIC_ASSERT(CONDITION,MSG) \
      if (ei_static_assert<CONDITION ? true : false>::MSG) {}

  #endif // CXX0X

#else // EIGEN_NO_STATIC_ASSERT

  #define EIGEN_STATIC_ASSERT(CONDITION,MSG) ei_assert((CONDITION) && #MSG)

#endif // EIGEN_NO_STATIC_ASSERT


// static assertion failing if the type \a TYPE is not a vector type
#define EIGEN_STATIC_ASSERT_VECTOR_ONLY(TYPE) \
  EIGEN_STATIC_ASSERT(TYPE::IsVectorAtCompileTime, \
                      you_tried_calling_a_vector_method_on_a_matrix)

// static assertion failing if the type \a TYPE is not fixed-size
#define EIGEN_STATIC_ASSERT_FIXED_SIZE(TYPE) \
  EIGEN_STATIC_ASSERT(TYPE::SizeAtCompileTime!=Eigen::Dynamic, \
                      you_called_a_fixed_size_method_on_a_dynamic_size_matrix_or_vector)

// static assertion failing if the type \a TYPE is not a vector type of the given size
#define EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(TYPE, SIZE) \
  EIGEN_STATIC_ASSERT(TYPE::IsVectorAtCompileTime && TYPE::SizeAtCompileTime==SIZE, \
                      this_method_is_only_for_vectors_of_a_specific_size)

// static assertion failing if the type \a TYPE is not a vector type of the given size
#define EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(TYPE, ROWS, COLS) \
  EIGEN_STATIC_ASSERT(TYPE::RowsAtCompileTime==ROWS && TYPE::ColsAtCompileTime==COLS, \
                      this_method_is_only_for_matrices_of_a_specific_size)

// static assertion failing if the two vector expression types are not compatible (same fixed-size or dynamic size)
#define EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(TYPE0,TYPE1) \
  EIGEN_STATIC_ASSERT( \
      (int(TYPE0::SizeAtCompileTime)==Eigen::Dynamic \
    || int(TYPE1::SizeAtCompileTime)==Eigen::Dynamic \
    || int(TYPE0::SizeAtCompileTime)==int(TYPE1::SizeAtCompileTime)),\
    you_mixed_vectors_of_different_sizes)

// static assertion failing if the two matrix expression types are not compatible (same fixed-size or dynamic size)
#define EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(TYPE0,TYPE1) \
  EIGEN_STATIC_ASSERT( \
     ((int(TYPE0::RowsAtCompileTime)==Eigen::Dynamic \
    || int(TYPE1::RowsAtCompileTime)==Eigen::Dynamic \
    || int(TYPE0::RowsAtCompileTime)==int(TYPE1::RowsAtCompileTime)) \
   && (int(TYPE0::ColsAtCompileTime)==Eigen::Dynamic \
    || int(TYPE1::ColsAtCompileTime)==Eigen::Dynamic \
    || int(TYPE0::ColsAtCompileTime)==int(TYPE1::ColsAtCompileTime))),\
    you_mixed_matrices_of_different_sizes)


#endif // EIGEN_STATIC_ASSERT_H
