// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

static void test_simple_patch()
{
  Tensor<float, 4> tensor(2,3,5,7);
  tensor.setRandom();
  array<ptrdiff_t, 4> patch_dims;
  patch_dims[0] = 1;
  patch_dims[1] = 1;
  patch_dims[2] = 1;
  patch_dims[3] = 1;

  Tensor<float, 5> no_patch;
  no_patch = tensor.extract_patches(patch_dims);

  VERIFY_IS_EQUAL(no_patch.dimension(0), 1);
  VERIFY_IS_EQUAL(no_patch.dimension(1), 1);
  VERIFY_IS_EQUAL(no_patch.dimension(2), 1);
  VERIFY_IS_EQUAL(no_patch.dimension(3), 1);
  VERIFY_IS_EQUAL(no_patch.dimension(4), tensor.size());

  for (int i = 0; i < tensor.size(); ++i) {
    VERIFY_IS_EQUAL(tensor.data()[i], no_patch.data()[i]);
  }

  patch_dims[0] = 1;
  patch_dims[1] = 2;
  patch_dims[2] = 2;
  patch_dims[3] = 1;
  Tensor<float, 5> twod_patch;
  twod_patch = tensor.extract_patches(patch_dims);

  VERIFY_IS_EQUAL(twod_patch.dimension(0), 1);
  VERIFY_IS_EQUAL(twod_patch.dimension(1), 2);
  VERIFY_IS_EQUAL(twod_patch.dimension(2), 2);
  VERIFY_IS_EQUAL(twod_patch.dimension(3), 1);
  VERIFY_IS_EQUAL(twod_patch.dimension(4), 2*2*4*7);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 7; ++l) {
          int patch_loc = i + 2 * (j + 2 * (k + 4 * l));
          for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
              VERIFY_IS_EQUAL(tensor(i,j+x,k+y,l), twod_patch(0,x,y,0,patch_loc));
            }
          }
        }
      }
    }
  }

  patch_dims[0] = 1;
  patch_dims[1] = 2;
  patch_dims[2] = 3;
  patch_dims[3] = 5;
  Tensor<float, 5> threed_patch;
  threed_patch = tensor.extract_patches(patch_dims);

  VERIFY_IS_EQUAL(threed_patch.dimension(0), 1);
  VERIFY_IS_EQUAL(threed_patch.dimension(1), 2);
  VERIFY_IS_EQUAL(threed_patch.dimension(2), 3);
  VERIFY_IS_EQUAL(threed_patch.dimension(3), 5);
  VERIFY_IS_EQUAL(threed_patch.dimension(4), 2*2*3*3);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          int patch_loc = i + 2 * (j + 2 * (k + 3 * l));
          for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 3; ++y) {
              for (int z = 0; z < 5; ++z) {
                VERIFY_IS_EQUAL(tensor(i,j+x,k+y,l+z), threed_patch(0,x,y,z,patch_loc));
              }
            }
          }
        }
      }
    }
  }
}


void test_cxx11_tensor_patch()
{
   CALL_SUBTEST(test_simple_patch());
   //   CALL_SUBTEST(test_expr_shuffling());
}
