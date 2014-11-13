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

  Tensor<float, 5> single_pixel_patch;
  single_pixel_patch = tensor.extract_image_patches<1, 1>();

  VERIFY_IS_EQUAL(single_pixel_patch.dimension(0), 2);
  VERIFY_IS_EQUAL(single_pixel_patch.dimension(1), 1);
  VERIFY_IS_EQUAL(single_pixel_patch.dimension(2), 1);
  VERIFY_IS_EQUAL(single_pixel_patch.dimension(3), 3*5);
  VERIFY_IS_EQUAL(single_pixel_patch.dimension(4), 7);

  for (int i = 0; i < tensor.size(); ++i) {
    VERIFY_IS_EQUAL(single_pixel_patch.data()[i], tensor.data()[i]);
  }

  Tensor<float, 5> entire_image_patch;
  entire_image_patch = tensor.extract_image_patches<3, 5>();

  VERIFY_IS_EQUAL(entire_image_patch.dimension(0), 2);
  VERIFY_IS_EQUAL(entire_image_patch.dimension(1), 3);
  VERIFY_IS_EQUAL(entire_image_patch.dimension(2), 5);
  VERIFY_IS_EQUAL(entire_image_patch.dimension(3), 3*5);
  VERIFY_IS_EQUAL(entire_image_patch.dimension(4), 7);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      int patchId = i+3*j;
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 5; ++c) {
          for (int d = 0; d < 2; ++d) {
            for (int b = 0; b < 7; ++b) {
              float expected = 0.0f;
              if (r-1+i >= 0 && c-2+j >= 0 && r-1+i < 3 && c-2+j < 5) {
                expected = tensor(d, r-1+i, c-2+j, b);
              }
              VERIFY_IS_EQUAL(entire_image_patch(d, r, c, patchId, b), expected);
            }
          }
        }
      }
    }
  }

  Tensor<float, 5> twod_patch;
  twod_patch = tensor.extract_image_patches<2, 2>();

  VERIFY_IS_EQUAL(twod_patch.dimension(0), 2);
  VERIFY_IS_EQUAL(twod_patch.dimension(1), 2);
  VERIFY_IS_EQUAL(twod_patch.dimension(2), 2);
  VERIFY_IS_EQUAL(twod_patch.dimension(3), 3*5);
  VERIFY_IS_EQUAL(twod_patch.dimension(4), 7);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      int patchId = i+3*j;
      for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
          for (int d = 0; d < 2; ++d) {
            for (int b = 0; b < 7; ++b) {
              float expected = 0.0f;
              if (r-1+i >= 0 && c-1+j >= 0 && r-1+i < 3 && c-1+j < 5) {
                expected = tensor(d, r-1+i, c-1+j, b);
              }
              VERIFY_IS_EQUAL(twod_patch(d, r, c, patchId, b), expected);
            }
          }
        }
      }
    }
  }
}


static void test_patch_no_extra_dim()
{
  Tensor<float, 3> tensor(2,3,5);
  tensor.setRandom();

  Tensor<float, 4> single_pixel_patch;
  single_pixel_patch = tensor.extract_image_patches<1, 1>();

  VERIFY_IS_EQUAL(single_pixel_patch.dimension(0), 2);
  VERIFY_IS_EQUAL(single_pixel_patch.dimension(1), 1);
  VERIFY_IS_EQUAL(single_pixel_patch.dimension(2), 1);
  VERIFY_IS_EQUAL(single_pixel_patch.dimension(3), 3*5);

  for (int i = 0; i < tensor.size(); ++i) {
    VERIFY_IS_EQUAL(single_pixel_patch.data()[i], tensor.data()[i]);
  }

  Tensor<float, 4> entire_image_patch;
  entire_image_patch = tensor.extract_image_patches<3, 5>();

  VERIFY_IS_EQUAL(entire_image_patch.dimension(0), 2);
  VERIFY_IS_EQUAL(entire_image_patch.dimension(1), 3);
  VERIFY_IS_EQUAL(entire_image_patch.dimension(2), 5);
  VERIFY_IS_EQUAL(entire_image_patch.dimension(3), 3*5);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      int patchId = i+3*j;
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 5; ++c) {
          for (int d = 0; d < 2; ++d) {
            float expected = 0.0f;
            if (r-1+i >= 0 && c-2+j >= 0 && r-1+i < 3 && c-2+j < 5) {
              expected = tensor(d, r-1+i, c-2+j);
            }
            VERIFY_IS_EQUAL(entire_image_patch(d, r, c, patchId), expected);
          }
        }
      }
    }
  }

  Tensor<float, 4> twod_patch;
  twod_patch = tensor.extract_image_patches<2, 2>();

  VERIFY_IS_EQUAL(twod_patch.dimension(0), 2);
  VERIFY_IS_EQUAL(twod_patch.dimension(1), 2);
  VERIFY_IS_EQUAL(twod_patch.dimension(2), 2);
  VERIFY_IS_EQUAL(twod_patch.dimension(3), 3*5);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      int patchId = i+3*j;
      for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
          for (int d = 0; d < 2; ++d) {
            float expected = 0.0f;
            if (r-1+i >= 0 && c-1+j >= 0 && r-1+i < 3 && c-1+j < 5) {
              expected = tensor(d, r-1+i, c-1+j);
            }
            VERIFY_IS_EQUAL(twod_patch(d, r, c, patchId), expected);
          }
        }
      }
    }
  }
}


static void test_imagenet_patches()
{
  // Test the code on typical configurations used by the 'imagenet' benchmarks at
  // https://github.com/soumith/convnet-benchmarks
  Tensor<float, 4> l_in(3, 128, 128, 128);
  l_in.setRandom();
  Tensor<float, 5> l_out = l_in.extract_image_patches(11, 11);
  VERIFY_IS_EQUAL(l_out.dimension(0), 3);
  VERIFY_IS_EQUAL(l_out.dimension(1), 11);
  VERIFY_IS_EQUAL(l_out.dimension(2), 11);
  VERIFY_IS_EQUAL(l_out.dimension(3), 128*128);
  VERIFY_IS_EQUAL(l_out.dimension(4), 128);
  for (int b = 0; b < 128; ++b) {
    for (int i = 0; i < 128; ++i) {
      for (int j = 0; j < 128; ++j) {
        int patchId = i+128*j;
        for (int c = 0; c < 11; ++c) {
          for (int r = 0; r < 11; ++r) {
            for (int d = 0; d < 3; ++d) {
              float expected = 0.0f;
              if (r-5+i >= 0 && c-5+j >= 0 && r-5+i < 128 && c-5+j < 128) {
                expected = l_in(d, r-5+i, c-5+j, b);
              }
              VERIFY_IS_EQUAL(l_out(d, r, c, patchId, b), expected);
            }
          }
        }
      }
    }
  }

  l_in.resize(64, 64, 64, 128);
  l_in.setRandom();
  l_out = l_in.extract_image_patches(9, 9);
  VERIFY_IS_EQUAL(l_out.dimension(0), 64);
  VERIFY_IS_EQUAL(l_out.dimension(1), 9);
  VERIFY_IS_EQUAL(l_out.dimension(2), 9);
  VERIFY_IS_EQUAL(l_out.dimension(3), 64*64);
  VERIFY_IS_EQUAL(l_out.dimension(4), 128);
  for (int b = 0; b < 128; ++b) {
    for (int i = 0; i < 64; ++i) {
      for (int j = 0; j < 64; ++j) {
        int patchId = i+64*j;
        for (int c = 0; c < 9; ++c) {
          for (int r = 0; r < 9; ++r) {
            for (int d = 0; d < 64; ++d) {
              float expected = 0.0f;
              if (r-4+i >= 0 && c-4+j >= 0 && r-4+i < 64 && c-4+j < 64) {
                expected = l_in(d, r-4+i, c-4+j, b);
              }
              VERIFY_IS_EQUAL(l_out(d, r, c, patchId, b), expected);
            }
          }
        }
      }
    }
  }

  l_in.resize(128, 16, 16, 128);
  l_in.setRandom();
  l_out = l_in.extract_image_patches(7, 7);
  VERIFY_IS_EQUAL(l_out.dimension(0), 128);
  VERIFY_IS_EQUAL(l_out.dimension(1), 7);
  VERIFY_IS_EQUAL(l_out.dimension(2), 7);
  VERIFY_IS_EQUAL(l_out.dimension(3), 16*16);
  VERIFY_IS_EQUAL(l_out.dimension(4), 128);
  for (int b = 0; b < 128; ++b) {
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        int patchId = i+16*j;
        for (int c = 0; c < 7; ++c) {
          for (int r = 0; r < 7; ++r) {
            for (int d = 0; d < 128; ++d) {
              float expected = 0.0f;
              if (r-3+i >= 0 && c-3+j >= 0 && r-3+i < 16 && c-3+j < 16) {
                expected = l_in(d, r-3+i, c-3+j, b);
              }
              VERIFY_IS_EQUAL(l_out(d, r, c, patchId, b), expected);
            }
          }
        }
      }
    }
  }

  l_in.resize(384, 13, 13, 128);
  l_in.setRandom();
  l_out = l_in.extract_image_patches(3, 3);
  VERIFY_IS_EQUAL(l_out.dimension(0), 384);
  VERIFY_IS_EQUAL(l_out.dimension(1), 3);
  VERIFY_IS_EQUAL(l_out.dimension(2), 3);
  VERIFY_IS_EQUAL(l_out.dimension(3), 13*13);
  VERIFY_IS_EQUAL(l_out.dimension(4), 128);
  for (int b = 0; b < 128; ++b) {
    for (int i = 0; i < 13; ++i) {
      for (int j = 0; j < 13; ++j) {
        int patchId = i+13*j;
        for (int c = 0; c < 3; ++c) {
          for (int r = 0; r < 3; ++r) {
            for (int d = 0; d < 384; ++d) {
              float expected = 0.0f;
              if (r-1+i >= 0 && c-1+j >= 0 && r-1+i < 13 && c-1+j < 13) {
                expected = l_in(d, r-1+i, c-1+j, b);
              }
              VERIFY_IS_EQUAL(l_out(d, r, c, patchId, b), expected);
            }
          }
        }
      }
    }
  }
}


void test_cxx11_tensor_image_patch()
{
  CALL_SUBTEST(test_simple_patch());
  CALL_SUBTEST(test_patch_no_extra_dim());
  CALL_SUBTEST(test_imagenet_patches());
}
