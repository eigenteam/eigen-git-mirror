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
    if (tensor.data()[i] != single_pixel_patch.data()[i]) {
      std::cout << "Mismatch detected at index " << i << " : " << tensor.data()[i] << " vs " << single_pixel_patch.data()[i] << std::endl;
    }
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
              if (entire_image_patch(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
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

  // Based on the calculation described in TensorTraits.h, padding happens to be 0.
  int row_padding = 0;
  int col_padding = 0;
  int stride = 1;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      int patchId = i+3*j;
      for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
          for (int d = 0; d < 2; ++d) {
            for (int b = 0; b < 7; ++b) {
              float expected = 0.0f;
              int row_offset = r*stride + i - row_padding;
              int col_offset = c*stride + j - col_padding;
              if (row_offset >= 0 && col_offset >= 0 && row_offset < tensor.dimension(1) && col_offset < tensor.dimension(2)) {
                expected = tensor(d, row_offset, col_offset, b);
              }
              if (twod_patch(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(twod_patch(d, r, c, patchId, b), expected);
            }
          }
        }
      }
    }
  }
}

// Verifies VALID padding (no padding) with incrementing values.
static void test_patch_padding_valid()
{
  int input_depth = 3;
  int input_rows = 3;
  int input_cols = 3;
  int input_batches = 1;
  int ksize = 2;  // Corresponds to the Rows and Cols for tensor.extract_image_patches<>.
  int stride = 2;  // Only same stride is supported.
  Tensor<float, 4> tensor(input_depth, input_rows, input_cols, input_batches);
  // Initializes tensor with incrementing numbers.
  for (int i = 0; i < tensor.size(); ++i) {
    tensor.data()[i] = i + 1;
  }
  Tensor<float, 5> result = tensor.extract_image_patches(ksize, ksize, stride, stride, PADDING_VALID);

  VERIFY_IS_EQUAL(result.dimension(0), input_depth);  // depth
  VERIFY_IS_EQUAL(result.dimension(1), ksize);  // kernel rows
  VERIFY_IS_EQUAL(result.dimension(2), ksize);  // kernel cols
  VERIFY_IS_EQUAL(result.dimension(3), 1);  // number of patches
  VERIFY_IS_EQUAL(result.dimension(4), input_batches);  // number of batches

  // No padding is carried out.
  int row_padding = 0;
  int col_padding = 0;

  for (int i = 0; (i+stride+ksize-1) < input_rows; i += stride) {  // input rows
    for (int j = 0; (j+stride+ksize-1) < input_cols; j += stride) {  // input cols
      int patchId = i+input_rows*j;
      for (int r = 0; r < ksize; ++r) {  // patch rows
        for (int c = 0; c < ksize; ++c) {  // patch cols
          for (int d = 0; d < input_depth; ++d) {  // depth
            for (int b = 0; b < input_batches; ++b) {  // batch
              float expected = 0.0f;
              int row_offset = r + i - row_padding;
              int col_offset = c + j - col_padding;
              if (row_offset >= 0 && col_offset >= 0 && row_offset < input_rows && col_offset < input_cols) {
                expected = tensor(d, row_offset, col_offset, b);
              }
              if (result(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(result(d, r, c, patchId, b), expected);
            }
          }
        }
      }
    }
  }
}

// Verifies VALID padding (no padding) with the same value.
static void test_patch_padding_valid_same_value()
{
  int input_depth = 1;
  int input_rows = 5;
  int input_cols = 5;
  int input_batches = 2;
  int ksize = 3;  // Corresponds to the Rows and Cols for tensor.extract_image_patches<>.
  int stride = 2;  // Only same stride is supported.
  Tensor<float, 4> tensor(input_depth, input_rows, input_cols, input_batches);
  tensor = tensor.constant(11.0f);
  Tensor<float, 5> result = tensor.extract_image_patches(ksize, ksize, stride, stride, PADDING_VALID);

  VERIFY_IS_EQUAL(result.dimension(0), input_depth);  // depth
  VERIFY_IS_EQUAL(result.dimension(1), ksize);  // kernel rows
  VERIFY_IS_EQUAL(result.dimension(2), ksize);  // kernel cols
  VERIFY_IS_EQUAL(result.dimension(3), 4);  // number of patches
  VERIFY_IS_EQUAL(result.dimension(4), input_batches);  // number of batches

  // No padding is carried out.
  int row_padding = 0;
  int col_padding = 0;

  for (int i = 0; (i+stride+ksize-1) <= input_rows; i += stride) {  // input rows
    for (int j = 0; (j+stride+ksize-1) <= input_cols; j += stride) {  // input cols
      int patchId = i+input_rows*j;
      for (int r = 0; r < ksize; ++r) {  // patch rows
        for (int c = 0; c < ksize; ++c) {  // patch cols
          for (int d = 0; d < input_depth; ++d) {  // depth
            for (int b = 0; b < input_batches; ++b) {  // batch
              float expected = 0.0f;
              int row_offset = r + i - row_padding;
              int col_offset = c + j - col_padding;
              if (row_offset >= 0 && col_offset >= 0 && row_offset < input_rows && col_offset < input_cols) {
                expected = tensor(d, row_offset, col_offset, b);
              }
              if (result(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(result(d, r, c, patchId, b), expected);
            }
          }
        }
      }
    }
  }
}

// Verifies SAME padding.
static void test_patch_padding_same()
{
  int input_depth = 3;
  int input_rows = 4;
  int input_cols = 2;
  int input_batches = 1;
  int ksize = 2;  // Corresponds to the Rows and Cols for tensor.extract_image_patches<>.
  int stride = 2;  // Only same stride is supported.
  Tensor<float, 4> tensor(input_depth, input_rows, input_cols, input_batches);
  // Initializes tensor with incrementing numbers.
  for (int i = 0; i < tensor.size(); ++i) {
    tensor.data()[i] = i + 1;
  }
  Tensor<float, 5> result = tensor.extract_image_patches(ksize, ksize, stride, stride, PADDING_SAME);

  VERIFY_IS_EQUAL(result.dimension(0), input_depth);  // depth
  VERIFY_IS_EQUAL(result.dimension(1), ksize);  // kernel rows
  VERIFY_IS_EQUAL(result.dimension(2), ksize);  // kernel cols
  VERIFY_IS_EQUAL(result.dimension(3), 2);  // number of patches
  VERIFY_IS_EQUAL(result.dimension(4), input_batches);  // number of batches

  // Based on the calculation described in TensorTraits.h, padding happens to be
  // 0.
  int row_padding = 0;
  int col_padding = 0;

  for (int i = 0; (i+stride+ksize-1) <= input_rows; i += stride) {  // input rows
    for (int j = 0; (j+stride+ksize-1) <= input_cols; j += stride) {  // input cols
      int patchId = i+input_rows*j;
      for (int r = 0; r < ksize; ++r) {  // patch rows
        for (int c = 0; c < ksize; ++c) {  // patch cols
          for (int d = 0; d < input_depth; ++d) {  // depth
            for (int b = 0; b < input_batches; ++b) {  // batch
              float expected = 0.0f;
              int row_offset = r*stride + i - row_padding;
              int col_offset = c*stride + j - col_padding;
              if (row_offset >= 0 && col_offset >= 0 && row_offset < input_rows && col_offset < input_cols) {
                expected = tensor(d, row_offset, col_offset, b);
              }
              if (result(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(result(d, r, c, patchId, b), expected);
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
    if (tensor.data()[i] != single_pixel_patch.data()[i]) {
      std::cout << "Mismatch detected at index " << i << " : " << tensor.data()[i] << " vs " << single_pixel_patch.data()[i] << std::endl;
    }
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
            if (entire_image_patch(d, r, c, patchId) != expected) {
              std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << std::endl;
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

  // Based on the calculation described in TensorTraits.h, padding happens to be 0.
  int row_padding = 0;
  int col_padding = 0;
  int stride = 1;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      int patchId = i+3*j;
      for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
          for (int d = 0; d < 2; ++d) {
            float expected = 0.0f;
            int row_offset = r*stride + i - row_padding;
            int col_offset = c*stride + j - col_padding;
            if (row_offset >= 0 && col_offset >= 0 && row_offset < tensor.dimension(1) && col_offset < tensor.dimension(2)) {
              expected = tensor(d, row_offset, col_offset);
            }
            if (twod_patch(d, r, c, patchId) != expected) {
              std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << std::endl;
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
              if (l_out(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
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
              if (l_out(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
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
              if (l_out(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
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
              if (l_out(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
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
  CALL_SUBTEST(test_patch_padding_valid());
  CALL_SUBTEST(test_patch_padding_valid_same_value());
  CALL_SUBTEST(test_patch_padding_same());
  CALL_SUBTEST(test_imagenet_patches());
}
