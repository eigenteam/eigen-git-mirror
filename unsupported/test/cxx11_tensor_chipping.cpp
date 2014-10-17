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


static void test_simple_chip()
{
  Tensor<float, 5> tensor(2,3,5,7,11);
  tensor.setRandom();

  Tensor<float, 4> chip1;
  chip1 = tensor.chip<0>(1);
  VERIFY_IS_EQUAL(chip1.dimension(0), 3);
  VERIFY_IS_EQUAL(chip1.dimension(1), 5);
  VERIFY_IS_EQUAL(chip1.dimension(2), 7);
  VERIFY_IS_EQUAL(chip1.dimension(3), 11);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(chip1(i,j,k,l), tensor(1,i,j,k,l));
        }
      }
    }
  }

  Tensor<float, 4> chip2 = tensor.chip<1>(1);
  VERIFY_IS_EQUAL(chip2.dimension(0), 2);
  VERIFY_IS_EQUAL(chip2.dimension(1), 5);
  VERIFY_IS_EQUAL(chip2.dimension(2), 7);
  VERIFY_IS_EQUAL(chip2.dimension(3), 11);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(chip2(i,j,k,l), tensor(i,1,j,k,l));
        }
      }
    }
  }

  Tensor<float, 4> chip3 = tensor.chip<2>(2);
  VERIFY_IS_EQUAL(chip3.dimension(0), 2);
  VERIFY_IS_EQUAL(chip3.dimension(1), 3);
  VERIFY_IS_EQUAL(chip3.dimension(2), 7);
  VERIFY_IS_EQUAL(chip3.dimension(3), 11);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          VERIFY_IS_EQUAL(chip3(i,j,k,l), tensor(i,j,2,k,l));
        }
      }
    }
  }

  Tensor<float, 4> chip4(tensor.chip<3>(5));
  VERIFY_IS_EQUAL(chip4.dimension(0), 2);
  VERIFY_IS_EQUAL(chip4.dimension(1), 3);
  VERIFY_IS_EQUAL(chip4.dimension(2), 5);
  VERIFY_IS_EQUAL(chip4.dimension(3), 11);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(chip4(i,j,k,l), tensor(i,j,k,5,l));
        }
      }
    }
  }

  Tensor<float, 4> chip5(tensor.chip<4>(7));
  VERIFY_IS_EQUAL(chip5.dimension(0), 2);
  VERIFY_IS_EQUAL(chip5.dimension(1), 3);
  VERIFY_IS_EQUAL(chip5.dimension(2), 5);
  VERIFY_IS_EQUAL(chip5.dimension(3), 7);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(chip5(i,j,k,l), tensor(i,j,k,l,7));
        }
      }
    }
  }
}


static void test_chip_in_expr() {
  Tensor<float, 5> input1(2,3,5,7,11);
  input1.setRandom();
  Tensor<float, 4> input2(3,5,7,11);
  input2.setRandom();

  Tensor<float, 4> result = input1.chip<0>(0) + input2;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        for (int l = 0; l < 11; ++l) {
          float expected = input1(0,i,j,k,l) + input2(i,j,k,l);
          VERIFY_IS_EQUAL(result(i,j,k,l), expected);
        }
      }
    }
  }

  Tensor<float, 3> input3(3,7,11);
  input3.setRandom();
  Tensor<float, 3> result2 = input1.chip<0>(0).chip<1>(2) + input3;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 11; ++k) {
        float expected = input1(0,i,2,j,k) + input3(i,j,k);
        VERIFY_IS_EQUAL(result2(i,j,k), expected);
      }
    }
  }
}


static void test_chip_as_lvalue()
{
  Tensor<float, 5> input1(2,3,5,7,11);
  input1.setRandom();

  Tensor<float, 4> input2(3,5,7,11);
  input2.setRandom();
  Tensor<float, 5> tensor = input1;
  tensor.chip<0>(1) = input2;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          for (int m = 0; m < 11; ++m) {
            if (i != 1) {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input1(i,j,k,l,m));
            } else {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input2(j,k,l,m));
            }
          }
        }
      }
    }
  }

  Tensor<float, 4> input3(2,5,7,11);
  input3.setRandom();
  tensor = input1;
  tensor.chip<1>(1) = input3;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          for (int m = 0; m < 11; ++m) {
            if (j != 1) {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input1(i,j,k,l,m));
            } else {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input3(i,k,l,m));
            }
          }
        }
      }
    }
  }

  Tensor<float, 4> input4(2,3,7,11);
  input4.setRandom();
  tensor = input1;
  tensor.chip<2>(3) = input4;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          for (int m = 0; m < 11; ++m) {
            if (k != 3) {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input1(i,j,k,l,m));
            } else {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input4(i,j,l,m));
            }
          }
        }
      }
    }
  }

  Tensor<float, 4> input5(2,3,5,11);
  input5.setRandom();
  tensor = input1;
  tensor.chip<3>(4) = input5;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          for (int m = 0; m < 11; ++m) {
            if (l != 4) {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input1(i,j,k,l,m));
            } else {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input5(i,j,k,m));
            }
          }
        }
      }
    }
  }

  Tensor<float, 4> input6(2,3,5,7);
  input6.setRandom();
  tensor = input1;
  tensor.chip<4>(5) = input6;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          for (int m = 0; m < 11; ++m) {
            if (m != 5) {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input1(i,j,k,l,m));
            } else {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input6(i,j,k,l));
            }
          }
        }
      }
    }
  }
}


static void test_chip_raw_data()
{
  Tensor<float, 5> tensor(2,3,5,7,11);
  tensor.setRandom();

  typedef TensorEvaluator<decltype(tensor.chip<4>(3)), DefaultDevice> Evaluator4;
  auto chip = Evaluator4(tensor.chip<4>(3), DefaultDevice());
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          int chip_index = i + 2 * (j + 3 * (k + 5 * l));
          VERIFY_IS_EQUAL(chip.data()[chip_index], tensor(i,j,k,l,3));
        }
      }
    }
  }

  typedef TensorEvaluator<decltype(tensor.chip<0>(0)), DefaultDevice> Evaluator0;
  auto chip0 = Evaluator0(tensor.chip<0>(0), DefaultDevice());
  VERIFY_IS_EQUAL(chip0.data(), static_cast<float*>(0));

  typedef TensorEvaluator<decltype(tensor.chip<1>(0)), DefaultDevice> Evaluator1;
  auto chip1 = Evaluator1(tensor.chip<1>(0), DefaultDevice());
  VERIFY_IS_EQUAL(chip1.data(), static_cast<float*>(0));

  typedef TensorEvaluator<decltype(tensor.chip<2>(0)), DefaultDevice> Evaluator2;
  auto chip2 = Evaluator2(tensor.chip<2>(0), DefaultDevice());
  VERIFY_IS_EQUAL(chip2.data(), static_cast<float*>(0));

  typedef TensorEvaluator<decltype(tensor.chip<3>(0)), DefaultDevice> Evaluator3;
  auto chip3 = Evaluator3(tensor.chip<3>(0), DefaultDevice());
  VERIFY_IS_EQUAL(chip3.data(), static_cast<float*>(0));
}


void test_cxx11_tensor_chipping()
{
  CALL_SUBTEST(test_simple_chip());
  CALL_SUBTEST(test_chip_in_expr());
  CALL_SUBTEST(test_chip_as_lvalue());
  CALL_SUBTEST(test_chip_raw_data());
}
