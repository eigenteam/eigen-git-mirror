#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::RowMajor;

static void test_comparison_sugar() {
  // we already trust comparisons between tensors, we're simply checking that
  // the sugared versions are doing the same thing
  Tensor<int, 3> t(6, 7, 5);

  t.setRandom();
  // make sure we have at least one value == 0
  t(0,0,0) = 0;

  Tensor<bool,0> b;

#define TEST_TENSOR_EQUAL(e1, e2) \
  b = ((e1) == (e2)).all();       \
  VERIFY(b())

#define TEST_OP(op) TEST_TENSOR_EQUAL(t op 0, t op t.constant(0))

  TEST_OP(==);
  TEST_OP(!=);
  TEST_OP(<=);
  TEST_OP(>=);
  TEST_OP(<);
  TEST_OP(>);
#undef TEST_OP
#undef TEST_TENSOR_EQUAL
}


static void test_scalar_sugar() {
  Tensor<float, 3> A(6, 7, 5);
  Tensor<float, 3> B(6, 7, 5);
  A.setRandom();
  B.setRandom();

  const float alpha = 0.43f;
  const float beta = 0.21f;

  Tensor<float, 3> R = A * A.constant(alpha) + B * B.constant(beta);
  Tensor<float, 3> S = A * alpha + B * beta;

  // TODO: add enough syntactic sugar to support this
  // Tensor<float, 3> T = alpha * A + beta * B;

  for (int i = 0; i < 6*7*5; ++i) {
    VERIFY_IS_APPROX(R(i), S(i));
  }
}


void test_cxx11_tensor_sugar()
{
  CALL_SUBTEST(test_comparison_sugar());
  CALL_SUBTEST(test_scalar_sugar());
}
