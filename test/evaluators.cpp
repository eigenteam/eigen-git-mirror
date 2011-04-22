#define EIGEN_ENABLE_EVALUATORS
#include "main.h"

using internal::copy_using_evaluator;
using namespace std;

#define VERIFY_IS_APPROX_EVALUATOR(DEST,EXPR) VERIFY_IS_APPROX(copy_using_evaluator(DEST,(EXPR)), (EXPR).eval());
#define VERIFY_IS_APPROX_EVALUATOR2(DEST,EXPR,REF) VERIFY_IS_APPROX(copy_using_evaluator(DEST,(EXPR)), (REF).eval());

void test_evaluators()
{
  // Testing Matrix evaluator and Transpose
  Vector2d v = Vector2d::Random();
  const Vector2d v_const(v);
  Vector2d v2;
  RowVector2d w;

  VERIFY_IS_APPROX_EVALUATOR(v2, v);
  VERIFY_IS_APPROX_EVALUATOR(v2, v_const);

  // Testing Transpose
  VERIFY_IS_APPROX_EVALUATOR(w, v.transpose()); // Transpose as rvalue
  VERIFY_IS_APPROX_EVALUATOR(w, v_const.transpose());

  copy_using_evaluator(w.transpose(), v); // Transpose as lvalue
  VERIFY_IS_APPROX(w,v.transpose().eval());

  copy_using_evaluator(w.transpose(), v_const);
  VERIFY_IS_APPROX(w,v_const.transpose().eval());

  // Testing Array evaluator
  ArrayXXf a(2,3);
  ArrayXXf b(3,2);
  a << 1,2,3, 4,5,6;
  const ArrayXXf a_const(a);

  VERIFY_IS_APPROX_EVALUATOR(b, a.transpose());

  VERIFY_IS_APPROX_EVALUATOR(b, a_const.transpose());

  // Testing CwiseNullaryOp evaluator
  copy_using_evaluator(w, RowVector2d::Random());
  VERIFY((w.array() >= -1).all() && (w.array() <= 1).all()); // not easy to test ...

  VERIFY_IS_APPROX_EVALUATOR(w, RowVector2d::Zero());

  VERIFY_IS_APPROX_EVALUATOR(w, RowVector2d::Constant(3));
  
  // mix CwiseNullaryOp and transpose
  VERIFY_IS_APPROX_EVALUATOR(w, Vector2d::Zero().transpose());

  {
    int s = internal::random<int>(1,100);
    MatrixXf a(s,s), b(s,s), c(s,s), d(s,s);
    a.setRandom();
    b.setRandom();
    c.setRandom();
    d.setRandom();
    VERIFY_IS_APPROX_EVALUATOR(d, (a + b));
    VERIFY_IS_APPROX_EVALUATOR(d, (a + b).transpose());
    VERIFY_IS_APPROX_EVALUATOR2(d, prod(a,b).transpose(), (a*b).transpose());
    VERIFY_IS_APPROX_EVALUATOR2(d, prod(a,b) + prod(b,c), a*b + b*c);
    
//     copy_using_evaluator(d, a.transpose() + (a.transpose() * (b+b)));
//     cout << d << endl;
  }
  
  // this does not work because Random is eval-before-nested: 
  // copy_using_evaluator(w, Vector2d::Random().transpose());
  
  // test CwiseUnaryOp
  VERIFY_IS_APPROX_EVALUATOR(v2, 3 * v);
  VERIFY_IS_APPROX_EVALUATOR(w, (3 * v).transpose());
  VERIFY_IS_APPROX_EVALUATOR(b, (a + 3).transpose());
  VERIFY_IS_APPROX_EVALUATOR(b, (2 * a_const + 3).transpose());

  // test CwiseBinaryOp
  VERIFY_IS_APPROX_EVALUATOR(v2, v + Vector2d::Ones());
  VERIFY_IS_APPROX_EVALUATOR(w, (v + Vector2d::Ones()).transpose().cwiseProduct(RowVector2d::Constant(3)));

  // dynamic matrices and arrays
  MatrixXd mat1(6,6), mat2(6,6);
  VERIFY_IS_APPROX_EVALUATOR(mat1, MatrixXd::Identity(6,6));
  VERIFY_IS_APPROX_EVALUATOR(mat2, mat1);
  copy_using_evaluator(mat2.transpose(), mat1);
  VERIFY_IS_APPROX(mat2.transpose(), mat1);

  ArrayXXd arr1(6,6), arr2(6,6);
  VERIFY_IS_APPROX_EVALUATOR(arr1, ArrayXXd::Constant(6,6, 3.0));
  VERIFY_IS_APPROX_EVALUATOR(arr2, arr1);

  // test direct traversal
  Matrix3f m3;
  Array33f a3;
  VERIFY_IS_APPROX_EVALUATOR(m3, Matrix3f::Identity());  // matrix, nullary
  // TODO: find a way to test direct traversal with array
  VERIFY_IS_APPROX_EVALUATOR(m3.transpose(), Matrix3f::Identity().transpose());  // transpose
  VERIFY_IS_APPROX_EVALUATOR(m3, 2 * Matrix3f::Identity());  // unary
  VERIFY_IS_APPROX_EVALUATOR(m3, Matrix3f::Identity() + m3);  // binary
  VERIFY_IS_APPROX_EVALUATOR(m3.block(0,0,2,2), Matrix3f::Identity().block(1,1,2,2));  // block

  // test linear traversal
  VERIFY_IS_APPROX_EVALUATOR(m3, Matrix3f::Zero());  // matrix, nullary
  VERIFY_IS_APPROX_EVALUATOR(a3, Array33f::Zero());  // array
  VERIFY_IS_APPROX_EVALUATOR(m3.transpose(), Matrix3f::Zero().transpose());  // transpose
  VERIFY_IS_APPROX_EVALUATOR(m3, 2 * Matrix3f::Zero());  // unary
  VERIFY_IS_APPROX_EVALUATOR(m3, Matrix3f::Zero() + m3);  // binary  

  // test inner vectorization
  Matrix4f m4, m4src = Matrix4f::Random();
  Array44f a4, a4src = Matrix4f::Random();
  VERIFY_IS_APPROX_EVALUATOR(m4, m4src);  // matrix
  VERIFY_IS_APPROX_EVALUATOR(a4, a4src);  // array
  VERIFY_IS_APPROX_EVALUATOR(m4.transpose(), m4src.transpose());  // transpose
  // TODO: find out why Matrix4f::Zero() does not allow inner vectorization
  VERIFY_IS_APPROX_EVALUATOR(m4, 2 * m4src);  // unary
  VERIFY_IS_APPROX_EVALUATOR(m4, m4src + m4src);  // binary

  // test linear vectorization
  MatrixXf mX(6,6), mXsrc = MatrixXf::Random(6,6);
  ArrayXXf aX(6,6), aXsrc = ArrayXXf::Random(6,6);
  VERIFY_IS_APPROX_EVALUATOR(mX, mXsrc);  // matrix
  VERIFY_IS_APPROX_EVALUATOR(aX, aXsrc);  // array
  VERIFY_IS_APPROX_EVALUATOR(mX.transpose(), mXsrc.transpose());  // transpose
  VERIFY_IS_APPROX_EVALUATOR(mX, MatrixXf::Zero(6,6));  // nullary
  VERIFY_IS_APPROX_EVALUATOR(mX, 2 * mXsrc);  // unary
  VERIFY_IS_APPROX_EVALUATOR(mX, mXsrc + mXsrc);  // binary

  // test blocks and slice vectorization
  VERIFY_IS_APPROX_EVALUATOR(m4, (mXsrc.block<4,4>(1,0)));
  VERIFY_IS_APPROX_EVALUATOR(aX, ArrayXXf::Constant(10, 10, 3.0).block(2, 3, 6, 6));

  Matrix4f m4ref = m4;
  copy_using_evaluator(m4.block(1, 1, 2, 3), m3.bottomRows(2));
  m4ref.block(1, 1, 2, 3) = m3.bottomRows(2);
  VERIFY_IS_APPROX(m4, m4ref);

  mX.setIdentity(20,20);
  MatrixXf mXref = MatrixXf::Identity(20,20);
  mXsrc = MatrixXf::Random(9,12);
  copy_using_evaluator(mX.block(4, 4, 9, 12), mXsrc);
  mXref.block(4, 4, 9, 12) = mXsrc;
  VERIFY_IS_APPROX(mX, mXref);

  // test Map
  const float raw[3] = {1,2,3};
  float buffer[3] = {0,0,0};
  Vector3f v3;
  Array3f a3f;
  VERIFY_IS_APPROX_EVALUATOR(v3, Map<const Vector3f>(raw));
  VERIFY_IS_APPROX_EVALUATOR(a3f, Map<const Array3f>(raw));
  Vector3f::Map(buffer) = 2*v3;
  VERIFY(buffer[0] == 2);
  VERIFY(buffer[1] == 4);
  VERIFY(buffer[2] == 6);

  // test CwiseUnaryView
  mat1.setRandom();
  mat2.setIdentity();
  MatrixXcd matXcd(6,6), matXcd_ref(6,6);
  copy_using_evaluator(matXcd.real(), mat1);
  copy_using_evaluator(matXcd.imag(), mat2);
  matXcd_ref.real() = mat1;
  matXcd_ref.imag() = mat2;
  VERIFY_IS_APPROX(matXcd, matXcd_ref);

  // test Select
  VERIFY_IS_APPROX_EVALUATOR(aX, (aXsrc > 0).select(aXsrc, -aXsrc));

  // test Replicate
  mXsrc = MatrixXf::Random(6, 6);
  VectorXf vX = VectorXf::Random(6);
  mX.resize(6, 6);
  VERIFY_IS_APPROX_EVALUATOR(mX, mXsrc.colwise() + vX);
  matXcd.resize(12, 12);
  VERIFY_IS_APPROX_EVALUATOR(matXcd, matXcd_ref.replicate(2,2));
  VERIFY_IS_APPROX_EVALUATOR(matXcd, (matXcd_ref.replicate<2,2>()));

  // test partial reductions
  VectorXd vec1(6);
  VERIFY_IS_APPROX_EVALUATOR(vec1, mat1.rowwise().sum());
  VERIFY_IS_APPROX_EVALUATOR(vec1, mat1.colwise().sum().transpose());

  // test MatrixWrapper and ArrayWrapper
  mat1.setRandom(6,6);
  arr1.setRandom(6,6);
  VERIFY_IS_APPROX_EVALUATOR(mat2, arr1.matrix());
  VERIFY_IS_APPROX_EVALUATOR(arr2, mat1.array());
  VERIFY_IS_APPROX_EVALUATOR(mat2, (arr1 + 2).matrix());
  VERIFY_IS_APPROX_EVALUATOR(arr2, mat1.array() + 2);
  mat2.array() = arr1 * arr1;
  VERIFY_IS_APPROX(mat2, (arr1 * arr1).matrix());
  arr2.matrix() = MatrixXd::Identity(6,6);
  VERIFY_IS_APPROX(arr2, MatrixXd::Identity(6,6).array());
}
