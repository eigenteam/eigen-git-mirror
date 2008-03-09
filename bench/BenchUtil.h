
#include <Eigen/Core>
#include "BenchTimer.h"

using namespace std;
USING_PART_OF_NAMESPACE_EIGEN

#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition.hpp>
#include <boost/preprocessor/seq.hpp>
#include <boost/preprocessor/array.hpp>
#include <boost/preprocessor/arithmetic.hpp>
#include <boost/preprocessor/comparison.hpp>
#include <boost/preprocessor/punctuation.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/stringize.hpp>

template<typename MatrixType> void initMatrix_random(MatrixType& mat) __attribute__((noinline));
template<typename MatrixType> void initMatrix_random(MatrixType& mat)
{
  mat.setRandom();// = MatrixType::random(mat.rows(), mat.cols());
}

template<typename MatrixType> void initMatrix_identity(MatrixType& mat) __attribute__((noinline));
template<typename MatrixType> void initMatrix_identity(MatrixType& mat)
{
  mat.setIdentity();
}
