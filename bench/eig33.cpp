#include <iostream>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <bench/BenchTimer.h>

using namespace Eigen;
using namespace std;

template<typename Matrix, typename Roots>
inline void computeRoots (const Matrix& rkA, Roots& adRoot)
{
  typedef typename Matrix::Scalar Scalar;
  const Scalar msInv3 = 1.0/3.0;
  const Scalar msRoot3 = ei_sqrt(Scalar(3.0));

  Scalar dA00 = rkA(0,0);
  Scalar dA01 = rkA(0,1);
  Scalar dA02 = rkA(0,2);
  Scalar dA11 = rkA(1,1);
  Scalar dA12 = rkA(1,2);
  Scalar dA22 = rkA(2,2);

  // The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
  // eigenvalues are the roots to this equation, all guaranteed to be
  // real-valued, because the matrix is symmetric.
  Scalar dC0 = dA00*dA11*dA22 + Scalar(2)*dA01*dA02*dA12 - dA00*dA12*dA12 - dA11*dA02*dA02 - dA22*dA01*dA01;
  Scalar dC1 = dA00*dA11 - dA01*dA01 + dA00*dA22 - dA02*dA02 + dA11*dA22 - dA12*dA12;
  Scalar dC2 = dA00 + dA11 + dA22;

  // Construct the parameters used in classifying the roots of the equation
  // and in solving the equation for the roots in closed form.
  Scalar dC2Div3 = dC2*msInv3;
  Scalar dADiv3 = (dC1 - dC2*dC2Div3)*msInv3;
  if (dADiv3 > Scalar(0))
    dADiv3 = Scalar(0);

  Scalar dMBDiv2 = Scalar(0.5)*(dC0 + dC2Div3*(Scalar(2)*dC2Div3*dC2Div3 - dC1));

  Scalar dQ = dMBDiv2*dMBDiv2 + dADiv3*dADiv3*dADiv3;
  if (dQ > Scalar(0))
    dQ = Scalar(0);

  // Compute the eigenvalues by solving for the roots of the polynomial.
  Scalar dMagnitude = ei_sqrt(-dADiv3);
  Scalar dAngle = std::atan2(ei_sqrt(-dQ),dMBDiv2)*msInv3;
  Scalar dCos = ei_cos(dAngle);
  Scalar dSin = ei_sin(dAngle);
  adRoot(0) = dC2Div3 + 2.f*dMagnitude*dCos;
  adRoot(1) = dC2Div3 - dMagnitude*(dCos + msRoot3*dSin);
  adRoot(2) = dC2Div3 - dMagnitude*(dCos - msRoot3*dSin);

  // Sort in increasing order.
  if (adRoot(0) >= adRoot(1))
    std::swap(adRoot(0),adRoot(1));
  if (adRoot(1) >= adRoot(2))
  {
    std::swap(adRoot(1),adRoot(2));
    if (adRoot(0) >= adRoot(1))
      std::swap(adRoot(0),adRoot(1));
  }
}

template<typename Matrix, typename Vector>
void eigen33(const Matrix& mat, Matrix& evecs, Vector& evals)
{
  typedef typename Matrix::Scalar Scalar;
    // Scale the matrix so its entries are in [-1,1].  The scaling is applied
    // only when at least one matrix entry has magnitude larger than 1.

    Scalar scale = mat.cwiseAbs()/*.template triangularView<Lower>()*/.maxCoeff();
    scale = std::max(scale,Scalar(1));
    Matrix scaledMat = mat / scale;

    // Compute the eigenvalues
//     scaledMat.setZero();
    computeRoots(scaledMat,evals);

    // compute the eigen vectors
    // here we assume 3 differents eigenvalues

    // "optimized version" which appears to be slower with gcc!
//     Vector base;
//     Scalar alpha, beta;
//     base <<   scaledMat(1,0) * scaledMat(2,1),
//               scaledMat(1,0) * scaledMat(2,0),
//              -scaledMat(1,0) * scaledMat(1,0);
//     for(int k=0; k<2; ++k)
//     {
//       alpha = scaledMat(0,0) - evals(k);
//       beta  = scaledMat(1,1) - evals(k);
//       evecs.col(k) = (base + Vector(-beta*scaledMat(2,0), -alpha*scaledMat(2,1), alpha*beta)).normalized();
//     }
//     evecs.col(2) = evecs.col(0).cross(evecs.col(1)).normalized();

    // naive version
    Matrix tmp;
    tmp = scaledMat;
    tmp.diagonal().array() -= evals(0);
    evecs.col(0) = tmp.row(0).cross(tmp.row(1)).normalized();

    tmp = scaledMat;
    tmp.diagonal().array() -= evals(1);
    evecs.col(1) = tmp.row(0).cross(tmp.row(1)).normalized();

    tmp = scaledMat;
    tmp.diagonal().array() -= evals(2);
    evecs.col(2) = tmp.row(0).cross(tmp.row(1)).normalized();
    
    // Rescale back to the original size.
    evals *= scale;
}

int main()
{
  BenchTimer t;
  int tries = 10;
  int rep = 400000;
  typedef Matrix3f Mat;
  typedef Vector3f Vec;
  Mat A = Mat::Random(3,3);
  A = A.adjoint() * A;

  SelfAdjointEigenSolver<Mat> eig(A);
  BENCH(t, tries, rep, eig.compute(A));
  std::cout << "Eigen:  " << t.best() << "s\n";

  Mat evecs;
  Vec evals;
  BENCH(t, tries, rep, eigen33(A,evecs,evals));
  std::cout << "Direct: " << t.best() << "s\n\n";

  std::cerr << "Eigenvalue/eigenvector diffs:\n";
  std::cerr << (evals - eig.eigenvalues()).transpose() << "\n";
  for(int k=0;k<3;++k)
    if(evecs.col(k).dot(eig.eigenvectors().col(k))<0)
      evecs.col(k) = -evecs.col(k);
  std::cerr << evecs - eig.eigenvectors() << "\n\n";
}
