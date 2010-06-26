#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
int main()
{
  Vector3d v(1,2,3);
  Vector3d w(0,1,2);

  std::cout << "Dot product: " << v.dot(w) << std::endl;
  std::cout << "Cross product:\n" << v.cross(w) << std::endl;
}
