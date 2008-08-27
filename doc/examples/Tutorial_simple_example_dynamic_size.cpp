#include <Eigen/Core>

// import most common Eigen's types 
USING_PART_OF_NAMESPACE_EIGEN

int main(int, char *[])
{
  for (int size=1; size<=4; ++size)
  {
    MatrixXi m(size,size+1);            // creates a size x (size+1) matrix of int
    for (int j=0; j<m.cols(); ++j)      // loop over the columns
      for (int i=0; i<m.rows(); ++i)    // loop over the rows
        m(i,j) = i+j*m.rows();          // to access matrix elements use operator (int,int)
    std::cout << m << "\n\n";
  }

  VectorXf v4(4);
  // to access vector elements
  // you can use either operator () or operator []
  v4[0] = 1; v4[1] = 2; v4(2) = 3; v4(3) = 4;
  std::cout << "\nv4:\n" << v4 << std::endl;
}
