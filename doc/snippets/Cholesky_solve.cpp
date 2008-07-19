typedef Matrix<float,Dynamic,2> DataMatrix;
// let's generate some samples on the 3D plane of equation z = 2x+3y (with some noise)
DataMatrix samples = DataMatrix::random(12,2);
VectorXf elevations = 2*samples.col(0) + 3*samples.col(1) + VectorXf::random(12)*0.1;
// and let's solve samples * x = elevations in least square sense:
cout << (samples.adjoint() * samples).cholesky().solve((samples.adjoint()*elevations).eval()) << endl;
