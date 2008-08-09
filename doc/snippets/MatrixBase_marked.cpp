Matrix3d m = Matrix3d::Zero();
m.part<Eigen::Upper>().setOnes();
cout << "Here is the matrix m:" << endl << m << endl;
Matrix3d n = Matrix3d::Ones();
n.part<Eigen::Lower>() *= 2;
cout << "Here is the matrix n:" << endl << n << endl;
cout << "And now here is m.inverse()*n, taking advantage of the fact that"
        " m is upper-triangular:" << endl
     << m.marked<Eigen::Upper>().solveTriangular(n);
