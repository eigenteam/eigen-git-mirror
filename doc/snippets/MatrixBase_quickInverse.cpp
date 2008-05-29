Matrix4d m = Matrix4d::zero();
m.part<Eigen::Upper>().setOnes();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "We know for sure that it is invertible." << endl;
cout << "Here is its inverse:" << m.quickInverse() << endl;
