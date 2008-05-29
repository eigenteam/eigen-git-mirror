Matrix4i m = Matrix4i::random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is the bottom-right 2x3 corner in m:" << endl
     << m.corner<2,3>(Eigen::BottomRight) << endl;
m.corner<2,3>(Eigen::BottomRight).setZero();
cout << "Now the matrix m is:" << endl << m << endl;
