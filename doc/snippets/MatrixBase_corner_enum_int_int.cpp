Matrix4i m = Matrix4i::random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is the bottom-right 2x3 corner in m:" << endl
     << m.corner(Eigen::BottomRight, 2, 3) << endl;
m.corner(Eigen::BottomRight, 2, 3).setZero();
cout << "Now the matrix m is:" << endl << m << endl;
