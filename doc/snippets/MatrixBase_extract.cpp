Matrix3i m = Matrix3i::Random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is the upper-triangular matrix extracted from m:" << endl
     << m.part<Eigen::Upper>() << endl;
cout << "Here is the strictly-upper-triangular matrix extracted from m:" << endl
     << m.part<Eigen::StrictlyUpper>() << endl;
cout << "Here is the unit-lower-triangular matrix extracted from m:" << endl
     << m.part<Eigen::UnitLower>() << endl;
