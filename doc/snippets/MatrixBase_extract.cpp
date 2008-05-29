Matrix3i m = Matrix3i::random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is the upper-triangular matrix extracted from m:" << endl
     << m.extract<Eigen::Upper>() << endl;
cout << "Here is the strictly-upper-triangular matrix extracted from m:" << endl
     << m.extract<Eigen::StrictlyUpper>() << endl;
cout << "Here is the unit-lower-triangular matrix extracted from m:" << endl
     << m.extract<Eigen::UnitLower>() << endl;
