Matrix2d md = Matrix2d::identity() * 0.45;
Matrix2f mf = Matrix2f::identity();
cout << md + mf.cast<double>() << endl;
