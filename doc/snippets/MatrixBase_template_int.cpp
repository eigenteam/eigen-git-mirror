RowVector4i v = RowVector4i::Random();
cout << "Here is the vector v:" << endl << v << endl;
cout << "Here is v.block<2>(1):" << endl << v.block<2>(1) << endl;
v.block<2>(2).setZero();
cout << "Now the vector v is:" << endl << v << endl;
