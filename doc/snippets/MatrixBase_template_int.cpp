RowVector5i v = RowVector5i::Random();
cout << "Here is the vector v:" << endl << v << endl;
cout << "Here is v.block<2>(1):" << endl << v.start<2>() << endl;
v.block<2>(2).setZero();
cout << "Now the vector v is:" << endl << v << endl;
