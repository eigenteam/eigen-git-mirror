RowVector4i v = RowVector4i::random();
cout << "Here is the vector v:" << endl << v << endl;
cout << "Here is v.end(2):" << endl << v.end(2) << endl;
v.end(2).setZero();
cout << "Now the vector v is:" << endl << v << endl;
