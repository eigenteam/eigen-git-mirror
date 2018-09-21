Matrix4i m = Matrix4i::Random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is m(all).transpose():" << endl << m(all).transpose() << endl;
cout << "Here is m.reshaped(fix<1>,AutoSize):" << endl << m.reshaped(fix<1>,AutoSize) << endl;
