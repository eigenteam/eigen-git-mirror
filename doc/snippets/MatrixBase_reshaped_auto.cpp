Matrix4i m = Matrix4i::Random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is m.reshaped(2, AutoSize):" << endl << m.reshaped(2, AutoSize) << endl;
cout << "Here is m.reshaped(AutoSize, fix<8>, RowOrder):" << endl << m.reshaped(AutoSize, fix<8>, RowOrder) << endl;
