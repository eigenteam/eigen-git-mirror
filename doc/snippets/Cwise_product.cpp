Matrix3i a = Matrix3i::Random(), b = Matrix3i::Random();
Matrix3i c = a.cwise() * b;
cout << "a:\n" << a << "\nb:\n" << b << "\nc:\n" << c << endl;

