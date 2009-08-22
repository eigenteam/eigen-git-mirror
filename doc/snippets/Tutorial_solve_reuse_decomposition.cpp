Matrix3f A(3,3);
A << 1,2,3,  4,5,6,  7,8,10;
PartialLU<Matrix3f> luOfA(A); // compute LU decomposition of A
Vector3f b;
b << 3,3,4;
Vector3f x;
luOfA.solve(b, &x);
cout << "The solution with right-hand side (3,3,4) is:" << endl;
cout << x << endl;
b << 1,1,1;
luOfA.solve(b, &x);
cout << "The solution with right-hand side (1,1,1) is:" << endl;
cout << x << endl;
