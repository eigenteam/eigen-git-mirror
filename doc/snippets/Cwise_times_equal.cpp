Vector3d v(1,2,3);
Vector3d w(2,3,0);
v.cwise() *= w;
cout << v << endl;
