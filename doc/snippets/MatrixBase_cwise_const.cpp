Vector3d v(-1,2,-3);
cout << "the absolute values:" << endl << v.cwise().abs() << endl;
cout << "the absolute values plus one:" << endl << v.cwise().abs().cwise()+1 << endl;
cout << "sum of the squares: " << v.cwise().square().sum() << endl;
