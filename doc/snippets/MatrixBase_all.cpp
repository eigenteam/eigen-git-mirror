Vector3f boxMin(Vector3f::Zero()), boxMax(Vector3f::Ones());
Vector3f p0 = Vector3f::Random(), p1 = Vector3f::Random().cwise().abs();
// let's check if p0 and p1 are inside the axis aligned box defined by the corners boxMin,boxMax:
cout << "Is (" << p0.transpose() << ") inside the box: "
     << ((boxMin.cwise()<p0).all() && (boxMax.cwise()>p0).all()) << endl;
cout << "Is (" << p1.transpose() << ") inside the box: "
     << ((boxMin.cwise()<p1).all() && (boxMax.cwise()>p1).all()) << endl;
