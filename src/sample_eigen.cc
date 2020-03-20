#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

int main()
{
    int m = 3;
    int n = 2;
    MatrixXf mat(3, 2);
    MatrixXf mat2(2, 2);
    std::cout << "Here is mat:\n" << mat << std::endl;
    std::cout << "Here is mat:\n" << mat2 << std::endl;
    //mat << 1, 2,
    //3, 4;
    mat.row(0).col(0) << 1.1;
    mat.row(0).col(1) << 1.1;
    mat.row(1).col(0) << 1.1;
    mat.row(1).col(1) << 1.1;
    mat.row(2).col(0) << 1.1;
    mat.row(2).col(1) << 1.1;

    mat2.row(0).col(0) << 1.1;
    mat2.row(0).col(1) << 1.1;
    mat2.row(1).col(0) << 1.1;
    mat2.row(1).col(1) << 1.1;

    std::cout << "Here is mat:\n" << mat << std::endl;
    std::cout << "Here is mat:\n" << mat2 << std::endl;
    std::cout << "Here is mat*mat:\n" << mat*mat2 << std::endl;
}
