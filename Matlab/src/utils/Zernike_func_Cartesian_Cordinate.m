function F = Zernike_func_Cartesian_Cordinate (A1)

[M,N] = size(A1);
x = linspace(0,1,N);
y = linspace(0,1,M);
[X,Y] = meshgrid(x,y);

F = zeros(M, N, 11);
F(:,:,1) = 1;
F(:,:,2) = 2*X;
F(:,:,3) = 2*Y;
F(:,:,4) = sqrt(3)*(2*X.^2 + 2*Y.^2 -1);
F(:,:,5) = sqrt(6)*(2*X.*Y);
F(:,:,6) = sqrt(6)*(X.^2 - Y.^2);
F(:,:,7) = sqrt(8)*(3*X.^2.*Y + 3*Y.^3 -2*Y);
F(:,:,8) = sqrt(8)*(3*X.^3 + 3*X.*Y.^2 -2*X);
F(:,:,9) = sqrt(8)*(3*X.^2.*Y - Y.^3);
F(:,:,10) = sqrt(8)*(X.^3 - 3*X.*Y.^2);
F(:,:,11) = sqrt(10)*(4*X.^3.*Y + 4*X.*Y.^3);

% F(:,:,11) = sqrt(10)*(- 6*X.*Y + 8*X.^3.*Y + 8*X.*Y.^3);


