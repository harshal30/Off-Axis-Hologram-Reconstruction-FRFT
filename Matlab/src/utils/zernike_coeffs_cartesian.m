function [a1,Z1] = zernike_coeffs_cartesian(PW2, Z,mask)

P1 = size(Z,3);

[M,N] = size(PW2);

phi = reshape(PW2,M*N,1);
mask1 = reshape(mask,M*N,1);
for i=1:P1
    temp = reshape(Z(:,:,i),M*N,1);
    Z1(:,i) = temp;
    temp(mask1(:) == 0) = [];
    Z11(:,i) = temp;
end
phi1 = phi;
phi1(mask1(:) == 0) = [];
a1 = pinv(Z11)*phi1;