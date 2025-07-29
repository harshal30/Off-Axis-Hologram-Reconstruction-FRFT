function recon_sharpness = funcAutoFocusGRA(I)
[M,N] = size(I);
[FX, FY] = gradient(abs(I));
temp = (FX).^2 + (FY).^2;
recon_sharpness  = mean(temp(:));