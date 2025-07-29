function recon_sharpness = funcAutoFocusDFS(I)
[M,N] = size(I);
[FX, FY] = gradient(abs(I));
temp = (FX).^2 + (FY).^2;
recon_sharpness = var(temp,0,'all');;