function recon_sharpness = funcAutoFocusAMS(I)
[M,N] = size(I);
recon_sharpness = sum(sum(abs(I)));
