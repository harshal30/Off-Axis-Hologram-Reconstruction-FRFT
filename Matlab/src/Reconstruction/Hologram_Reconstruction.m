clc;
clear all;
close all;
%%

Ho = double(imread('.\data\raw_hologram.tif'));
Ho_upd = padarray(Ho,[250 250],0,'both');
[M,N] = size(Ho_upd);
zeta = 1:M; 
eta = 1:N;
[Zeta,Eta]= meshgrid(zeta,eta);
pixel_pitch = 5.5e-6/20;
lambda = 532*10^(-9);      % wavelength in meter
h1 = M*pixel_pitch;                    % length of hologram
h2 = N*pixel_pitch;
% figure, imagesc(Ho);colormap gray;
% figure, imagesc(Ho_upd);colormap gray;

%%
angles = 0.79:0.001:0.81;
% angles = 0.78:0.01:0.82;
% angles = 0:0.1:1;
frft_o2_1 = zeros(M,N,length(angles));
for i = 1:length(angles)
    frft_o2_1(:,:,i) = frft22d(Ho_upd,[angles(i), angles(i)]);
    imagesc(log(abs(frft_o2_1(:,:,i)))); colormap gray;
    title(['The angle is ',sprintf('%d',angles(i))])
    pause(1);
end
Y_max = zeros(1,length(angles));
for i= 1:length(angles)
    Y_max(i) = max(max((log(abs(frft_o2_1(:,:,i))))));
end
figure, stem(angles,Y_max);
% xlabel('${a}$');
% ylabel('$\max|\mathcal{F}^{a}(I)|$');
%%
spectrum_o = frft22d(Ho_upd,[0.803, 0.803]);
spectrum_abs_o = abs(spectrum_o);
log_spectrum = (log(abs(spectrum_o)));
figure, imagesc(log(abs(spectrum_o)));
%% 
maximum = max(max(spectrum_abs_o(1: N,  1:N)));
[x0, y0] = find(spectrum_abs_o==maximum);

%% Shifting the complex-valued spectrum to the center
spectrum2 = circshift(spectrum_o, [-(x0-(N/2)-0), (N/2)-y0]-0);
figure, imagesc(log(abs(spectrum2))); colormap 'gray';
%%
imageSize = [2548, 2548];
center = imageSize / 2; 
radius = 250; 
[x, y] = meshgrid(1:imageSize(2), 1:imageSize(1));
distance = sqrt((x - center(2)).^2 + (y - center(1)).^2);
mask = distance <= radius;
%%
spectrum3_upd1 = spectrum2 .* mask;

figure, imagesc(log(abs(spectrum3_upd1))); colormap 'gray';

%% PCA Compensation
%IFFT the cropped spectrum to get a subsampled hologram
[M,N] = size(spectrum3_upd1);
IFFT_ROI = fftshift( ifft2( fftshift( spectrum3_upd1 ) ) );
IFFT_ROI = frft22d(spectrum3_upd1,[-0.803,-0.803]);
%get the exponential term
ConjPhase = exp( 1i*angle( IFFT_ROI ) );
% figure, imagesc(angle(ConjPhase));colormap gray;
%singular value decomposition
[ U,S,V ] = svd( ConjPhase ); 

SS = zeros( N,M );
num = 1;%size( S,1 ); %number of principal components will be taken
for i=1:num %take first 'num' values
    SS( i,i ) = S( i,i );
end

%least-squares fitting
Unwrap_U = unwrap( angle( U( :,1:2 ) ) );
SF_U1 = polyfit( 1:N,Unwrap_U( :,1 )',2 ); %second degree
SF_U2 = polyfit( 1:N,Unwrap_U( :,2 )',2 ); %second degree
EstimatedSF_U1 = polyval( SF_U1,1:N );
EstimatedSF_U2 = polyval( SF_U2,1:N );

New_U1 = exp( 1i*( EstimatedSF_U1' ) );
New_U2 = exp( 1i*(EstimatedSF_U2' ) );
U = U*0;
U( :,1:2 ) = [ New_U1 New_U2 ];

Unwrap_V = unwrap( angle( V( :,1:2 ) ) );
SF_V1 = polyfit( 1:N,Unwrap_V( :,1 ).',2 ); %second degree
SF_V2 = polyfit( 1:N,Unwrap_V( :,2 ).',2 ); %second degree
EstimatedSF_V1 = polyval( SF_V1,1:N );
EstimatedSF_V2 = polyval( SF_V2,1:N );

New_V1 = exp( 1i*( EstimatedSF_V1.' ) );
New_V2 = exp( 1i*( EstimatedSF_V2.' ) );
V = V*0;
V( :,1:2 ) = [ New_V1 New_V2 ];

%get the aberration term
Z = U*SS*V';
figure, imagesc(angle(Z));colormap gray;

%FFT and replace the corresponding original region of the spectrum
Psi_pca = fftshift( fft2( fftshift( IFFT_ROI.*conj(Z) ) ) );
% figure, imshow(flipud(rot90(log(abs(Psi_pca)))), []);
% figure, imagesc(abs(Psi_pca))
%% Numerical Reconstruction
z = 0.18;
[M,N] = size(Psi_pca);
z_start = -100e-6;%-180e-6;          % start source-to-sample distance in meter
z_end = 100e-6;%100e-6;            % end source-to-sample distance in meter
z_step =5e-6;%10e-6;          % step source-to-sample distance in meter
p = 5.5e-6;
p = p/20;
lambda = 532*10^(-9);      % wavelength in meter
h1 = M*p;                    % length of hologram
h2 = N*p; 
% OBJECT AND PSF RECONSTRUCTED AT DIFFERENT Z-DISTANCES

S = round((z_end - z_start)/z_step);
reconstructionA = zeros(M,N,S);
L = zeros(1,S);
localAMSA = zeros(1,S);
localGRAA = zeros(1,S);
localDFSA = zeros(1,S);
localSVDA = zeros(1,S);

% hologramI = 0;
for ii = 1:S
z0 = z_start + ii*z_step;
%z0 = 0.9;
% l1 = z0*h1/z;
% l2 = z0*h2/z;
% z1 = z0*(z-z0)/z;
%Mg = z/z0;
% prop = Propagator(N,N, lambda, l1, l2, -z1);
prop = Propagator(M,N, lambda, h1, h2, -z0);
% recA = fftshift(ifft2(fftshift(Psi.*prop)));
% recA =(ifft2(ifftshift(fftshift(frft22d(spectrum3_upd1,[-0.8,-0.8])).*prop)));
% recA = frft22d(spectrum3_upd1.*prop,[-0.8,-0.8]);
recA =fftshift(ifft2(fftshift(Psi_pca.*prop)));
reconstructionA(:,:,ii) = recA(:,:);%(abs(recA)-min(abs(recA(:))))/(max(abs(recA(:)))-min(abs(recA(:))));
%L(ii) = funcAutoFocusAMS((reconstructionA(:,:,ii)));
% imshow(flipud(rot90(abs(reconstructionA(:,:,ii)))), []);
imagesc(abs(reconstructionA(:,:,ii))); colormap gray;
title(['The reconstruction distance is ',sprintf('%d',z0)])
pause(0.5);
localAMSA(ii) = funcAutoFocusAMS(reconstructionA(:,:,ii));
localGRAA(ii) = funcAutoFocusGRA(reconstructionA(:,:,ii));
localDFSA(ii) = funcAutoFocusDFS(reconstructionA(:,:,ii));
localSVDA(ii) = funcAutoFocusSVD(abs(reconstructionA(:,:,ii)), 103);
end

%%

localAMSA = (localAMSA - min(localAMSA))/(max(localAMSA) - min(localAMSA));
localGRAA = (localGRAA - min(localGRAA))/(max(localGRAA) - min(localGRAA));
localDFSA = (localDFSA - min(localDFSA))/(max(localDFSA) - min(localDFSA));
localSVDA = (localSVDA - min(localSVDA))/(max(localSVDA) - min(localSVDA));
X = linspace(-100,100,40);
figure,  plot(X,localAMSA,'LineWidth',1.5,'MarkerSize',4,'Marker','*',...
    'LineStyle',':','color',[1 0 0]); 
xlabel('z(um)','FontSize',12,'FontWeight','bold');  
% xticklabels({'70','72.5','75','77.5','80','82.5','85','87.5','90'});
axis tight;
hold on
plot(X,localGRAA,'LineWidth',1.5,'MarkerSize',4,'Marker','	Diamond',...
    'LineStyle',':','color',[0.47,0.67,0.19]);
 plot(X,localDFSA,'LineWidth',1.5,'MarkerSize',4,'Marker','hexagram',...
    'LineStyle',':','color',[0.30,0.75,0.93]);
plot(X,localSVDA,'LineWidth',1.5,'MarkerSize',4,'Marker','	Diamond',...
    'LineStyle',':','color',[0.4940 0.1840 0.5560]);
legend({'AMS'; 'GRA'; 'DFS'; 'SVD'},'FontSize',11,'FontWeight','bold');
hold off
%%
R =find(localAMSA == max(localAMSA));
% R =find(localSVDA == min(localSVDA));
% R =find(localDFSA == max(localDFSA));
% R =find(localGRAA == min(localGRAA));
% R = 5;
ph = reconstructionA(:,:,R);
phase = angle(ph);
figure; imagesc(phase); colormap jet;
figure; imagesc(angle(reconstructionA(:,:,R))); colormap jet; %colormap(1-colormap);
title(['The reconstruction distance is ',sprintf('%d',z0 - ((ii-R)*z_step)), 'm'])
figure; imagesc(abs(reconstructionA(:,:,R))); colormap gray; axis off;%colormap(1-colormap)
title(['The reconstruction distance is ',sprintf('%d',z0 - ((ii-R)*z_step)), 'm'])
%% Phase unwrapping  
rec_phase_unwrapped = Unwrap_TIE_DCT_Iter(phase);
figure; imagesc(rec_phase_unwrapped); colormap gray; axis on;
title('Reconstructed phase unwrapped')

%% Phase Aberration Compensation
temp = rec_phase_unwrapped;%(201:1800,201:1800);
[M,N] = size(temp);
mask = ones(size(temp));
F = Zernike_func_Cartesian_Cordinate(temp);
[a1,Z1] = zernike_coeffs_cartesian(temp,F,mask);  
Z2 = Z1*a1;
Z2_sim = reshape(Z2,M,N);
True_Phase = temp - Z2_sim;
%True_Phase( True_Phase >= 1 ) = 0;
figure, imagesc(-True_Phase); colormap('gray');axis on;
%% save image
rec_phase_unwrapped1 = -True_Phase;%(30:end-30,30:end-30);
figure, imagesc(rec_phase_unwrapped1); colormap('gray');axis off;
TP = rec_phase_unwrapped1 - min(rec_phase_unwrapped1(:));
TP = (TP - (max(TP(:))/2))/(max(TP(:))/2);
TP = TP*127.5 + 127.5;
figure, imagesc(TP); colormap('gray');axis off;

%%
% function [phase_unwrap]=unwrap_TIE(phase_wrap)
%       psi=exp(1i*phase_wrap);
%       edx = [zeros([size(psi,1),1]), wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
%       edy = [zeros([1,size(psi,2)]); wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];
%        lap = diff(edx, 1, 2) + diff(edy, 1, 1); %calculate Laplacian using the finite difference
%         rho=imag(conj(psi).*lap);   % calculate right hand side of Eq.(4) in the manuscript
%    phase_unwrap = solvePoisson(rho); 
% end
% function phi = solvePoisson(rho)
%     % solve the poisson equation using DCT
%     dctRho = dct2(rho);
%     [N, M] = size(rho);
%     [I, J] = meshgrid([0:M-1], [0:N-1]);
%     dctPhi = dctRho ./ 2 ./ (cos(pi*I/M) + cos(pi*J/N) - 2);
%     dctPhi(1,1) = 0; % handling the inf/nan value
%     % now invert to get the result
%     phi = idct2(dctPhi);
% end