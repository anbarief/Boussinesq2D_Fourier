%%%%% Author: Arief Anbiya
%%%%% Date: 18-6-2025

close all; clear all;

xmax = 40; xmin = -40; x_p = xmax-xmin;
ymax = 20; ymin = -20; y_p = ymax-ymin;
L_x = x_p/2; L_y = y_p/2;
dx = 0.25; x = [(xmin):dx:(xmax)]; N1 = length(x);
dy = 0.25; y = [ymin:dy:ymax]; N2 = length(y);
dt = 0.1; t = [0:dt:50]; nt = length(t);

if (mod(N1,2) == 0)
    kx_right = [0:1:((N1/2)-1)]; kx_left = [(-N1/2):1:-1];
else
    kx_right = [0:1:(((N1-1)/2))]; kx_left = [(-(N1-1)/2):1:-1];
end
kx_total = [kx_right, kx_left]; wx = transpose([kx_right/x_p, kx_left/x_p]);

if (mod(N2,2) == 0)
    ky_right = [0:1:((N2/2)-1)]; ky_left = [(-N2/2):1:-1];
else
    ky_right = [0:1:(((N2-1)/2))]; ky_left = [(-(N2-1)/2):1:-1];
end
ky_total = [ky_right, ky_left]; wy = transpose([ky_right/y_p, ky_left/y_p]);

[Kx, Ky] = meshgrid(kx_total, ky_total); [Wx, Wy] = meshgrid(wx, wy);
x0 = 0; y0 = 0; [X,Y] = meshgrid(x,y);

trim_matrix = ones(N2,N1);
for i=1:1:N1
  for ii=1:1:N2
  if (pi^2) > ((L_x^4)*(ky_total(ii)^2) + ((L_x*L_y)^2)*kx_total(i)^2)/((L_y^2)*(kx_total(i)^4))
    trim_matrix(ii,i)=0;
  endif
  endfor
endfor

%%%%% boundary set %%%%%%%%%
boundary_matrix = ones(N2,N1);
for j_idx = [1:1:N2]
  for i_idx = [1:1:N1]

  if   (y(j_idx) >= ymin) && (y(j_idx) <= ymin+5)
    boundary_matrix(j_idx,i_idx) = 0;
  endif

  if   (y(j_idx) >= ymax-5) && (y(j_idx) <= ymax)
    boundary_matrix(j_idx,i_idx) = 0;
  endif

  endfor
endfor
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%% SETUP INITIAL CONDITIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha=0.25; beta=2; b0=16;
co = sqrt(beta^2  + alpha^2  + alpha^4);
exact_solution = @(t) -(2*(alpha^4)*b0)*exp(alpha*X + beta*Y - co*t)./((1 + (alpha^2)*b0*exp(alpha*X + beta*Y - co*t) ).^2);
prime_initial_condition = (2*b0*(alpha^4))*( co*exp(alpha*X + beta*Y).*( 1 + b0*(alpha^2)*exp(alpha*X + beta*Y)).^2 ...
 - 2*co*b0*(alpha^2)*(1 + b0*(alpha^2)*exp(alpha*X + beta*Y)).*exp(2*alpha*X + 2*beta*Y))./((1 + b0*(alpha^2)*exp(alpha*X + beta*Y)).^4);

u_STRANG(:,:,1) = exact_solution(0);
v_STRANG(:,:,1) = prime_initial_condition;
uhat_STRANG(:,:, 1) = trim_matrix.*fft2(u_STRANG(:,:,1));
vhat_STRANG(:,:, 1) = trim_matrix.*fft2(v_STRANG(:,:,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%% STRANG SPLITTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = (2*pi*Wx).^4  -  (2*pi*Wy).^2 - (2*pi*Wx).^2;
percent_updated = 0;
tic;
for it=[2:1:nt]


%%STEP 1 NONLINEAR:
uhat_nonlin1 = uhat_STRANG(:,:,it-1);
what_nonlin1 = fft2(real(ifft2(uhat_nonlin1)).^2);
vhat_nonlin1 = vhat_STRANG(:,:,it-1) + (dt/2)*( 3*what_nonlin1.*(2*pi*Wx).^2  );

%%STEP 2 LINEAR:
uhat_lin = 0.5*(uhat_nonlin1 + (vhat_nonlin1)./(j*sqrt(-lambda))).*exp(j*sqrt(-lambda)*dt) ...
+  (uhat_nonlin1  - 0.5*(uhat_nonlin1 + (vhat_nonlin1)./(j*sqrt(-lambda))) ).*exp(-j*sqrt(-lambda)*dt);

vhat_lin = (j*sqrt(-lambda)).*(0.5*(uhat_nonlin1 + (vhat_nonlin1)./(j*sqrt(-lambda))).*exp(j*sqrt(-lambda)*dt)) ...
+  (-j*sqrt(-lambda)).*(uhat_nonlin1  - 0.5*(uhat_nonlin1 + (vhat_nonlin1)./(j*sqrt(-lambda))) ).*exp(-j*sqrt(-lambda)*dt);

uhat_lin(1,1) = uhat_nonlin1(1,1) + vhat_nonlin1(1,1)*dt;
vhat_lin(1,1) = vhat_nonlin1(1,1);

%%STEP 3 NONLINEAR:
uhat_nonlin2 = trim_matrix.*uhat_lin;
what_nonlin2 = trim_matrix.*fft2(real(ifft2(uhat_nonlin2)).^2);
vhat_nonlin2 = vhat_lin + (dt/2)*( 3*what_nonlin2.*(2*pi*Wx).^2  );

%%FINALIZE
uhat_STRANG(:,:,it) = trim_matrix.*uhat_nonlin2;
vhat_STRANG(:,:,it) = trim_matrix.*vhat_nonlin2;

u_STRANG(:, :, it) = boundary_matrix.*real(ifft2(uhat_STRANG(:, :, it)));
v_STRANG(:, :, it) = boundary_matrix.*real(ifft2(vhat_STRANG(:, :, it)));

uhat_STRANG(:,:, it) = trim_matrix.*fft2(u_STRANG(:,:,it));
vhat_STRANG(:,:, it) = trim_matrix.*fft2(v_STRANG(:,:,it));

percent = floor(100*it/nt);
if ((mod(percent,5)==0) && (percent ~= percent_updated))
      disp(['Computing... ', num2str(percent), '%']);
      percent_updated = percent;
endif
##
endfor
toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
answer = input("PLAY ANIMATION? y/n", "s");
while answer == 'y'
close all
surf(X,Y,u_STRANG(:,:,1),'edgecolor','none'); colormap(ocean); view(0,90); axis equal; xlabel('x'); ylabel('y'); colorbar;
pause(5);
for i = [2:1:nt]
  if mod(i,2) == 0
   surf(X,Y,u_STRANG(:,:,i),'edgecolor','none'); colormap(ocean); view(0,90); axis equal; xlabel('x'); ylabel('y'); colorbar;
   title(["t=", num2str(t(i))], 'FontSize', 20);
   pause(0.01);
  endif
end
answer = input("Play animation again? y/n", "s");
end

