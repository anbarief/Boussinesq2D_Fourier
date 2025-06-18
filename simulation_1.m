%%%%% Author: Arief Anbiya
%%%%% Date: 18-6-2025

close all; clear all;

xmax = 70; xmin = -70; x_p = xmax-xmin;
ymax = 70; ymin = -70; y_p = ymax-ymin;
L_x = x_p/2; L_y = y_p/2;
dx = 0.25; x = [(xmin):dx:(xmax)]; N1 = length(x);
dy = 0.25; y = [ymin:dy:ymax]; N2 = length(y);
dt = 0.1; t = [0:dt:20]; nt = length(t);

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

%%%%%%%%%%%%%%%% SETUP INITIAL CONDITIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k=0.6; h=0.6; b0=10^6;
w = sqrt(h^2  + k^2  + k^4);
exact_solution = @(t) -(2*(k^4)*b0)*exp(k*X + h*Y - w*t)./((1 + (k^2)*b0*exp(k*X + h*Y - w*t) ).^2);
prime_initial_condition = (2*b0*(k^4))*( w*exp(k*X + h*Y).*( 1 + b0*(k^2)*exp(k*X + h*Y)).^2 ...
 - 2*w*b0*(k^2)*(1 + b0*(k^2)*exp(k*X + h*Y)).*exp(2*k*X + 2*h*Y))./((1 + b0*(k^2)*exp(k*X + h*Y)).^4);

u_RK4(:,:,1) = exact_solution(0);
v_RK4(:,:,1) = prime_initial_condition;
uhat_RK4(:,:, 1) = trim_matrix.*fft2(u_RK4(:,:,1));
vhat_RK4(:,:, 1) = trim_matrix.*fft2(v_RK4(:,:,1));

u_STRANG(:,:,1) = exact_solution(0);
v_STRANG(:,:,1) = prime_initial_condition;
uhat_STRANG(:,:, 1) = trim_matrix.*fft2(u_STRANG(:,:,1));
vhat_STRANG(:,:, 1) = trim_matrix.*fft2(v_STRANG(:,:,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%% RUNGE-KUTTA 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
percent_updated = 0;
tic;
error_vec_RK4(1) = 0;
for it=[2:1:nt]

k1_a = dt*( vhat_RK4(:,:, it-1)   );
k1_b = dt*( ((((1*j)*2*pi*Wx).^2 +  ((1*j)*2*pi*Wx).^4  + ((1*j)*2*pi*Wy).^2   ).*uhat_RK4(:,:,it-1)) + (-3*((1*j)*(2*pi*Wx)).^2).*(trim_matrix.*fft2(u_RK4(:,:,it-1).^2))  );

k2_a = dt*( (vhat_RK4(:,:,it-1) + 0.5*k1_b) );
k2_b = dt*( (((1*j)*2*pi*Wx).^2 +  ((1*j)*2*pi*Wx).^4   + ((1*j)*2*pi*Wy).^2     ).*(uhat_RK4(:,:,it-1) + 0.5*k1_a) + (-3*((1*j)*(2*pi*Wx)).^2).*(trim_matrix.*fft2(real(ifft2(uhat_RK4(:,:,it-1) + 0.5*k1_a)).^2))  );

k3_a = dt*( (vhat_RK4(:,:,it-1) + 0.5*k2_b) );
k3_b = dt*( (((1*j)*2*pi*Wx).^2 +  ((1*j)*2*pi*Wx).^4  + ((1*j)*2*pi*Wy).^2  ).*(uhat_RK4(:,:,it-1) + 0.5*k2_a) + (-3*((1*j)*(2*pi*Wx)).^2).*(trim_matrix.*fft2(real(ifft2(uhat_RK4(:,:,it-1) + 0.5*k2_a)).^2))  );

k4_a = dt*( (vhat_RK4(:,:,it-1) + k3_b) );
k4_b = dt*( (((1*j)*2*pi*Wx).^2 +  ((1*j)*2*pi*Wx).^4 + ((1*j)*2*pi*Wy).^2 ).*(uhat_RK4(:,:,it-1) + k3_a) + (-3*((1*j)*(2*pi*Wx)).^2).*(trim_matrix.*fft2(real(ifft2(uhat_RK4(:,:,it-1) + k3_a)).^2))  );

uhat_RK4(:, :, it) = (uhat_RK4(:, :, it-1) + (1/6)*(k1_a + k4_a) + (1/3)*(k2_a + k3_a));
vhat_RK4(:, :, it) = (vhat_RK4(:, :, it-1) + (1/6)*(k1_b + k4_b) + (1/3)*(k2_b + k3_b));

u_RK4(:, :, it) = real(ifft2(uhat_RK4(:, :, it)));
error_vec_RK4(it) = max(max(abs( u_RK4(round(N2/2)-20:round(N2/2)+20,:,it) - exact_solution(t(it))(round(N2/2)-20:round(N2/2)+20,:) )));

percent = floor(100*it/nt);
if ((mod(percent,5)==0) && (percent ~= percent_updated))
      disp(['Computing... ', num2str(percent), '%']);
      percent_updated = percent;
endif
##
endfor
toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%% STRANG SPLITTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = (2*pi*Wx).^4  -  (2*pi*Wy).^2 - (2*pi*Wx).^2;
percent_updated = 0;
tic;
error_vec_STRANG(1) = 0;
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

u_STRANG(:, :, it) = real(ifft2(uhat_STRANG(:, :, it)));

error_vec_STRANG(it) = max(max(abs( u_STRANG(round(N2/2)-20:round(N2/2)+20,:,it) - exact_solution(t(it))(round(N2/2)-20:round(N2/2)+20,:) )));

percent = floor(100*it/nt);
if ((mod(percent,5)==0) && (percent ~= percent_updated))
      disp(['Computing... ', num2str(percent), '%']);
      percent_updated = percent;
endif
##
endfor
toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% COMPARE ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(t, error_vec_RK4, '-r', t, error_vec_STRANG, '-b'); legend("RK4", "STRANG");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
answer = input("PLAY ANIMATION? y/n", "s");
while answer == 'y'
close all
surf(X,Y,u_RK4(:,:,1),'edgecolor','none'); colormap(ocean); view(0,90); axis equal; xlabel('x'); ylabel('y'); colorbar;
pause(5);
for i = [2:1:nt]
  if mod(i,2) == 0
   surf(X,Y,u_RK4(:,:,i),'edgecolor','none'); colormap(ocean); view(0,90); axis equal; xlabel('x'); ylabel('y'); colorbar;
  endif
end
answer = input("Play animation again? y/n", "s");
end

