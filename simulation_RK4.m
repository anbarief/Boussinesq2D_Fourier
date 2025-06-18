%%%%% Author: Arief Anbiya
%%%%% Date: 18-6-2025

close all; clear all;

xmax = 40; xmin = -40; x_p = xmax-xmin;
ymax = 20; ymin = -20; y_p = ymax-ymin;
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
alpha=0.25; beta=2; b0=16;
co = sqrt(beta^2  + alpha^2  + alpha^4);
exact_solution = @(t) -(2*(alpha^4)*b0)*exp(alpha*X + beta*Y - co*t)./((1 + (alpha^2)*b0*exp(alpha*X + beta*Y - co*t) ).^2);
prime_initial_condition = (2*b0*(alpha^4))*( co*exp(alpha*X + beta*Y).*( 1 + b0*(alpha^2)*exp(alpha*X + beta*Y)).^2 ...
 - 2*co*b0*(alpha^2)*(1 + b0*(alpha^2)*exp(alpha*X + beta*Y)).*exp(2*alpha*X + 2*beta*Y))./((1 + b0*(alpha^2)*exp(alpha*X + beta*Y)).^4);

u_RK4(:,:,1) = exact_solution(0);
v_RK4(:,:,1) = prime_initial_condition;
uhat_RK4(:,:, 1) = trim_matrix.*fft2(u_RK4(:,:,1));
vhat_RK4(:,:, 1) = trim_matrix.*fft2(v_RK4(:,:,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%% RUNGE-KUTTA 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
percent_updated = 0;
tic;
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

percent = floor(100*it/nt);
if ((mod(percent,5)==0) && (percent ~= percent_updated))
      disp(['Computing... ', num2str(percent), '%']);
      percent_updated = percent;
endif
##
endfor
toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
answer = input("PLAY ANIMATION? y/n", "s");
while answer == 'y'
close all
surf(X,Y,u_RK4(:,:,1),'edgecolor','none'); colormap(ocean); view(0,90); axis equal; xlabel('x'); ylabel('y'); colorbar;
pause(5);
for i = [2:1:nt]
  if mod(i,2) == 0
   surf(X,Y,u_RK4(:,:,i),'edgecolor','none'); colormap(ocean); view(0,90); axis equal; xlabel('x'); ylabel('y'); colorbar;
   title(["t=", num2str(t(i))], 'FontSize', 20);
   pause(0.01);
  endif
end
answer = input("Play animation again? y/n", "s");
end

