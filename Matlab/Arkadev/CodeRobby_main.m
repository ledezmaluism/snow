   clc
 clear
close all;
tic 


%%%%% Code written by Arkadev Roy 10/25/2019
T=0.65; % Transmitivity of the output coupler

alphaa=0.00691; % loss for signal
alphab=0.00691; % loss for pump


pin=[0.5]; % average pump input power (Watt)
%pin=[0.6,0.7,0.8,0.9,1];     %%%%%% pin can be a matrix, in that case you can study the effect of input power variation
for f=1:1:length(pin)
%dT=[0.75, 0.85, 0.87, 0.89, 0.91, 0.93]; %%%%%% dT can be a matrix, in
%that case you can study the effect of detuning and find the peak structure

dT=-3.5; % detuning in fs
for q=1:1:length(dT)
deltaT=dT(q);

u=112.778; % walk-off parameter (fs/mm)

L=1; % Length of crystal in mm
beta2a=-53.64; % second order GVD signal (fs^2/mm)
beta3a=756.14; % third order GVD signal (fs^3/mm)
beta4a=-2924.19; % fourth order GVD signal (fs^4/mm)

beta2b=240.92; % second order GVD pump (fs^2/mm)
beta3b=211.285; % second order GVD pump (fs^3/mm)
beta4b=-18.3758; % second order GVD pump (fs^4/mm)

kappa=0.01; % kappa is the effective nonlinear co-efficient

tw=70; % Imput pump pulse width (fs)

l=(3*10^8*deltaT*10^-15)/(1045*10^-9); %detuning converted from fs to l parameter

phi2=25*2; % cavity dispersion second order (fs^2/mm)
phi3=76;  % cavity dispersion third order (fs^3/mm)
phi4=-13020;

nt = 2^12; % number of fft points/ number of discretizations of the fast time axis
Tmax = 1000; % Extent of the fast time axis in fs

dtau = (2*Tmax)/nt;% step size in fast axis
step_num=100; % number of round-trips/ also a measure of slow time evolution

%---tau and omega arrays 
tau = (-nt/2:nt/2-1)*dtau; % discretization of fast axis
omega =  pi/Tmax* [(0:nt/2-1) (-nt/2:-1)]; %corresponding frequency grid



phic=phi2/2*omega.^2+phi3/6*(omega).^3+phi4/24*(omega).^4; % dispersion of cavity in fourier domain

phi=pi*l+(phic)+deltaT*(omega); % the feedback transfer function in fourier domain



pavg=pin(f);% average pump power
p=4*0.88*10^6/tw*pavg; % calculation of peak pump power for sech pulse

dstep=L/50; % discretization in crystal i.e the crystal is divided into 50 segments

m=1;
noise=10^-10*randn(1,length(tau)); % random noise of small amplitude is take as the initial condition for the signal
seed=sqrt(p)*sech(tau/tw*1.76); % input pump profile



temp1=noise; % temporal profile of signal
temp2=noise; % temporal profile of pump

dispersion1 = (alphaa/2-1i*beta2a/2*omega.^2-1i*beta3a/6*omega.^3-1i*beta4a/24*omega.^4); % dispersion of signal in fourier domain
dispersion2 = (alphab/2-1i*beta2b/2*omega.^2-1i*beta3b/6*omega.^3-1i*beta4b/24*omega.^4-1i*u*omega); % dispersion of pump in fourier domain

for n=1:1:step_num % outer loop for roundtrip evolution dynamics
    temp2=seed+noise; % every roundtrip a new pump pulse is injected 
    for x=1:1:L/dstep % inner loop for evolution inside crystal

        %f_temp1 spectral profile of signal
        %f_temp2 spectral profile of pump
    f_temp1 = ifft(temp1).*exp(dispersion1*(-dstep)); % linear split step for signal
    f_temp2 = ifft(temp2).*exp(dispersion2*(-dstep)); % linear split step for pump
    
    
    uu1 = fft(f_temp1); % signal converted to time domain
    uu2 = fft(f_temp2); % pump converted to time domain
    
   [temp1, temp2] = rk4opononlinear(@nonlinearopo,dstep,uu1,uu2,nt,Tmax); % nonlinear split step
      end
   

   temp1=fft(ifft(temp1)*sqrt(1-T).*exp(1i*(phi))); %feedback path
 

      save(m,:)=(abs(temp1)/abs(max(temp1))).^2; %saving the normalized intensity profile of signal every roundtrip
      m=m+1;
    
         
fprintf(' complete %f\n',(step_num*length(dT)*(f-1)+step_num*(q-1)+n)/step_num/length(pin)/length(dT)*100); % helps to keep track of the progress of the simulation
end


uu1=(temp1);  % temporal profile of the signal at steady state
uu2=(temp2);  % temporal profile of the pump after the cavity round-trip
temp1 =fftshift(ifft(uu1)); % spectral profile of the signal at steady state
temp2 =fftshift(ifft(uu2)); % spectral profile of the pump after exiting the cavity
temp3=fftshift(ifft(seed)); % spectral profile of the pump before entering the cavity 

% Different Plots
  figure(3*q-2)

  plot (tau, abs(uu1).^2,'b','LineWidth',3);
  xlabel('Fast Time')
  ylabel('Power(W)')
  hold on
  plot (tau, abs(uu2).^2,'r:','LineWidth',3);
  xlim([-7*tw,7*tw]);
set(gca,'FontSize', 18);

 figure(3*q-1)

   fconv=(3*10^8/2090/10^-9+fftshift(omega/2/pi*10^15));
   lambdaconv=3*10^8./fconv*10^9;
%   
   plot((lambdaconv),(20*log10(abs(temp1/max(temp1)))),'b','LineWidth',3)
   xlim([1500 3000])
   ylim([-30,0])
   xlabel('wavelength (nm)')
   ylabel('PSD (dB)')
%    hold on
%     plot((lambdaconv),(20*log10(abs(temp3/max(temp3)))),'r:','LineWidth',3)
%     
  

figure(3*q)
mesh([1:1:step_num],tau, save')
xlabel('Roundtrips')
ylabel('Fast Time (fs)')
ylim([-4*tw,4*tw])
view(2)
title('Round-trip Evolution')
set(gca,'FontSize', 18);
pause(1)


end
end

toc
