clc
clear
close all;
tic

%%%%% Based on code written by Arkadev Roy 10/25/2019
%%%%% Modified by Luis 1/9/2020
%% Units
% Let's work in the following units:
%%
% * time = fs
% * frequency = PHz (1/time = 10^15 Hz)
% * distance = mm
%%
%
c = physconst('LightSpeed')*1e-12; %mm/fs
%%
% Input Parameters

Nround_trips = 100; % number of round-trips/ also a measure of slow time evolution
T=0.65; % Transmitivity of the output coupler
L=1; % Length of crystal in mm

frep = 250e6; %repetition rate
tw = 70; % Imput pump pulse width (fs)

dstep=L/50; % discretization in crystal i.e the crystal is divided into 50 segments

%Vectors for sweeping
pin=0.5; % average pump input power (Watt)
%pin=[0.6,0.7,0.8,0.9,1];     %%%%%% pin can be a matrix, in that case you can study the effect of input power variation

dT=4; % detuning in fs
%dT=[0.75, 0.85, 0.87, 0.89, 0.91, 0.93]; %%%%%% dT can be a matrix, in
%that case you can study the effect of detuning and find the peak structure

%---tau and omega arrays
NFFT = 2^10; % number of fft points/ number of discretizations of the fast time axis
Tmax = 1000; % Extent of the fast time axis in fs
dtau = (2*Tmax)/NFFT;% step size in fast axis
tau = (-NFFT/2:NFFT/2-1)*dtau; % discretization of fast axis
omega =  pi/Tmax* [(0:NFFT/2-1) (-NFFT/2:-1)]; %corresponding frequency grid

%Crystal dispersion
alphaa=0.00691; % loss for signal
alphab=0.00691; % loss for pump
u=112.778; % walk-off parameter (fs/mm)
beta2a=-53.64; % second order GVD signal (fs^2/mm)
beta3a=756.14; % third order GVD signal (fs^3/mm)
beta4a=-2924.19; % fourth order GVD signal (fs^4/mm)
beta2b=240.92; % second order GVD pump (fs^2/mm)
beta3b=211.285; % second order GVD pump (fs^3/mm)
beta4b=-18.3758; % second order GVD pump (fs^4/mm)

phi2=25*2; % cavity dispersion second order (fs^2/mm)
phi3=76;  % cavity dispersion third order (fs^3/mm)
phi4=-13020;

Da = (alphaa/2-1i*beta2a/2*omega.^2-1i*beta3a/6*omega.^3-1i*beta4a/24*omega.^4); % dispersion of signal in fourier domain
Db = (alphab/2-1i*beta2b/2*omega.^2-1i*beta3b/6*omega.^3-1i*beta4b/24*omega.^4-1i*u*omega); % dispersion of pump in fourier domain
Da = exp(Da*(-dstep)); %dispersion operator
Db = exp(Db*(-dstep)); %dispersion operator

phic = phi2/2*omega.^2+phi3/6*(omega).^3+phi4/24*(omega).^4; % dispersion of cavity in fourier domain

%Nonlinear coupling
Wp= 10*10^-6; % beam waist of pump (m)
Ws= 14*10^-6; % beam waist of signal (m)
deff= 2/pi*16*10^-12; %2/pi*20*10^-12
ns=2.2333; % refractive index of signal
np=2.1935; % refractive index of pump
Omegas = 2*pi*3*10^8/(2090*10^-9)+omega*10^15; %Absolute frequency
kappas = sqrt(2*377)*deff*Omegas/Ws/ns/sqrt(pi*np)/(3*10^8)*10^-3;
kappas = max(kappas,0);
% kappas = 0.01;

noise=10^-10*randn(1,length(tau)); % random noise of small amplitude is take as the initial condition for the signal
%%

for kp=1:1:length(pin)
    for kd=1:1:length(dT)
        
        deltaT=dT(kd); %detuning
        l=(3*10^8*deltaT*10^-15)/(1045*10^-9); %detuning converted from fs to l parameter
        phi=pi*l+(phic)+deltaT*(omega); % the feedback transfer function in fourier domain
        
        pavg=pin(kp);% average pump power
        p = 0.88*pavg/(frep*tw*1e-15); %calculation of peak pump power for sech pulse
        seed=sqrt(p)*sech(tau/tw*1.76); % input pump profile
        
        a = noise; % temporal profile of signal
        b = noise; % temporal profile of pump
        
        data_a = zeros(Nround_trips, NFFT);
        data_b = zeros(Nround_trips, NFFT);
        for n=1:1:Nround_trips % outer loop for roundtrip evolution dynamics
            %b=seed+noise; % every roundtrip a new pump pulse is injected
            b = seed;
            
            [a, b] = single_pass(a, b, L, dstep, Da, Db, kappas);
            
            a=fft(ifft(a)*sqrt(1-T).*exp(1i*(phi))); %feedback path
            data_a(n,:)=(abs(a)/abs(max(a))).^2; %saving the normalized intensity profile of signal every roundtrip
            data_b(n,:)=(abs(b)/abs(max(b))).^2;
            
%             data_a(n,:)= abs(a).^2; %saving the intensity profile of signal every roundtrip
%             data_b(n,:)= abs(b).^2;
            
            
            if mod(n,50)==0
                fprintf('Input Power = %0.2f W, Detuning = %0.2f fs, Completed roundtrip %i\n', [pavg, deltaT, n])
            end
            %fprintf(' complete %f\n',(step_num*length(dT)*(kp-1)+step_num*(kd-1)+n)/step_num/length(pin)/length(dT)*100); % helps to keep track of the progress of the simulation
        end
           
        A = fftshift(ifft(a)); % spectral profile of the signal at steady state
        B = fftshift(ifft(b)); % spectral profile of the pump after exiting the cavity
        temp3 = fftshift(ifft(seed)); % spectral profile of the pump before entering the cavity
        
        % Different Plots
        figure(3*kd-2) 
        plot (tau, abs(a).^2,'b','LineWidth',3);
        xlabel('Fast Time')
        ylabel('Power(W)')
        hold on
        plot (tau, abs(b).^2,'r:','LineWidth',3);
        xlim([-7*tw,7*tw]);
        set(gca,'FontSize', 18);
        
        figure(3*kd-1)
        fconv=(3*10^8/2090/10^-9+fftshift(omega/2/pi*10^15));
        lambdaconv=3*10^8./fconv*10^9;
        plot((lambdaconv),(20*log10(abs(A/max(A)))),'b','LineWidth',3)
        xlim([1500 3000])
        ylim([-30,0])
        xlabel('wavelength (nm)')
        ylabel('PSD (dB)')
        
        figure(3*kd)
        mesh(1:1:Nround_trips,tau, data_a')
        xlabel('Roundtrips')
        ylabel('Fast Time (fs)')
        ylim([-4*tw,4*tw])
        view(2)
        title('Round-trip Evolution')
        set(gca,'FontSize', 18);
        pause(0.1)
        
    end
end

toc

%%
%Functions

function [a,b] = single_pass(a, b, L, dstep, Da, Db, kappas)

for x=1:1:L/dstep % inner loop for evolution inside crystal
    
    %f_temp1 spectral profile of signal
    %f_temp2 spectral profile of pump
    a = ifft(a).*Da;
    b = ifft(b).*Db;
    
    a = fft(a); % signal converted to time domain
    b = fft(b); % pump converted to time domain
    
    [a, b] = rk4opononlinear(@nonlinearopo, dstep, a, b, kappas); % nonlinear split step
end

end

function  [R,S]=rk4opononlinear(nonlinearopo,h,E1,E2,kappas)

[k1, l1] = nonlinearopo(E1,E2, kappas);
k1=h*k1;
l1=h*l1;
[k2, l2] = nonlinearopo(E1+k1/2,E2+l1/2, kappas);
k2= h*k2;
l2=h*l2;
[k3, l3] = nonlinearopo(E1+k2/2,E2+l2/2, kappas);
k3=h*k3;
l3=h*l3;
[k4, l4] = nonlinearopo(E1+k3,E2+l3, kappas);
k4=h*k4;
l4=h*l4;

k = (k1+2*k2+2*k3+k4)/6;
l = (l1+2*l2+2*l3+l4)/6;

R = E1+k;
S = E2+l;
end

function [f,g] = nonlinearopo(E1, E2, kappas)

f = fft((kappas).*ifft(E2.*conj(E1)));
g = -fft((kappas).*(ifft(E1.*E1)));

end