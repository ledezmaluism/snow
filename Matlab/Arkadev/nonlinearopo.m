function [f,g] = nonlinearopo(E1,E2,nt, Tmax)

%%%%% Code written by Arkadev Roy 10/25/2019

Wp= 10*10^-6; % beam waist of pump (m)
Ws= 14*10^-6; % beam waist of signal (m)
deff= 2/pi*16*10^-12; %2/pi*20*10^-12
ns=2.2333; % refractive index of signal
np=2.1935; % refractive index of pump

dtau = (2*Tmax)/nt;% step size in tau
tau = (-nt/2:nt/2-1)*dtau; 

omega =  pi/Tmax* [(0:nt/2-1) (-nt/2:-1)];

Omegas = 2*pi*3*10^8/(2090*10^-9)+omega*10^15;

kappas= sqrt(2*377)*deff*Omegas/Ws/ns/sqrt(pi*np)/(3*10^8)*10^-3;

for i=1:1:length(kappas)
    if(kappas(i)<=0)
        kappas(i)= 0;
    end
end

f = fft((kappas).*ifft(E2.*conj(E1)));

g=-fft((kappas).*(ifft(E1.*E1)));

end