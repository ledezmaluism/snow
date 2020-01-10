function  [R,S]=rk4opononlinear(nonlinearopo,h,E1,E2, nt, Tmax)

%%%%% Code written by Arkadev Roy 10/25/2019
             
[k1, l1] = nonlinearopo(E1,E2, nt, Tmax);
k1=h*k1;
l1=h*l1;
[k2, l2] = nonlinearopo(E1+k1/2,E2+l1/2, nt, Tmax);
k2= h*k2;
l2=h*l2;
[k3, l3] = nonlinearopo(E1+k2/2,E2+l2/2, nt, Tmax);
k3=h*k3;
l3=h*l3;
[k4, l4] = nonlinearopo(E1+k3,E2+l3, nt, Tmax);
k4=h*k4;
l4=h*l4;

k  = (k1+2*k2+2*k3+k4)/6;
l  = (l1+2*l2+2*l3+l4)/6;

R  = E1+k;
S= E2+l;

end
