s = settings;
s.matlab.editor.AllowFigureAnimation.PersonalValue = true;

for rt = 1:Nround_trips
    a = data_a(rt,:);
    b = data_b(rt,:);
    A = fftshift(ifft(a));
    B = fftshift(ifft(b));  
    
    subplot(2,1,1);
    plot(tau, a, tau, b);
    title(['Round Trip ' num2str(rt)])
    ylim([0, 3e4]);
    grid('on')
    
    subplot(2,1,2);
    plot(fftshift(omega/2/pi), abs(A), fftshift(omega/2/pi), abs(B))
    ylim([0, 1e3]);
    
    %drawnow
    pause(0.05)
end