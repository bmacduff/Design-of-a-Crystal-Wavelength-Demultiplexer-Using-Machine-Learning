function [ft, f, y] = TimeEvolution(N_nu, V_q_nu, numCavities, B1, t)

    alpha = 1;
    nn = 20;
    hbar = 1.054571 * 10^-34;
    c = 2.99792 * 10^8;
    epsilon0 = 8.854187 * 10^-12;
    aa = 1;

    
    OMEGA0 = 0.307937102281673 - 0.000008461087745i;
    
    omega1 = 0.307483214025719 - 0.000008046645524i;

    k1 = (1/aa) * acos((1/B1)*(1-omega1/OMEGA0));
    
    omega2 = 0.307674933843196 - 0.000008151192949i;
    
    k2 = (1/aa) * acos((1/B1)*(1-omega2/OMEGA0));
    
    q0 = 25;
    
    f = zeros(150 + numCavities, 1);
    f1 = zeros(150 + numCavities, 1);
    f2 = zeros(150 + numCavities, 1);
    
    for q = 1:50
    
        f1(q) = alpha*pi/(nn * aa ^ (3/2)) * sqrt(hbar * omega1 * c/epsilon0) * exp(-(alpha ^ 2 * pi^2 * (q-q0)^2)/(nn^2 * aa^2))  * exp(-1j*aa*k1*(q-q0));
    
        %f2(q) = alpha*pi/(nn * aa ^ (3/2)) * sqrt(hbar * omega2 * c/epsilon0) * exp(-(alpha ^ 2 * pi^2 * (q-q0)^2)/(nn^2 * aa^2)) * exp(-1j*aa*k2*(q-q0));
    
        f(q) = f1(q);% + f2(q);
    
    end
    
    
    ft = exp(-1j*N_nu*t);
    
    C = inv(V_q_nu);
    
    y = V_q_nu * (ft.*(C*f));
    
    y = y';

    % y1 = y(50 + numCavities + 4);
    % y2 = y(50 + numCavities + 7);

end