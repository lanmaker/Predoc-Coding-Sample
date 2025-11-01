%% Galor-Weil Model Simulation

clear all; close all; clc;

%% Parameters and initial conditions

T = 40;
X = 1;
alpha = 0.6;
gamma = 0.225;
rho = 0.879;
tau = 0.2;
tilde_c = 1;
theta = 1;
a_star = 8;

% Initial conditions
g = zeros(T+1,1);
A = zeros(T+1,1);
L = zeros(T+1,1);
e = zeros(T+1,1);
h = zeros(T+1,1);
z = zeros(T+1,1);
c = zeros(T+1,1);
n = zeros(T+1,1);

g(1) = 0.125;      % g_0 (initial growth rate, not directly used in production)
A(1) = 3.17;
L(1) = 0.77;
e(1) = 0;
h(1) = (e(1) + rho*tau)/(e(1) + rho*tau + g(1)) ;          % After calculation, h_0 = 0.584

%% Time loop: Solve the model period by period

for t = 1:T
    % Compute g_{t+1}
    factor = min(theta*L(t), a_star);
    g(t+1) = (e(t) + rho*tau) * factor;
    
    % Update technology: A_{t+1}
    A(t+1) = (1 + g(t+1)) * A(t);
    
    % Compute output per adult z_t
    z(t) = h(t)^alpha * (A(t)/L(t))^(1-alpha);
    
    % Determine whether the subsistence constraint binds
    if (1 - gamma)*z(t) >= tilde_c
        % Interior solution holds:
        c(t) = (1 - gamma)*z(t);
        % Candidate education for next period:
        candidate_e = sqrt((1 - rho)*tau*g(t+1)) - rho*tau;
        e(t+1) = max(candidate_e, 0);
        % Fertility from FOC: n_t*(tau + e(t+1)) = gamma
        n(t) = gamma / (tau + e(t+1));
    else
        % Subsistence binds:
        c(t) = tilde_c;
        candidate_e = sqrt((1 - rho)*tau*g(t+1)) - rho*tau;
        e(t+1) = max(candidate_e, 0);
        % Use resource constraint: z(t)*[1 - n_t*(tau+e(t+1))] = tilde_c 
        % so that n_t*(tau+e(t+1)) = 1 - (tilde_c/z(t))
        n(t) = (1 - (tilde_c/z(t))) / (tau + e(t+1));
        % Compute h_{t+1}
        h(t+1) = (e(t+1) + rho*tau)/(e(t+1) + rho*tau + g(t+1))
    end
    
    % Update population: L_{t+1} = L_t * n_t
    L(t+1) = L(t) * n(t);
    
    % Compute next period's human capital: h_{t+1}
    h(t+1) = (e(t+1) + rho*tau) / (e(t+1) + rho*tau + g(t+1));
end

% For the final period T+1, compute z(T+1) if desired.
z(T+1) = h(T+1)^alpha * (A(T+1)/L(T+1))^(1-alpha);

%% Display the first 10 periods

disp('   t      g_{t+1}    A_{t+1}    L_{t+1}    e_{t+1}    h_t      z_t      c_t      n_t');
for t = 1:10
    fprintf('%3d   %8.4f   %8.4f   %8.4f   %8.4f   %8.4f   %8.4f   %8.4f   %8.4f\n',...
        t-1, g(t+1), A(t+1), L(t+1), e(t+1), h(t), z(t), c(t), n(t));
end

%% Plots: Education and Gross Growth Rates

figure;

% --- Top subplot: Education over time ---
subplot(2,1,1);
plot(0:T, e, 'b-', 'LineWidth',2);
xlabel('Time');
ylabel('Education, e_t');
title('Time Path of Education');
grid on;

% --- Bottom subplot: Gross growth rates ---
% Compute gross growth rates for A, L, and z for t=0 to T-1.
growth_A = A(2:end) ./ A(1:end-1);
growth_L = L(2:end) ./ L(1:end-1);
growth_z = z(2:end) ./ z(1:end-1);

subplot(2,1,2);
plot(0:T-1, growth_A, '-o', 'LineWidth',2); hold on;
plot(0:T-1, growth_L, '-s', 'LineWidth',2);
plot(0:T-1, growth_z, '-^', 'LineWidth',2); hold off;
xlabel('Time');
ylabel('Gross Growth Rate');
legend('A_t', 'L_t', 'z_t','Location','Best');
title('Gross Growth Rates of A_t, L_t, and z_t');
grid on;