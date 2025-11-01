% ECON 484 Exam 2
% -------------------------------------------------------------------------
% Lifecycle model without SS (T=60, TR=0, ζ=0)
% -------------------------------------------------------------------------
clear; clc; close all;

%% 1) Model parameters
T     = 60;        % periods (no retirement)
gamma = 2;         % leisure exponent
eta   = 2;         % risk aversion
psi   = 0.1;       % consumption shift
A     = 1;         % TFP
theta = 0.3;       % capital share
delta = 0;         % depreciation

%% 2) Scenarios: {beta, z-profile, label}
scenarios = { ...
  struct('beta',0.98, 'z', ones(T,1),                             'lab','Baseline, $\\beta=0.98$'), ...
  struct('beta',0.90, 'z', ones(T,1),                             'lab','Impatient, $\\beta=0.90$'), ...
  struct('beta',0.98, 'z', 1.2 - (0.02*(1:T)' - 1).^2,            'lab','Hump-$z_s$, $\\beta=0.98$') ...
  };
nScen = numel(scenarios);

% Preallocate storage
h_all   = cell(nScen,1);
k_all   = cell(nScen,1);
c_all   = cell(nScen,1);
i_all   = cell(nScen,1);
K       = zeros(nScen,1);
r       = zeros(nScen,1);
agePeak = zeros(nScen,1);
maxK    = zeros(nScen,1);
U       = zeros(nScen,1);
incRat  = zeros(nScen,1);
wthRat  = zeros(nScen,1);

%% 3) fsolve options (Levenberg–Marquardt)
opts = optimoptions('fsolve', ...
    'Display','off', ...
    'Algorithm','levenberg-marquardt', ...
    'MaxFunctionEvaluations',2e4, ...
    'MaxIterations',2e4, ...
    'FunctionTolerance',1e-8, ...
    'StepTolerance',1e-8);

% initial guess: [h(1:T); k(2:T)]
x0 = [0.3*ones(T,1); 0.1*ones(T-1,1)];

%% 4) Solve each scenario (warm‐start)
for i = 1:nScen
    s = scenarios{i};
    [x,h,k,c,i_s,Uval,stats] = solve_model(T, gamma, eta, psi, ...
        s.beta, A, theta, delta, s.z, x0, opts);
    % store
    h_all{i}   = h;
    k_all{i}   = k;
    c_all{i}   = c;
    i_all{i}   = i_s;
    K(i)       = stats.K;
    r(i)       = stats.r;
    agePeak(i) = stats.age_maxk;
    maxK(i)    = stats.maxk;
    U(i)       = Uval;
    % inequality: top 1% vs bottom 40%
    % populations of size T: top1 ≈ richest individual, bottom40 = poorest 0.4*T
    y  = stats.w .* (s.z .* h);
    ys = sort(y);
    incRat(i) = ys(end) / mean(ys(1:floor(0.4*T)));
    ks = sort(k);
    wthRat(i) = ks(end) / mean(ks(1:floor(0.4*T)));
    % warm‐start next
    x0 = x;
end

%% 5) Write LaTeX table
fid = fopen('model_results.tex','w');
fprintf(fid, '\\begin{tabular}{lcccccc}\\hline\\hline\n');
fprintf(fid, 'Scenario & $K$ & $r$ & Peak (age, $k$) & $U$ & Inc. ratio & Wth. ratio\\\\\n');
fprintf(fid, '\\hline\n');
for i = 1:nScen
    fprintf(fid, '%s & %.2f & %.3f & %2d (%.2f) & %.2f & %.2f & %.2f\\\\\n', ...
        scenarios{i}.lab, K(i), r(i), agePeak(i), maxK(i), U(i), incRat(i), wthRat(i));
end
fprintf(fid, '\\hline\\hline\n\\end{tabular}\n');
fclose(fid);

%% 6) Part (a): Baseline life-cycle profiles
ages = (1:T)';
figure('Name','Baseline profiles','Color','w');
subplot(2,2,1), plot(ages, c_all{1}), title('Consumption'),   xlabel('age'), ylabel('c_s');
subplot(2,2,2), plot(ages, i_all{1}), title('Investment'),    xlabel('age'), ylabel('i_s');
subplot(2,2,3), plot(ages, h_all{1}), title('Labor supply'),  xlabel('age'), ylabel('h_s');
subplot(2,2,4), plot(ages, k_all{1}), title('Wealth'),        xlabel('age'), ylabel('k_s');
set(gcf,'PaperPositionMode','auto');
print(gcf, 'profiles_baseline.eps', '-depsc2');

%% 7) Part (c): β=0.90 vs β=0.98 overlays
vars = {'Consumption','Labor','Investment','Wealth'};
for v = 1:4
    figure('Name', [vars{v} ' profiles (β-comp)'], 'Color','w');
    switch v
      case 1, plot(ages, c_all{1}, 'b-', ages, c_all{2}, 'r--','LineWidth',1.5);
      case 2, plot(ages, h_all{1}, 'b-', ages, h_all{2}, 'r--','LineWidth',1.5);
      case 3, plot(ages, i_all{1}, 'b-', ages, i_all{2}, 'r--','LineWidth',1.5);
      case 4, plot(ages, k_all{1}, 'b-', ages, k_all{2}, 'r--','LineWidth',1.5);
    end
    legend({'β=0.98','β=0.90'}, 'Location','northwest');
    xlabel('age'), ylabel([vars{v}(1) '_s']), title([vars{v} ' profiles']);
    set(gcf,'PaperPositionMode','auto');
    print(gcf, ['beta_compare_' lower(vars{v}) '.eps'], '-depsc2');
end

%% 8) Part (d): Productivity hump z_s
% (a) Plot z_s
figure('Name','Age–productivity','Color','w');
plot(ages, ones(T,1),'k--', ages, scenarios{3}.z,'g-','LineWidth',1.5);
legend({'flat','hump-$z_s$'}, 'Interpreter','latex','Location','southeast');
xlabel('age'), ylabel('z_s'), title('Age–productivity');
set(gcf,'PaperPositionMode','auto');
print(gcf, 'age_productivity.eps', '-depsc2');

% (b) Overlay for each variable
for v = 1:4
    figure('Name', [vars{v} ' profiles (z-hump)'], 'Color','w');
    switch v
      case 1, plot(ages, c_all{1}, 'b-', ages, c_all{3}, 'g--','LineWidth',1.5);
      case 2, plot(ages, h_all{1}, 'b-', ages, h_all{3}, 'g--','LineWidth',1.5);
      case 3, plot(ages, i_all{1}, 'b-', ages, i_all{3}, 'g--','LineWidth',1.5);
      case 4, plot(ages, k_all{1}, 'b-', ages, k_all{3}, 'g--','LineWidth',1.5);
    end
    legend({'flat','hump-$z_s$'}, 'Interpreter','latex','Location','northwest');
    xlabel('age'), ylabel([vars{v}(1) '_s']), title([vars{v} ' profiles']);
    set(gcf,'PaperPositionMode','auto');
    print(gcf, ['hump_compare_' lower(vars{v}) '.eps'], '-depsc2');
end

%% ------------------------------------------------------------------------
% solve_model: solves stationary equilibrium for given beta & z_s
function [x,h,k,c,i_s,U,stats] = solve_model( ...
    T,gamma,eta,psi,beta,A,theta,delta,z,x0,opts)

    args = {gamma,eta,psi,beta,A,theta,delta,T,z};
    x    = fsolve(@(x) eqns_lifecycle(x,args{:}), x0, opts);

    % unpack decisions
    h = x(1:T);
    k = [0; x(T+1:end)];    % k(1)=0; k(2:T) from x

    % aggregate
    L = mean(z .* h);
    K = mean(k);
    Y = A * K^theta * L^(1-theta);
    r = theta * Y / K - delta;
    w = (1-theta) * Y / L;

    % consumption & investment
    k1   = [k(2:end); 0];
    c    = (1+r)*k + w.*(z.*h) - k1;
    i_s  = k1 - (1-delta)*k;

    % lifetime utility
    Disc = beta .^ (0:T-1)';
    Uvec=(((c+psi).*(1-h).^gamma).^(1-eta)-1)./(1-eta);
    U   = Disc'*Uvec;

    % stats
    [stats.maxk, stats.age_maxk] = max(k);
    stats.K = K;
    stats.r = r;
    stats.w = w;
end

%% ------------------------------------------------------------------------
% eqns_lifecycle: intratemporal & Euler equations, given x = [h; k2..kT]
function F = eqns_lifecycle(...
    x, gamma,eta,psi,beta,A,theta,delta,T,z)

    % unpack
    h = x(1:T);
    k = [0; x(T+1:end)];

    % aggregates
    L = mean(z .* h);
    K = mean(k);
    Y = A * K^theta * L^(1-theta);
    r = theta * Y / K - delta;
    w = (1-theta) * Y / L;

    % consumption
    k1 = [k(2:end); 0];
    c  = (1+r)*k + w.*(z.*h) - k1;

    % intratemporal FOCs (s=1..T)
    F1 = gamma*(c(1:T)+psi)./(1-h) - w.*z;

    % Euler FOCs (s=1..T-1)
    cs   = c(1:end-1);
    csp1 = c(2:end);
    hs   = h(1:end-1);
    hsp1 = h(2:end);
    term1 = ((csp1+psi)./(cs+psi)).^eta;
    term2 = ((1-hs)./(1-hsp1)).^(gamma*(1-eta));
    F2    = term1 .* term2 - beta*(1+r);

    % stack
    F = [F1; F2];
end