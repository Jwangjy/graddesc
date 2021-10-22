%%% Advanced Econometrics: Problem Set 1
%%% Jieyang Wang, UNI: 4067
clear;
clc;

%% Question 1a)

% Multi-variate vector-valued function G(x)
G = @(x) [
    x(1)^2 + x(2)^2 - 10;
    x(1) - 3*x(2) + 10];

% Jacobian of G
JG = @(x) [
    2*x(1), 2*x(2);
    1, -3];

% Hessian of G
HG = @(x) [
    2, 0;
    0, 0;
    0, 2;
    0, 0];

% Objective function F(x) to minimize in order to solve G(x)=0
F = @(x) 0.5 * sum(G(x).^2);

% Gradient of F (partial derivatives)
dF = @(x) JG(x).' * G(x);

% Hessian of F
HF = @(x) HG(x).'* kron(eye(2), G(x)) + JG(x).'*JG(x);

% Parameters
GAMMA = 0.01;    % step size (learning rate)
MAX_ITER = 1000;  % maximum number of iterations
FUNC_TOL = 0.00001;   % termination tolerance for F(x)

fvals = [];       % store F(x) values across iterations

tic
% Iterate for Gradient Descent method
iter = 1;         % iterations counter
x = [0; 0;];
fvals(iter) = F(x);
while iter < MAX_ITER && fvals(end) > FUNC_TOL
    iter = iter + 1;
    x = x - GAMMA * dF(x);  % gradient descent
    fvals(iter) = F(x);     % evaluate objective function
end
toc

% Plot
figure(1);
plot(1:iter, fvals, 'LineWidth',2); grid on;
title('Objective Function under Gradient Descent'); xlabel('Iteration'); ylabel('F(x)');

% Display of answers
disp(['Number of iterations for Gradient method is: ' num2str(iter(end))])
disp('Solution provided by formula is:')
disp(x)


tic
% Iterate for Newton method
iter1 = 1;      % iterations counter
x = [0; 0];    % initial guess
fvals1 = []; 
fvals1(iter1) = F(x);
while iter1 < MAX_ITER  && fvals1(iter1) > FUNC_TOL
    iter1 = iter1 + 1;
    x = x - inv(HF(x)) * dF(x); % Newton method
    fvals1(iter1) = F(x);     % evaluate objective function
end
toc

% Plot
figure(2);
plot(1:iter1, fvals1, 'LineWidth',2); grid on;
title('Objective Function under Newton method'); xlabel('Iteration'); ylabel('F(x)');

% Display of answers
disp(['Number of iterations for Newton method is: ' num2str(iter(end))])
disp('Solution provided by formula is:')
disp(x)

% For initial guesses of [0,0], [5,5], [1,3], [-5,-5], we all reach the
% same answer of [-1,3] for x and y values. There is only one answer.


%% Question 1b)
x0 = [0,0];
tic
x = fminsearch(F,x0)
toc

% We received the same answer of ~(-1,3) but in a much faster running time.
% We can see that runtime for descent and newton methods are around twice
% as long as for fminsearch


%% Question 1c)
x0 = [0,0];
tic
x = fsolve(G,x0)
toc
% fsolve provides the same answer as our functions above at (-1,3) at
% around the same time taken as the fminsearch function


%% Question 1d)
clear;
clc;

% Multi-variate vector-valued function G(x)
G = @(x) [
    x(1)^2 + x(2)^2 - 26;
    3*x(1)^2 + 25*x(2)^2 - 100];

% Jacobian of G
JG = @(x) [
    2*x(1), 2*x(2);
    6*x(1), 50*x(2)];

% Hessian of G
HG = @(x) [
    2, 0;
    6, 0;
    0, 2;
    0, 50];

% Objective function F(x) to minimize in order to solve G(x)=0
F = @(x) 0.5 * sum(G(x).^2);

% Gradient of F (partial derivatives)
dF = @(x) JG(x).' * G(x);

% Hessian of F
HF = @(x) HG(x).'* kron(eye(2), G(x)) + JG(x).'*JG(x);

% Parameters
GAMMA = 0.0001;    % step size (learning rate)
MAX_ITER = 1000;  % maximum number of iterations
FUNC_TOL = 0.00001;   % termination tolerance for F(x)
x0 = [1;1]; % Initial guess
fvals = [];       % store F(x) values across iterations

% Iterate for Gradient Descent method
iter = 1;         % iterations counter
x = x0;
fvals(iter) = F(x);
while iter < MAX_ITER && fvals(end) > FUNC_TOL
    iter = iter + 1;
    x = x - GAMMA * dF(x);  % gradient descent
    fvals(iter) = F(x);     % evaluate objective function
end
% Plot
figure(3);
plot(1:iter, fvals, 'LineWidth',2); grid on;
title('Objective Function under Gradient Descent'); xlabel('Iteration'); ylabel('F(x)');

% Display of answers
disp(['Number of iterations for Gradient method is: ' num2str(iter(end))])
disp('Solution provided by formula is:')
disp(x)

% Iterate for Newton method
iter = 1;      % iterations counter
x = x0;    % initial guess
fvals = []; 
fvals(iter) = F(x);
while iter < MAX_ITER  && fvals(iter) > FUNC_TOL
    iter = iter + 1;
    x = x - inv(HF(x)) * dF(x); % Newton method
    fvals(iter) = F(x);     % evaluate objective function
end
% Plot
figure(4);
plot(1:iter, fvals, 'LineWidth',2); grid on;
title('Objective Function under Newton Method'); xlabel('Iteration'); ylabel('F(x)');

% Display of answers
disp(['Number of iterations for Newton method is: ' num2str(iter(end))])
disp('Solution provided by formula is:')
disp(x)

% Depending on whether we start at [1,1], [1,-1], [-1,1], or [-1,-1], we
% get four different answers of [5,1], [-5,-1], [-5,1] [5,-1]. Further, the
% descent and newton methods could arrive at two different answers from the
% same initial guess. This is because there are four answers to the set
% of equations.

% For an initial guess value of [0,0], both methods fail to converge as the
% initial start is equidistant from multiple answers.


%% Question 2a) - OLS

rng(2021) % Setting seed
n = 10; % Arbitrary n number
p = 2; % Arbitrary p number
X = randn(n,p);
one = ones([n,1]);
X = [one X]; % Matrix of X
sigma = (p+1)/10;
sigma2 = sigma^2;
theta = -1 + (2).*rand(p+1,1); % Matrix of coefficients
e = (sigma).*randn(n,1); % Matrix of error
y = X*theta + e; % Y values

XX = X.'*X;
XXinv = inv(XX);
Xy = X.'*y;
thetahat_OLS = XXinv * Xy;
SSR = (y-X*thetahat_OLS).'*(y-X*thetahat_OLS);
disp(['The SSR of OLS is: ' num2str(SSR)]);

%% Question 2b) - Gradient Descent

% Multi-variate vector-valued function G(x)
G = @(t) y-X*t;

% Jacobian of G
JG = -X;

% Objective function F(x) to minimize in order to solve G(x)=0
F = @(t) sum(G(t).^2);

% Gradient of F (partial derivatives)
dF = @(t) 2 * JG.' * G(t);

% Hessian of F
HF = 2*X.'*X;

% Parameters
GAMMA = 0.001;    % step size (learning rate)
MAX_ITER = 1000;  % maximum number of iterations
FUNC_TOL = 0.001;   % termination tolerance for F(x)
t0 = zeros([p+1,1]); % Initial guess
fvals = [];       % store F(x) values across iterations

% Iterate for Gradient Descent method
iter = 1;         % iterations counter
t = t0;
fvals(iter) = F(t);
while iter < MAX_ITER && fvals(end) > FUNC_TOL
    iter = iter + 1;
    t = t - GAMMA * dF(t);  % gradient descent
    fvals(iter) = F(t);     % evaluate objective function
end
% Plot
figure(5);
plot(1:iter, fvals, 'LineWidth',2); grid on;
title('Objective Function under Gradient Descent'); xlabel('Iteration'); ylabel('F(x)');

% Display of answers
disp(['Number of iterations for Gradient method is: ' num2str(iter(end))])
disp(['The SSR of gradient descent is: ' num2str(F(t))]);


%% Question 2c) - Newton Descent
% Iterate for Newton method
iter1 = 1;      % iterations counter
t0 = zeros([p+1,1]); % initial guess
fvals1 = []; 
fvals1(iter1) = F(t);
while iter1 < MAX_ITER  && fvals1(iter1) > FUNC_TOL
    iter1 = iter1 + 1;
    t = t - inv(HF) * dF(t); % Newton method
    fvals1(iter1) = F(t);     % evaluate objective function
end

% Plot
figure(6);
plot(1:iter1, fvals1, 'LineWidth',2); grid on;
title('Objective Function under Newton method'); xlabel('Iteration'); ylabel('F(x)');

% Display of answers
disp(['Number of iterations for Newton method is: ' num2str(iter(end))])
disp(['The SSR of Newton descent is: ' num2str(F(t))]);


%% Question 2d) - Comments

% The cost of the three methods remains around same for levels of n from 10 to
% 1000 and p from 2 to 200. At very small values of alpha, gradient descent
% potentially does not reach optimum point in 1000 iterations, while at
% larger values of alpha, gradient descent could potentially diverge.
