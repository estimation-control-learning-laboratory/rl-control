clear; clc; close all;
rng(1);

%% Parameters
dt = 0.1;
N  = 30;

umax = 1.0;
gamma = 0.98;

numIter   = 1000;
batchSize = 64;
learnRate = 2e-2;

sigma = 0.3;
goalTol = 0.05;

%% Policy parameters
% u = umax*tanh(W*e + b + sigma*noise)
W = 0.1*randn(2,2);
b = zeros(2,1);

rewardHist = zeros(numIter,1);
successHist = zeros(numIter,1);

%% Training loop
for iter = 1:numIter

    gradW = zeros(size(W));
    gradb = zeros(size(b));

    Rlist = zeros(batchSize,1);
    dWlist = zeros(2,2,batchSize);
    dblist = zeros(2,1,batchSize);
    successList = zeros(batchSize,1);

    for ep = 1:batchSize

        p  = -1 + 2*rand(2,1);
        pg = -1 + 2*rand(2,1);

        epGradW = zeros(size(W));
        epGradb = zeros(size(b));

        R = 0;

        for k = 1:N

            e = p - pg;

            zMean = W*e + b;
            eta = randn(2,1);
            z = zMean + sigma*eta;

            u = umax*tanh(z);

            % reward
            r = -norm(e);

            if norm(e) < goalTol
                r = r + 100;
                successList(ep) = 1;
                R = R + gamma^(k-1)*r;
                break;
            end

            R = R + gamma^(k-1)*r;

            % log probability gradient for Gaussian in z-space
            % log pi ~ -0.5 ||(z-zMean)/sigma||^2
            dlog_dzMean = (z - zMean)/(sigma^2);

            epGradW = epGradW + dlog_dzMean*e';
            epGradb = epGradb + dlog_dzMean;

            % dynamics
            p = p + dt*u;
        end

        Rlist(ep) = R;
        dWlist(:,:,ep) = epGradW;
        dblist(:,:,ep) = epGradb;
    end

    % Advantage normalization
    A = Rlist - mean(Rlist);
    A = A/(std(A) + 1e-8);

    for ep = 1:batchSize
        gradW = gradW + A(ep)*dWlist(:,:,ep);
        gradb = gradb + A(ep)*dblist(:,:,ep);
    end

    gradW = gradW/batchSize;
    gradb = gradb/batchSize;

    % gradient ascent on expected reward
    W = W + learnRate*gradW;
    b = b + learnRate*gradb;

    rewardHist(iter) = mean(Rlist);
    successHist(iter) = mean(successList);

    if mod(iter,50)==0
        fprintf('Iter %4d | Avg reward = %8.3f | Success = %.2f\n', ...
            iter,rewardHist(iter),successHist(iter));
    end
end

figure;
plot(rewardHist,'LineWidth',1.5);
grid on;
xlabel('Iteration');
ylabel('Average reward');
title('Training Reward');

figure;
plot(successHist,'LineWidth',1.5);
grid on;
xlabel('Iteration');
ylabel('Success rate');
title('Training Success Rate');

disp('Learned W:');
disp(W);

disp('Learned b:');
disp(b);
%% Save policy
save('singleIntegratorLinearRLPolicy.mat','W','b','umax');

%% Test learned policy
p0 = [-1; -1];
pg = [ 1;  1];

for i = 1:100
p0 = randn(2,1);
pg = randn(2,1);

[pHist,uHist] = simulatePolicy(W,b,p0,pg,N,dt,umax);

subplot(2,1,1)
plot(pHist(1,:),pHist(2,:),'LineWidth',2); hold on;
plot(p0(1),p0(2),'ko','MarkerSize',8,'LineWidth',2);
plot(pg(1),pg(2),'rx','MarkerSize',10,'LineWidth',2);
grid on; axis equal; hold on
xlabel('$p_x$','Interpreter','latex');
ylabel('$p_y$','Interpreter','latex');
title('Single-Integrator RL Trajectory','Interpreter','latex');

subplot(2,1,2)
plot((0:N-1)*dt,uHist','LineWidth',2);
grid on; hold on
xlabel('Time [s]');
ylabel('Control');
legend('$u_x$','$u_y$','Interpreter','latex');
end


%% ------------------------------------------------------------
%% Local function
%% ------------------------------------------------------------
function [pHist,uHist] = simulatePolicy(W,b,p,pg,N,dt,umax)

    pHist = zeros(2,N+1);
    uHist = zeros(2,N);

    pHist(:,1) = p;

    for k = 1:N
        e = p - pg;

        z = W*e + b;
        u = umax*tanh(z);

        p = p + dt*u;

        pHist(:,k+1) = p;
        uHist(:,k) = u;
    end
end