clear; clc; close all;
rng(1);

%% Parameters
dt = 0.1;
N  = 30;

umax = 1.0;
gamma = 0.98;

numIter   = 1500;
batchSize = 64;
learnRate = 1e-2;

sigma = 0.3;
goalTol = 0.05;

%% Small NN policy: 2 -> 8 -> 2
nh = 8;

W1 = 0.5*randn(nh,2);
b1 = zeros(nh,1);

W2 = 0.5*randn(2,nh);
b2 = zeros(2,1);

rewardHist = zeros(numIter,1);
successHist = zeros(numIter,1);

%% Training loop
for iter = 1:numIter

    gW1 = zeros(size(W1));
    gb1 = zeros(size(b1));
    gW2 = zeros(size(W2));
    gb2 = zeros(size(b2));

    Rlist = zeros(batchSize,1);
    successList = zeros(batchSize,1);

    epGrads = cell(batchSize,1);

    for ep = 1:batchSize

        p  = -1 + 2*rand(2,1);
        pg = -1 + 2*rand(2,1);

        egW1 = zeros(size(W1));
        egb1 = zeros(size(b1));
        egW2 = zeros(size(W2));
        egb2 = zeros(size(b2));

        R = 0;

        for k = 1:N

            e = p - pg;

            % Forward pass
            a1 = W1*e + b1;
            h1 = tanh(a1);
            zMean = W2*h1 + b2;

            % Exploration in pre-saturation space
            eta = randn(2,1);
            z = zMean + sigma*eta;

            u = umax*tanh(z);

            % Reward
            r = -norm(e);

            if norm(e) < goalTol
                r = r + 100;
                successList(ep) = 1;
                R = R + gamma^(k-1)*r;
                break;
            end

            R = R + gamma^(k-1)*r;

            % log-prob gradient wrt zMean
            dlog_dzMean = (z - zMean)/(sigma^2);

            % Backprop through NN mean
            dz_dW2 = dlog_dzMean*h1';
            dz_db2 = dlog_dzMean;

            dh = W2'*dlog_dzMean;
            da1 = dh.*(1 - h1.^2);

            dz_dW1 = da1*e';
            dz_db1 = da1;

            egW2 = egW2 + dz_dW2;
            egb2 = egb2 + dz_db2;
            egW1 = egW1 + dz_dW1;
            egb1 = egb1 + dz_db1;

            % Dynamics
            p = p + dt*u;
        end

        Rlist(ep) = R;

        epGrads{ep}.W1 = egW1;
        epGrads{ep}.b1 = egb1;
        epGrads{ep}.W2 = egW2;
        epGrads{ep}.b2 = egb2;
    end

    % Advantage normalization
    A = Rlist - mean(Rlist);
    A = A/(std(A) + 1e-8);

    for ep = 1:batchSize
        gW1 = gW1 + A(ep)*epGrads{ep}.W1;
        gb1 = gb1 + A(ep)*epGrads{ep}.b1;
        gW2 = gW2 + A(ep)*epGrads{ep}.W2;
        gb2 = gb2 + A(ep)*epGrads{ep}.b2;
    end

    gW1 = gW1/batchSize;
    gb1 = gb1/batchSize;
    gW2 = gW2/batchSize;
    gb2 = gb2/batchSize;

    % Gradient ascent on expected reward
    W1 = W1 + learnRate*gW1;
    b1 = b1 + learnRate*gb1;
    W2 = W2 + learnRate*gW2;
    b2 = b2 + learnRate*gb2;

    rewardHist(iter) = mean(Rlist);
    successHist(iter) = mean(successList);

    if mod(iter,50)==0
        fprintf('Iter %4d | Avg reward = %8.3f | Success = %.2f\n', ...
            iter,rewardHist(iter),successHist(iter));
    end
end

%% Save policy
save('singleIntegratorNNPolicy.mat','W1','b1','W2','b2','umax');


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


%% Test learned policy
p0 = [-1; -1];
pg = [ 1;  1];


dt = 0.01;
N = 300;
for i = 1:10
p0 = randn(2,1);
pg = randn(2,1);
[pHist,uHist] = simulatePolicy(W1,b1,W2,b2,p0,pg,N,dt,umax);

subplot(2,1,1)
plot(pHist(1,:),pHist(2,:),'LineWidth',2); hold on;
plot(p0(1),p0(2),'ko','MarkerSize',8,'LineWidth',2);
plot(pg(1),pg(2),'rx','MarkerSize',10,'LineWidth',2);
grid on; axis equal; hold on
xlabel('$p_x$','Interpreter','latex');
ylabel('$p_y$','Interpreter','latex');
title('Single-Integrator NN RL Trajectory','Interpreter','latex');

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
function [pHist,uHist] = simulatePolicy(W1,b1,W2,b2,p,pg,N,dt,umax)

    pHist = zeros(2,N+1);
    uHist = zeros(2,N);

    pHist(:,1) = p;

    for k = 1:N
        e = p - pg;

        h1 = tanh(W1*e + b1);
        z = W2*h1 + b2;

        u = umax*tanh(z);

        p = p + dt*u;

        pHist(:,k+1) = p;
        uHist(:,k) = u;
    end
end