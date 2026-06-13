clear; clc; close all;
rng(1);

%% Parameters
dt = 0.1;
N = 40;

umax = 1.0;
gamma = 0.98;

numIter = 500;
batchSize = 16;
learnRate = 5e-3;

goalTol = 0.05;

%% Policy network: input e = p - pg in R^2, output u in R^2
layers = [
    featureInputLayer(2)
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(16)
    tanhLayer
    fullyConnectedLayer(2)
];

net = dlnetwork(layers);

avgGrad = [];
avgSqGrad = [];

rewardHist = zeros(numIter,1);

%% Training
for iter = 1:numIter

    [loss,grad,avgReward] = dlfeval(@lossFcn,net,batchSize,N,dt, ...
        umax,gamma,goalTol);

    [net,avgGrad,avgSqGrad] = adamupdate(net,grad,avgGrad,avgSqGrad, ...
        iter,learnRate);

    rewardHist(iter) = avgReward;

    if mod(iter,50)==0
        fprintf('Iter %4d | Avg reward = %.3f | Loss = %.3f\n', ...
            iter,avgReward,extractdata(loss));
    end
end

figure;
plot(rewardHist,'LineWidth',1.5);
grid on;
xlabel('Iteration');
ylabel('Average reward');
title('Training Reward');

%% Test
close all
p0 = [-1; -1];
pg = [ 1;  1];

p0 = randn(2,1);
pg = randn(2,1);

[pHist,uHist] = simulatePolicy(net,p0,pg,N,dt,umax);

subplot(2,1,1)
plot(pHist(1,:),pHist(2,:),'LineWidth',2); hold on;
plot(p0(1),p0(2),'ko','MarkerSize',8,'LineWidth',2);
plot(pg(1),pg(2),'rx','MarkerSize',10,'LineWidth',2);
grid on; axis equal;
xlabel('$p_x$','Interpreter','latex');
ylabel('$p_y$','Interpreter','latex');
title('RL Single-Integrator Trajectory','Interpreter','latex');

subplot(2,1,2)
plot((0:N-1)*dt,uHist','LineWidth',2);
grid on;
xlabel('Time [s]');
ylabel('Control');
legend('$u_x$','$u_y$','Interpreter','latex');




%% ------------------------------------------------------------
%% Functions
%% ------------------------------------------------------------

function [loss,grad,avgReward] = lossFcn(net,batchSize,N,dt,umax,gamma,goalTol)

    sigma = 0.25;

    logProbList = cell(batchSize,1);
    Rlist = zeros(batchSize,1);

    for ep = 1:batchSize

        p  = -1 + 2*rand(2,1);
        pg = -1 + 2*rand(2,1);

        logProbSum = dlarray(0);
        R = 0;

        for k = 1:N

            e = p - pg;
            edl = dlarray(e,'CB');

            mu = forward(net,edl);
            mu = umax*tanh(mu);

            noise = sigma*randn(2,1);
            u = mu + noise;
            u = umax*tanh(u/umax);

            logProb = -0.5*sum(((u-mu)/sigma).^2);
            logProbSum = logProbSum + logProb;

            r = -norm(e)^2 - 0.01*norm(extractdata(u))^2;

            if norm(e) < goalTol
                r = r + 10;
                R = R + gamma^(k-1)*r;
                break;
            end

            R = R + gamma^(k-1)*r;

            p = p + dt*extractdata(u);
        end

        logProbList{ep} = logProbSum;
        Rlist(ep) = R;
    end

    A = Rlist - mean(Rlist);
    A = A/(std(A) + 1e-6);

    loss = dlarray(0);
    for ep = 1:batchSize
        loss = loss - logProbList{ep}*A(ep);
    end

    loss = loss/batchSize;
    grad = dlgradient(loss,net.Learnables);

    avgReward = mean(Rlist);
end

function [pHist,uHist] = simulatePolicy(net,p,pg,N,dt,umax)

    pHist = zeros(2,N+1);
    uHist = zeros(2,N);

    pHist(:,1) = p;

    for k = 1:N

        e = p - pg;
        edl = dlarray(e,'CB');

        u = forward(net,edl);
        u = umax*tanh(u);
        u = extractdata(u);

        p = p + dt*u;

        pHist(:,k+1) = p;
        uHist(:,k) = u;
    end
end