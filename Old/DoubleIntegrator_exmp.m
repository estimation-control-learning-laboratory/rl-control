clear; clc; close all;
rng(1);

%% Parameters
dt       = 0.05;
T        = 5;
N        = round(T/dt);

umax     = 2.0;
vmax     = 1.5;

gamma    = 0.99;
numIter  = 3000;
batchSize = 32;
learnRate = 1e-3;

rho_v = 0.05;
rho_u = 0.01;
rho_b = 10.0;

goalTol = 0.05;

%% Policy network: x = [p-pg; v] in R^4, u in R^2
layers = [
    featureInputLayer(4)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(2)
];

net = dlnetwork(layers);

avgGrad = [];
avgSqGrad = [];

rewardHist = zeros(numIter,1);

%% Training loop
for iter = 1:numIter

    [loss,grad,totalReward] = dlfeval(@policyGradientLoss,net,batchSize,N,dt, ...
        umax,vmax,gamma,rho_v,rho_u,rho_b,goalTol);

    [net,avgGrad,avgSqGrad] = adamupdate(net,grad,avgGrad,avgSqGrad,iter,learnRate);

    rewardHist(iter) = totalReward;

    if mod(iter,100)==0
        fprintf('Iter %4d | Avg reward = %.3f | Loss = %.3f\n', ...
            iter,totalReward,extractdata(loss));
    end
end

%% Test learned controller
p0 = [-1; -1];
pg = [ 1;  1];
v0 = [0; 0];

[pHist,vHist,uHist] = simulatePolicy(net,p0,v0,pg,N,dt,umax,false);

figure;
plot(pHist(1,:),pHist(2,:),'LineWidth',2); hold on;
plot(p0(1),p0(2),'ko','MarkerSize',8,'LineWidth',2);
plot(pg(1),pg(2),'rx','MarkerSize',10,'LineWidth',2);
grid on; axis equal;
xlabel('$p_x$','Interpreter','latex');
ylabel('$p_y$','Interpreter','latex');
title('RL Double Integrator Trajectory','Interpreter','latex');

figure;
plot((0:N-1)*dt,vecnorm(vHist(:,1:end-1)),'LineWidth',2); hold on;
yline(vmax,'--','LineWidth',1.5);
grid on;
xlabel('Time [s]');
ylabel('$\|v\|$','Interpreter','latex');
title('Velocity Magnitude','Interpreter','latex');

figure;
plot((0:N-1)*dt,uHist','LineWidth',2);
grid on;
xlabel('Time [s]');
ylabel('Acceleration');
legend('$u_x$','$u_y$','Interpreter','latex');

figure;
plot(rewardHist,'LineWidth',1.5);
grid on;
xlabel('Iteration');
ylabel('Average batch reward');
title('Training Reward');

%% ------------------------------------------------------------
%% Local functions
%% ------------------------------------------------------------

function [loss,grad,avgReward] = policyGradientLoss(net,batchSize,N,dt, ...
    umax,vmax,gamma,rho_v,rho_u,rho_b,goalTol)

    sigma = 0.3;  % exploration std

    totalLoss = dlarray(0);
    rewardsAll = zeros(batchSize,1);

    for ep = 1:batchSize

        p  = -1 + 2*rand(2,1);
        pg = -1 + 2*rand(2,1);
        v  = zeros(2,1);

        logProbSum = dlarray(0);
        R = 0;

        for k = 1:N

            x = [p-pg; v];
            xdl = dlarray(x,'CB');

            mu = forward(net,xdl);
            mu = umax*tanh(mu);

            eps = randn(2,1);
            u = mu + sigma*eps;
            u = umax*tanh(u/umax);

            logProb = -0.5*sum(((u-mu)/sigma).^2);
            logProbSum = logProbSum + logProb;

            pNext = p + dt*v;
            vNext = v + dt*extractdata(u);

            posErr = norm(p-pg);
            velNorm = norm(v);
            uNorm = norm(extractdata(u));

            velViolation = max(0,velNorm-vmax);

            r = -posErr^2 ...
                -rho_v*velNorm^2 ...
                -rho_u*uNorm^2 ...
                -rho_b*velViolation^2;

            if posErr < goalTol && velNorm < 0.1
                r = r + 20;
                R = R + gamma^(k-1)*r;
                break;
            end

            if velNorm > 1.5*vmax
                r = r - 50;
                R = R + gamma^(k-1)*r;
                break;
            end

            R = R + gamma^(k-1)*r;

            p = pNext;
            v = vNext;
        end

        totalLoss = totalLoss - logProbSum*R;
        rewardsAll(ep) = R;
    end

    loss = totalLoss/batchSize;
    grad = dlgradient(loss,net.Learnables);

    avgReward = mean(rewardsAll);
end

function [pHist,vHist,uHist] = simulatePolicy(net,p,v,pg,N,dt,umax,stochastic)

    pHist = zeros(2,N+1);
    vHist = zeros(2,N+1);
    uHist = zeros(2,N);

    pHist(:,1) = p;
    vHist(:,1) = v;

    for k = 1:N
        x = [p-pg; v];
        xdl = dlarray(x,'CB');

        u = forward(net,xdl);
        u = umax*tanh(u);
        u = extractdata(u);

        if stochastic
            u = u + 0.1*randn(2,1);
        end

        p = p + dt*v;
        v = v + dt*u;

        pHist(:,k+1) = p;
        vHist(:,k+1) = v;
        uHist(:,k) = u;
    end
end