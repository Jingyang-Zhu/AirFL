clear all;
clc;
rng(1)
%% 参数设置
tic
m = 25;                                                                     %workers 
d = 100;
iter = 1002;                                                                 %iteration
lr = 0.05;
count = 1;
sigma2 = 0.25;
% odmean = 500;
% odmin = 300;
% odmax = 1200;
% odeps = 0.1;
% od00 = randi([odmin odmax],m,1);
% while abs(odmean - mean(od00)) >= odeps
%     if odmean > mean(od00)
%         od00(find(od00 < odmean,1)) = randi([odmean odmax]);
%     elseif odmean < mean(od00)
%         od00(find(od00 > odmean,1)) = randi([odmin odmean]);
%     end
% end
% D = od00;
D = 500 * ones(m,1);
pi = D / sum(D);
batchsize = 500;
SNR = 0; %dB

%% 数据生成与清洗2
theta_truth = normrnd(0,1,d,1);
for n = 1:m
    sample = D(n);
    x{n} = normrnd(0,1,sample,d);
    v{n} = normrnd(0,sigma2,sample,1);
    theta = theta_truth;
    y{n} = x{n} * theta + v{n}; 
end

%% Fixed Local Steps
for n = 1:m
    epoch(n) = 5;
    epoch1(n) = 10;
end

%% 最优解
[xopavg,Fopavg] = optimal_linear_reg(x,y,d,m,pi,D,count,10,0.01,theta_truth);
FedSGD_obj = FedAvg_errorfree(x,y,lr,m,d,D,pi,iter,Fopavg,count,ones(m),batchsize);
FedAvg_obj = FedAvg_errorfree(x,y,lr,m,d,D,pi,iter,Fopavg,count,epoch,batchsize);
FedAvg_obj1 = FedAvg_errorfree(x,y,lr,m,d,D,pi,iter,Fopavg,count,epoch1,batchsize);

%% gap计算
FedSGD_gap = abs(mean(FedSGD_obj - Fopavg,1))';
FedAvg_gap = abs(mean(FedAvg_obj - Fopavg,1))';
FedAvg_gap1 = abs(mean(FedAvg_obj1 - Fopavg,1))';

%% 误差图
set(groot,'defaultLineLineWidth',1.5);
figure
semilogy(0:iter-2,FedSGD_gap,'-s',0:iter-2,FedAvg_gap,'-^',0:iter-2,FedAvg_gap1,'-o',...
     'MarkerIndices',[1:(iter-2)/25:iter-2],'MarkerSize',6);
grid off;
set(gca,'ylim',[1.0000e-10 1.0000e+2],'YTick',...
     [1e-10 1e-8 1e-6 1e-4 1e-2 1e+0 1e+2],...
     'FontSize',12,'FontName','Times New Roman',...
     'XTick',[0 (iter-2)/5:(iter-2)/5:iter-2]);
% % set(gca,'ylim',[0 2.0000e+5],...
% %     'FontSize',12,'FontName','Times New Roman',...
% %     'XTick',[0 (iter-2)/10:(iter-2)/10:iter-2]);
leg = legend('FedAvg, E = 1','FedAvg, E = 5','FedAvg, E = 5');
set(leg,'FontSize', 12,'FontName','Times New Roman','FontWeight','bold','location','best');
xlabel('Number of Communication Round, \itt','Color','k','FontSize',12,'FontName','Times New Roman','FontWeight','bold');
ylabel('Optimality gap','Color','k','FontSize',12,'FontName','Times New Roman','FontWeight','bold');
grid on;
% filename = 'result\final_1.eps';
% saveas(gcf,filename);
toc
