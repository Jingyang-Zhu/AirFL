%% Setting
sigma = 1;
% sum = \sum_{n} item(n)
% item(n) = eta^(coeff) * J(n)
% eta = 1 / (i^(decay) + 1) ( constant_step ?? )
coeff = [3 2 1 0];  
const_step = true;

%% Run
T = 10000;
N_lst = (1:T)*1;
result = zeros(T, 1);
index = 1;
for co = coeff
    for i = 1 : T
        result(i,index) = calculate_series(i, co, sigma);
    end
    index = index + 1;
end

%%
set(groot,'defaultLineLineWidth',1.5);
figure
grid off
loglog(N_lst, result);
leg = legend('$\delta_2 = 3,\delta_1 = 1$','$\delta_2 = 2,\delta_1 = 1$','$\delta_2 = 1,\delta_1 = 1$','$\delta_2 = 0,\delta_1 = 1$');
set(leg,'FontSize', 12,'FontName','Times New Roman','interpreter','latex','location','best','FontWeight','bold');
ylabel('Value of sequences','Color','k','FontSize',12,'FontName','Times New Roman','FontWeight','bold');
xlabel('Communication round','Color','k','FontSize',12,'FontName','Times New Roman','FontWeight','bold');









