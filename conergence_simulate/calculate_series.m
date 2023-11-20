function [sum] = calculate_series(i, coeff, sigma)
sum = 0; 
J = 1;
for j = i : -1 : 1
    eta = 0.5 / (j + 1);
    %eta = min(eta, 1/2);
    sum = sum + 1 * J * eta^coeff;
    J = J * (1 - sigma * eta);
end

% A = 1;
% for i = 1 : N
%     for j = i+1 : N
%         A = A * (1 - sigma * 0.5 / (j + 1));
%     end