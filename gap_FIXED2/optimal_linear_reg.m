function [xop,Fop] = optimal_linear_reg(x,y,d,m,pi,Dn,count,e,s)
C = zeros(d,d);
D = zeros(d,1);
for j = 1:m
        temp1 = x{j}' * x{j};
        temp2 = x{j}' * y{j};
        C = C + temp1;
        D = D + temp2;
end
xop = C \ D;

for p = 1:count
    for n = 1:m
        temp3(n) = pi(n) * 1/2 * norm((x{n} * xop  - y{n}),2)^2 / Dn(n);
    end
    Fop(p,1) = sum(temp3);
end

% x1 = rand(id,1);
% f = @(z)sf(z,x,y,m,pi,Dn);
% options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton');
% [xop,F] = fminunc(f,x1,options);
% %         options = optimoptions('fmincon','Display','off');
% %         prox = fmincon(f,x0,[],[],[],[],[],[],[],options);
% for p = 1:count
%     Fop(p,1) = F;
% end
% end
% 
% function Sumf = sf(z,x,y,m,pi,Dn)
% Sumf = 0;
% for j = 1:m
%     fj = 1/2 * pi(j) * norm((x{j} * z - y{j}),2)^2 / Dn(j);
%     Sumf = Sumf + fj;
% end
% return

end