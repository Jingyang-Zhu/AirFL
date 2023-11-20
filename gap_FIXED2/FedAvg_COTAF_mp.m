function obj = FedAvg_COTAF_mp(x,y,lr,m,id,od,pi,iter,Fop,count,epoch,batchsize,SNR,h)
obj = zeros(count,iter-1);

for p = 1:count
    % initialize
    x_0 = zeros(id,1);
    temp1 = 0;
    theta = x_0;
    
    % iteration
    for t = 1:iter-1
        for n = 1:m
            temp1(n) = pi(n) * F(theta,x{n},y{n},od(n));
        end
        obj(p,t) = sum(temp1);
        fprintf('round %d, %dth iteration, the loss is %2.4e\n',p,t,obj(p,t) - Fop(p,1));
        for n = 1:m
%             delta(:,:,n) = pi(n) * gradientchange(theta,x{n},y{n},epoch(n),lr,od(n),batchsize);
            delta(:,:,n) = pi(n) * gradientchange(theta,x{n},y{n},epoch(n),lr/(1+0.004*t),od(n),batchsize);
            normdelta(n,1) = norm(delta(:,:,n));
        end
        P(p,t) = sqrt(id) * min(h(:,t)) / max(normdelta);
        noise = normrnd(0,sqrt(10^(-SNR/10)),id,1);
        theta = theta + sum(delta,3) + noise / P(p,t);
    end
end
return

% gradient change
function gc = gradientchange(theta,x,y,epoch,lr,od,batchsize)
%x0 = x;
tmp = theta;
for i = 1:epoch
    GDinfo(:,:,i) = G(theta,x,y,od,batchsize);
    theta = theta - lr * GDinfo(:,:,i);
end
gc = theta - tmp;
return

% gradient
function Gn = G(theta,x,y,od,batchsize)
idx = randsample(od,batchsize);
xx = x(idx,:);
yy = y(idx);
Gn = xx' * (xx * theta - yy) / batchsize;
return

% loss function
function Fn = F(theta,x,y,od)
Fn = 1 / 2 * norm((x * theta - y),2)^2 / od;
return