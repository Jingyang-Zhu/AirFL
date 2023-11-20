function obj = FedAvg_errorfree(x,y,lr,m,id,od,pi,iter,Fop,count,epoch,batchsize)
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
            g(:,:,n) = gradientchange(theta,x{n},y{n},epoch(n),lr/(1+0.0075*t),od(n),batchsize);
%             g(:,:,n) = gradientchange(theta,x{n},y{n},epoch(n),lr,od(n),batchsize);

%             delta(:,:,n) = pi(n) * gradientchange(theta,x{n},y{n},epoch(n),lr/(1+0.002*t),od(n),batchsize);
            delta(:,:,n) = pi(n) * g(:,:,n);
        end
        theta = theta + sum(delta,3);
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