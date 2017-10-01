function [Xopt, err] = ADMMdp(A,b,n1,n2)

lambda = max(abs(A' * b)); % lambda_0 >= ||A*b||_inf regularization parameter
rho = 1; % penalty parameter for the violation of the linear constraint
beta = 0.5; % beta \in (0,1)
MAX_ITER = 200;

X = zeros(n1,n2);


best_lambda = lambda;
best_ds = inf;
for i=1:MAX_ITER
    lambda = lambda * beta;
    [X, err(i)] = solve(A,b,X,n1,n2,lambda,rho);
    
    ds = norm(A * X(:) - b);
    if ds < best_ds
        best_ds = ds;
        best_lambda = lambda;
    end

end
Xopt = solve(A,b,X,n1,n2,best_lambda,rho);
end

function [X,err] = solve(A,b,X0,n1,n2,lambda,rho)
% min rank(X)
% s.t. AX = b
% A is a linear operator (in R^{n_3 x n_1n_2}) and b in R^{n_3}
% X in R^{n_1 x n_2}
% \lambda is regularization parameter
% \rho is penalty parameter for the violation of the linear constraint.


alpha = 0.5;
% phi = @(x) alpha .* abs(x) ./ (1 + alpha .* abs(x));
% psi = @(x) phi(x) - alpha .* abs(x);
dpsi = @(x) sign(x) .* alpha ./ ( (alpha .* abs(x) + 1) .^ 2) - alpha .* sign(x);
% dpsi = @(x) sign(x);

MAX_ITER = 200;
TOL = 1e-4;

X = X0;
Z = zeros(n1,n2);
I = speye(n1*n2);
ata = A' * A;
atb = A' * b;
cga = ata + rho * I;

for i=1:MAX_ITER
    % solve W
    tau = lambda * alpha / rho;
    Y =  X + Z ./ rho;
    
    [U,S,V] = svd(Y);
    W = U * max(S - tau,0) * V';
    
    % Quasi-Newton's method
    [U,S,V] = svd(X);
    dPsi = U * dpsi(S) * V';


    cgb = atb - ata * X(:) - lambda * dPsi(:) + rho * (W(:) - X(:)) - Z(:);
    % dX = cga \ cgb;
    [dX, ~] = pcg(cga, cgb);
    
    dX = reshape(dX, n1, n2);
    
    err = norm(dX,'fro')/norm(X,'fro');
    if  err < TOL
        X = X + dX;
        break
    end
    X = X + dX;
    Z = Z - rho * (W - X);
end
end

