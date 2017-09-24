clc;
clear;

ratio = 0.80;
n1 = 100;
n2 = 100;
n3 = round(n1*n2*ratio);

r = 10;
Xl = randn(n1,r);
Xr = randn(n2,r);
Xreal = Xl*Xr';


p = randperm(n1*n2,n3);
A = sparse(n3, n1*n2);
for i=1:n3
    A(i, p(i)) = 1;
end
A = sparse(A);

b = A * Xreal(:); %+ randn(n3,1);

[Xopt, err] = ADMMdp(A,b,n1,n2);
plot(err);





err = norm(Xreal-Xopt,'fro')/norm(Xreal,'fro')






