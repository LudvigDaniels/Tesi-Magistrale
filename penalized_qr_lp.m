function [a_hat, beta_hat] = penalized_qr_lp(Xfull, Y, tau, lambda, weights)
% penalized_qr_lp: quantile regression penalizzata (L1 pesata) via linprog

[n_obs, p_full] = size(Xfull);
p = p_full;

nvar = 2*p + 2*n_obs;
idx_beta_pos = 1:p;
idx_beta_neg = p+1:2*p;
idx_rp       = 2*p+1:2*p+n_obs;
idx_rn       = 2*p+n_obs+1:2*p+2*n_obs;

Aeq = sparse(n_obs, nvar);
Aeq(:, idx_beta_pos) = Xfull;
Aeq(:, idx_beta_neg) = -Xfull;
Aeq(:, idx_rp)       = speye(n_obs);
Aeq(:, idx_rn)       = -speye(n_obs);
beq = Y;

f = zeros(nvar,1);
f(idx_beta_pos) = lambda .* weights(:);
f(idx_beta_neg) = lambda .* weights(:);
f(idx_rp) = tau;
f(idx_rn) = (1-tau);

lb = zeros(nvar,1);
options = optimoptions('linprog','Display','none','Algorithm','interior-point');
[z, ~, exitflag] = linprog(f, [], [], Aeq, beq, lb, [], options);
if exitflag <= 0
    warning('linprog non ottimale (exitflag=%d)', exitflag);
end

beta_pos = z(idx_beta_pos);
beta_neg = z(idx_beta_neg);
beta = beta_pos - beta_neg;

a_hat = beta(1);
beta_hat = beta;
end
