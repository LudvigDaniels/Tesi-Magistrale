function L = pinball_loss(residuals, tau)
% residuals = Y - Yhat
% pinball loss mean
u = residuals;
L = mean( u .* (u >= 0) * tau + (-u) .* (u < 0) * (1 - tau) );
end