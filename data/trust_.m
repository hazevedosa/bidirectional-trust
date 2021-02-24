function t = trust_(l, u, b, p)
    tt = 1 - 1 ./ (b .* (u - l)) .* log((1 + exp(b .* (p - l)))./(1 + exp(b .* (p - u))));
    t = min(max(tt, 0), 1);
end