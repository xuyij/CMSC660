function min_find()

fg=@(sigma)sqrt(2*pi*sigma^2)/1.973732*exp(-1/16*(4+1/(sigma)^2)^2 ...
    +(2+1/(2*sigma^2))*(4+1/(sigma^2))/4-1);
sigma_star = fminbnd(fg,0,10);
fprintf('%f',sigma_star);
end
