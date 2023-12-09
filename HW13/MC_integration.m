function MC_integration()


n=10^4;
sigma=1.098699;
c= 2.2039;
Z=1.973732;
f=@(x)exp(-x^4+2*x^2-1);
g=@(x)1/sqrt(2*pi*sigma^2)*exp(-x^2/(2*sigma^2));
nbins=20;
X=linspace(-5,5,5*10^3);
exact=arrayfun(f,X);

sample=[];
for i=1:n
    z=normrnd(0,sigma);
    u=rand;
    if u<=f(z)/(c*g(z))
        sample=[sample,z]
    end
end

x_abs=abs(sample);
expectation=mean(x_abs);
fprintf('%f',expectation);
end