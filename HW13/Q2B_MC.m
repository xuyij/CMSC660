function Q2B_MC(d)


n=10^5;


sample=[];

X_array=normrnd(0,1,n,d);
success=0;
sample=[];


for i=1:size(X_array,1)
    U=rand
    sample=[sample;X_array(i,:)*U^(1/d)/norm(X_array(i,:))];
end
n_sample=size(sample,1);

for i=1:size(X_array,1)
    if max(abs(sample(i,:)))<1/2
        success=success+1;
    end
end



B_vol=pi^(d/2)/(d/2*gamma(d/2));


fprintf('%f',B_vol*success/n_sample);
end