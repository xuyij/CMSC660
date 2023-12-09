function Q2A_MC(d)


n=10^5;


sample=[];
X_array=rand(n,d)-1/2;
X_square=X_array.*X_array;
success=0;
for i=1:n
    if sum(X_square(i,:))<1
        success=success+1
    end
end


fprintf('%f',success/n);
end