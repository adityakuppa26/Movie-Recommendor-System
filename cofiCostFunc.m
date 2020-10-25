function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

%        X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

sum=0;
for j=1:num_users
	for i=1:num_movies
		if(R(i,j)==1)
			sum = sum + (((X(i,:)*Theta(j,:)')-Y(i,j))^2);
		end
	end
end

sumu=0;
for j=1:num_users
    for k=1:num_features
        sumu = sumu + (Theta(j,k)^2);
    end
end

sumx=0;
for i=1:num_movies
    for k=1:num_features
        sumx = sumx + (X(i,k)^2);
    end
end
J= (sum/2)+((lambda/2)*(sumx+sumu));


for i=1:num_movies
	for k=1:num_features
		sum1=0;
		for j=1:num_users
			if(R(i,j)==1)
				sum1=sum1+((X(i,:)*Theta(j,:)')-Y(i,j))*Theta(j,k);
            
			end
        end
            sum1 = sum1+ (lambda*X(i,k));
	X_grad(i,k)=sum1;
	end
end


for j=1:num_users
	for k=1:num_features
		sum2=0;
		for i=1:num_movies
			if(R(i,j)==1)
				sum2=sum2+(((X(i,:)*Theta(j,:)')-	Y(i,j))*X(i,k));
			end
        end
        sum2=sum2+(lambda*Theta(j,k));
        Theta_grad(j,k)=sum2;
	end
end

grad = [X_grad(:); Theta_grad(:)];

end
