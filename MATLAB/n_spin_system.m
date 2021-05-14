function [ varargout ] = n_spin_system(varargin)
%N_SPIN_SYSTEM This function produces the spin matrices for a spin system
%of arbitrary size. The number of spins in the spin system will be
%determined by the number of output arguments. The input arguments are 
%the spin quantum numbers of the individual spins. One does not need to 
%specify all spin quantum numbers, but those that are unspecified will be
%assumed to be spin 1/2. Spin numbers may be listed as multiple inputs or
%as a single vector.
%
%Alternatively, one may use only one output, and list all spin quantum
%numbers,in which case all spin matrices will be put out in one cell. In
%this case, the number of spins is equal to the number of inputs
%
%   [S1 S2 S3 ...] = n_spin_system(1,3/2,2,....)
%   
%   or S = n_spin_system(1,3/2,2,...)
%
%   A. Smith, 2010

if nargin~=0&&ischar(varargin{end})&&strcmpi(varargin{end},'sparse')
    issparse=true;
    temp=cell(1,nargin-1);
    for k=1:nargin-1
        temp{k}=varargin{k};
    end
else
    temp=varargin;
    issparse=false;
end

if nargout==1&&nargin==0
    spins=1/2;
    nspins=1;
elseif nargout==1
    spins=cell2mat(temp);
    nspins=length(spins);    
else    
    nspins=nargout;
    spins=cell2mat(temp);
    spins=[spins 1/2*ones(1,nargout-length(spins))];
end
    
mat_sizes=2*spins+1;

if prod(mat_sizes)^2*nspins>5e12
    error('A system this large is likely to exceed the memory')
end

if prod(mat_sizes)^2*nspins>1.5e6||issparse
    for k=1:nspins
        out{k}.x=kron(speye(prod(mat_sizes(1:k-1))),kron(sparse(Jx(spins(k))),speye(prod(mat_sizes(k+1:end)))));
        out{k}.y=kron(speye(prod(mat_sizes(1:k-1))),kron(sparse(Jy(spins(k))),speye(prod(mat_sizes(k+1:end)))));
        out{k}.z=kron(speye(prod(mat_sizes(1:k-1))),kron(sparse(Jz(spins(k))),speye(prod(mat_sizes(k+1:end)))));
        out{k}.p=out{k}.x+1i*out{k}.y;
        out{k}.m=out{k}.x-1i*out{k}.y;
        if spins(k)==1/2
            out{k}.alpha=speye(prod(mat_sizes))/2+out{k}.z;
            out{k}.beta=speye(prod(mat_sizes))/2-out{k}.z;
        end
    end

else

    for k=1:nspins
        out{k}.x=kron(eye(prod(mat_sizes(1:k-1))),kron(Jx(spins(k)),eye(prod(mat_sizes(k+1:end)))));
        out{k}.y=kron(eye(prod(mat_sizes(1:k-1))),kron(Jy(spins(k)),eye(prod(mat_sizes(k+1:end)))));
        out{k}.z=kron(eye(prod(mat_sizes(1:k-1))),kron(Jz(spins(k)),eye(prod(mat_sizes(k+1:end)))));
        out{k}.p=out{k}.x+1i*out{k}.y;
        out{k}.m=out{k}.x-1i*out{k}.y;
        if spins(k)==1/2
            out{k}.alpha=eye(prod(mat_sizes))/2+out{k}.z;
            out{k}.beta=eye(prod(mat_sizes))/2-out{k}.z;
        end
    end
    
end
    

if nargout==1&&nspins~=1
    varargout{1}=out;
else
    varargout=out;
end
    

end
%% Subfunction Jm
function Out = Jm(j)
% General spin operator J_minus
% Input: Jm(j) with j = 1/2, 1, 3/2, ....
% With no input argument the j = 1/2 operator is calculated
% last change: tm, 16.07.03

if nargin == 0
	
	j = 1/2;
	
end

Mult = round(2*j+1);
Out = zeros(Mult,Mult);

for k =-j+1:j
	
	Out(k+j+1,k+j)= sqrt(j*(j+1) - k*(k-1));
	
end

end

%% Subfunction Jp

function Out = Jp(j)
% General spin operator J_plus
% Input: Jp(j) with j = 1/2, 1, 3/2, ....
% With no input argument the j = 1/2 operator is calculated
% last change: tm, 16.07.03

if nargin == 0
	
	j = 1/2;
	
end

Mult = round(2*j+1);
Out = zeros(Mult,Mult);

for k =-j:j-1
	
	Out(k+j+1,k+j+2)= sqrt(j*(j+1) - k*(k+1));
	
end

end

%% Subfunction Jx

function Out = Jx(j)
% General spin operator Jx
% Input: Jx(j) with j = 1/2, 1, 3/2, ....
% With no input argument the j = 1/2 operator is calculated
% last change: tm, 16.07.03

if nargin == 0
	
	j = 1/2;
	
end

Out = 0.5*(Jp(j)+Jm(j));

end

%% Subfunction Jy

function Out = Jy(j)
% General spin operator Jy
% Input: Jx(j) with j = 1/2, 1, 3/2, ....
% With no input argument the j = 1/2 operator is calculated
% last change: tm, 16.07.03

if nargin == 0
	
	j = 1/2;
	
end

Out = -0.5*1i*(Jp(j)-Jm(j));


end

%% Subfunction Jz

function Out = Jz(j)
% General spin operator Jz
% Input: Jx(j) with j = 1/2, 1, 3/2, ....
% With no input argument the j = 1/2 operator is calculated
% last change: tm, 16.07.03

if nargin == 0
	
	j = 1/2;
	
end

Mult = round(2*j+1);
Out = zeros(Mult,Mult);

for k =-j:j
	
	Out(k+j+1,k+j+1)= - k;
	
end

                                                                                                                                                                                                                                                                                                                                                                                       

end

