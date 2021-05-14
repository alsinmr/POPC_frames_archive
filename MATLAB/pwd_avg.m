function [ varargout ] = pwd_avg( q,sym )
%PWD_AVG Generates a powder average with quality quality q (1 to 12). 
%According to JCP 59 (8) 3992 (1973) (copied from maer gamma programs).
%
%   pwd = pwd_avg(q);
%
%   or
%
%   [alpha beta gamma weight] = pwd_avg(q);
%
%   Alternatively, one may use the kugelprogram from Jens Thoms Toerring,
%   which only provides alpha and beta averaging. This is triggered if the
%   symmetry is provided (Ci, C2h, D2h, D4h), in which case a powder
%   average over either a hemi-sphere, a quarter-sphere, eight-sphere, or
%   16th sphere is calculated. In this case, q indicates the number of
%   contours for the powder average used
%
%   pwd = pwd_avg(q,sym)
%
%   Finally, one may produce only beta angles by specifying the second
%   argument as 'beta' and the first argument as the number of desired
%   angles
%
%
% A. Smith
if nargin==2&&strcmp(sym,'beta')
    alpha=zeros(q,1);
    beta=linspace(0,pi/2,q+1)';
    beta=beta(2:end);
    gamma=zeros(q,1);
    weight=sin(beta);
    weight=weight/sum(weight);
    if nargout==1
        varargout{1}.alpha=alpha;
        varargout{1}.beta=beta;
        varargout{1}.gamma=gamma;
        varargout{1}.weight=weight;
        varargout{1}.N=q;
    else
        varargout{1}=alpha;
        varargout{2}=beta;
        varargout{3}=gamma;
        varargout{4}=weight;
    end
elseif nargin==2

    switch sym

        case {'Ci'}
            SymPhi = 1;
        case {'C2h'}
            SymPhi = 2;
        case {'D2h'}
            SymPhi = 4;
        case {'D4h'}
            SymPhi = 8;
        otherwise
            disp('Symmetry not recognized, use Ci')
            SymPhi =1;
    end
   


    np = ceil(q*sin((0.5:q - 0.5)*pi/(2*q)))/SymPhi;
    nF = sum(np);
    
    alpha=zeros(nF*4,1);
    beta=zeros(nF*4,1);
    gamma=zeros(nF*4,1);
    
    theta_j = 0;
    count = 1;

    for j = 1 : q,
        dtheta = acos(cos(theta_j) - np(j)/nF ) - theta_j;
        beta( count:count+4*np( j ) - 1) = theta_j + dtheta/2;
        dphi = pi/(2*SymPhi*np(j));
        alpha(count:count+4*np(j) - 1) = 0.5*dphi:dphi:(4*np(j) - 0.5)*dphi;
        count = count + 4*np(j);
        theta_j = theta_j + dtheta;
    end;
    if nargout==1
        varargout{1}.alpha=alpha;
        varargout{1}.beta=beta;
        varargout{1}.gamma=gamma;
        varargout{1}.weight=ones(nF*4,1)/nF/4;
        varargout{1}.N=nF*4;
    end
else
    value1=[2 50 100 144 200 300 538 1154 3000 5000 7000 10000];
    value2=[1 7 27 11 29 37 55 107 637 1197 1083 1759];
    value3=[1 11 41 53 79 61 229 271 933 1715 1787 3763];

    count=1:(value1(q)-1);

    alpha=2*pi*mod(value2(q)*count,value1(q))/value1(q);
    beta=pi*count/value1(q);
    gamma=2*pi*mod(value3(q)*count,value1(q))/value1(q);

    weight=sin(beta);
    weight=weight/sum(weight);

    if nargout==1
        varargout{1}.alpha=alpha';
        varargout{1}.beta=beta';
        varargout{1}.gamma=gamma';
        varargout{1}.weight=weight';
        varargout{1}.N=length(count);
    else
        varargout{1}=alpha';
        varargout{2}=beta';
        varargout{3}=gamma';
        varargout{4}=weight';
    end
end

end

