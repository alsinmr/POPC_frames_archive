function [ out ] = d2( beta,mp,m )
%D2 Generates the wigner rotation matrix elements for the d_{m',m}^2
%   matrix, for rank 2 tensors. This is called with the angle beta, and 
%   returns the full 5x5 matrix, allowing a transformation from m' to m. 
%   Note that matrix ordering is given such that m and m' run from +2 to -2
%
%   dl=d2(beta)
%
%   dl=d2(beta,mp,m)
%
%   A. Smith, Nov. 2012.
%   alsi@nmr.phys.chem.ethz.ch

if nargin==1
    out=[((1+cos(beta))/2)^2 (1+cos(beta))/2*sin(beta) sqrt(3/8)*sin(beta)^2 (1-cos(beta))/2*sin(beta) ((1-cos(beta))/2)^2;
        -(1+cos(beta))/2*sin(beta) cos(beta)^2-(1-cos(beta))/2 sqrt(3/8)*sin(2*beta) (1+cos(beta))/2-cos(beta)^2 (1-cos(beta))/2*sin(beta);
        sqrt(3/8)*sin(beta)^2 -sqrt(3/8)*sin(2*beta) (3*cos(beta)^2-1)/2 sqrt(3/8)*sin(2*beta) sqrt(3/8)*sin(beta)^2;
        -(1-cos(beta))/2*sin(beta) -cos(beta)^2+(1+cos(beta))/2 -sqrt(3/8)*sin(2*beta) -(1-cos(beta))/2+cos(beta)^2 (1+cos(beta))/2*sin(beta);
        ((1-cos(beta))/2)^2 -(1-cos(beta))/2*sin(beta) sqrt(3/8)*sin(beta)^2 -(1+cos(beta))/2*sin(beta) ((1+cos(beta))/2)^2];
elseif nargin==3
    test=m+1i*mp;
    switch test
        case (2)+1i*(2)
            out=((1+cos(beta))/2).^2; 
        case (2)+1i*(1)
            out=(1+cos(beta))/2.*sin(beta);
        case (2)+1i*(0)
            out=sqrt(3/8)*sin(beta).^2;
        case (2)+1i*(-1)
            out=(1-cos(beta))/2.*sin(beta);
        case (2)+1i*(-2)
            out=((1-cos(beta))/2).^2;
        case (1)+1i*(2)
            out=-(1+cos(beta))/2.*sin(beta);
        case (1)+1i*(1)
            out=cos(beta).^2-(1-cos(beta))/2;
        case (1)+1i*(0)
            out=sqrt(3/8)*sin(2*beta);
        case (1)+1i*(-1)
            out=(1+cos(beta))/2-cos(beta).^2;
        case (1)+1i*(-2)
            out=(1-cos(beta))/2.*sin(beta);
        case (0)+1i*(2)
            out=sqrt(3/8)*sin(beta).^2;
        case (0)+1i*(1)
            out=-sqrt(3/8)*sin(2*beta);
        case (0)+1i*(0)
            out=(3*cos(beta).^2-1)/2;
        case (0)+1i*(-1)
            out=sqrt(3/8)*sin(2*beta);
        case (0)+1i*(-2)
            out=sqrt(3/8)*sin(beta).^2;
        case (-1)+1i*(2)
            out=-(1-cos(beta))/2.*sin(beta);
        case (-1)+1i*(1)
            out=-cos(beta).^2+(1+cos(beta))/2;
        case (-1)+1i*(0)
            out=-sqrt(3/8)*sin(2*beta);
        case (-1)+1i*(-1)
            out=-(1-cos(beta))/2+cos(beta).^2;
        case (-1)+1i*(-2)
            out=(1+cos(beta))/2.*sin(beta);
        case (-2)+1i*(2)
            out=((1-cos(beta))/2).^2;
        case (-2)+1i*(1)
            out=-(1-cos(beta))/2.*sin(beta);
        case (-2)+1i*(0)
            out=sqrt(3/8)*sin(beta).^2;
        case (-2)+1i*(-1)
            out=-(1+cos(beta))/2.*sin(beta);
        case (-2)+1i*(-2)
            out=((1+cos(beta))/2).^2;
    end
else
    error('d2 must have either 1 or 3 inputs')
end

end

