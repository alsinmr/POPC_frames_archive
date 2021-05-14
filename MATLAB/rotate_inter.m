function [ A ] = rotate_inter( A0,pwd,varargin )
%ROTATE_INT Input is a rank 2 interaction in its principle axis system 
%(PAS) by given by its anisotropy and the asymmetry (A0=[delta eta]), a 
%structure containing a powder average (elements alpha,beta,gamma), and 
%optionally, euler angles for the interaction (more than one set may be 
%given, in which case they will be executed in order, starting from the 
%principle axis system (PAS)), and the rotor angle (if omitted, assumed to 
%be acos(sqrt(1/3))). Returns a matrix containing constants such that for 
%the kth element of the powder average, one may write the time dependent 
%Hamiltonian for the interaction in the rotating frame (keeping only the 
%T_(2,0) component of the interaction) as:
%
%  H = (conj(A(k,3))*exp(2*1i*omega_r*t)+conj(A(k,2))*exp(1i*omega_r*t)+...
%         A(k,1)+A(k,2)*exp(-1i*omega_r*t)+A(k,3)*exp(-2*1i*omega_r*t))*...
%         T_(2,0)*sqrt(3/2)
%
%Then, the term T_{2,0}*sqrt(3/2) depends on the interaction, for example:
%   Heteronuclear dipole:
%       T_(2,0)*sqrt(3/2) = Iz*Sz
%   Homonuclear dipole:
%       T_(2,0)*sqrt(3/2) = (I1z*I2z-1/2*(I1x*I2x+I1y*I2y))
%   Chemical shift anisotropy:
%       T_(2,0)*sqrt(3/2) = Iz
%
%We include multiplication by a factor of sqrt(2/3) in the calculation to 
%try to reduce the number of constants required when constructing the 
%Hamiltonians. This is to makes things easier for sloppy programmers and 
%drive the careful ones nuts. One may choose to not perform the final 
%tilting from the rotor frame to the lab frame (therefore omitting the 
%scaling by d2(theta_r,m,0), see d2.m)
%
%Then, the function call looks like the following:
%
%   A = rotate_inter( A0,pwd,euler1,euler2,...,theta_r )
%
%The euler rotations may be omitted if all interactions in the Hamiltonian
%are co-linear:
%
%   A = rotate_inter( A0,pwd,theta_r)
%
%And furthermore, the specification of the rotor angle is only necessary if
%spinning off the magic angle (defaults to acos(sqrt(1/3)))
%
%   A = rotate_inter( A0,pwd) 
%       or
%   A = rotate_inter( A0,pwd,euler1,euler2,...)
%
%The form of the arguments are
%
%   A0=[delta, eta]    (delta is the anisotropy, eta is the asymmetry)
%
%   pwd: Structure with fields alpha,beta,gamma (radians)
%
%   euler=[alpha beta gamma] (radians)
%
%   theta_r: Rotor angle (radians). Note: to omit rotation from rotor frame
%            to lab frame, set to NaN.
%
%   A. Smith, Mar. 2015


%% Determine if theta_r is given, and how many sets of euler rotations
% In the following section, we determine what arguments were given. The
% rotor angle and initial frame transformations are both optional, and are
% differentiated by the size of the argument (frame transformation is a
% 3-element vector, and rotor angle is a single number.
if nargin>2&&numel(varargin{end})==1
    theta_r=varargin{end};
    euler=varargin(1:end-1);
    nea=nargin-3;
elseif nargin>2
    euler=varargin(1:end);
    nea=nargin-2;
    theta_r=acos(sqrt(1/3));
else
    euler={[0 0 0]};
    theta_r=acos(sqrt(1/3));
    nea=1;
end
    
if isempty(euler)
    euler={[0 0 0]};
    nea=1;
end
    
%% Calculate tensor in the principle axis system. 
PAS2=zeros(1,3);            %Pre-allocate (note that in PAS, n=1 component always =0)
PAS2(3)=-0.5*A0(1)*A0(2);   %n=2 component (same as n=-2 component)
PAS2(1)=sqrt(3/2)*A0(1);    %n=0 component

%% Set default rotor angle if not given
if not(exist('theta_r','var'))||isempty(theta_r)
    theta_r=acos(sqrt(1/3));
end

%% Rotate into the molecular frame
MOL2=zeros(1,5);    %Pre-allocate the tensor for the molecular frame

for k=-2:2          %Calculate the k=-2,1,0,1,2 components in the molecuar frame
    index=k+3;
    for j=[-2 0 2]  %Calculate parts from the j=-2,0,2 components (n=+/- 1 PAS component is zero)
        MOL2(index)=MOL2(index)+...
            exp(-1i*(euler{1}(1)*j+euler{1}(3)*k))*d2(euler{1}(2),j,k)*PAS2(abs(j)+1);
    end
     %A2_(2,k)=sum_over_j (D_(j,k)(alpha,beta,gamma)*A1_(2,j)
     %  where
     %  D_(j,k)(alpha,beta,gamma)=exp(-1i*k*gamma)*d2(beta,j,k)*exp(-1i*j*alpha))
end

if nea>1            %If extra euler angles are specified, perform additional transformations
    for kk=2:nea
        temp=MOL2;  %Store current MOL2 in temp, perform roations as above
        MOL2=zeros(1,5);
        for k=-2:2
            index=k+3;
            for j=-2:2  %Since we no longer start from PAS, must consider all components j=-2,1,0,1,2
                MOL2(index)=MOL2(index)+...
                    exp(-1i*(euler{kk}(1)*j+euler{kk}(3)*k))*d2(euler{kk}(2),j,k)*temp(j+3);
            end
        end
    end
end

%% Rotate into the lab frame
A=zeros(length(pwd.alpha),3);       %Pre-allocate output vector. #Rows is size of the powder average, and 3 columns for n=0,1,2
alpha=pwd.alpha(:);                 %Make sure angle vectors are all columns
beta=pwd.beta(:);
gamma=pwd.gamma(:);

for k=0:2           %Calculate the 3 components, k=0,1,2 in the lab frame
    % Sum up components
    index=k+1;      %Zero index not allowed in matlab, so shift storage to 1,2,3 locations
    for j=-2:2      %Sum over components, transform into the rotor frame
        A(:,index)=A(:,index)+...
            exp(-1i*(alpha*j+gamma*k)).*d2(beta,j,k)*MOL2(j+3);
    end
    % Do final scaling, including rotation into lab frame.
    if isnan(theta_r)
        A(:,index)=A(:,index)*2/sqrt(6);    %If leaving in rotor frame, don't scale by d2. Scale by sqrt(2/3)
    else
        A(:,index)=A(:,index)*2/sqrt(6)*d2(theta_r,k,0);    %To go to the lab frame, scale by sqrt(2/3) and by d2(theta_r,k,0)
    end
end


         
end

