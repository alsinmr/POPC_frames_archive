% DIPSHIFT simulation, including explicit inclusion of multiple 1H, and
% frequency switched Lee-Goldburg decoupling
% Note that we do not include any chemical shift offset on the 13C, and 
% therefore do not explicitely calculate evolution beyond the H–C
% recoupling period.

S=0.2;  %Order parameter

dHC0=43e3;  %See Ferreira et al. 2013, Phys. Chem. Chem. Phys. 15, 1976
dHC=dHC0*S; %Residual coupling

%% Experimental parameters
vr=2e3;    %Spinning frequency (Hz)

v1=65e3;                    %Lee-goldburg decoupling strength (Hz)


np=33;                      %Number of experimental points in rotor period
nt=16;                      %Number of time points per exp. point (simulation resolution)

nr=2;                       %Number of rotor periods to loop DIPSHIFT

%% Set up time axes
dt=1/vr/(np-1);             %Time step for recorded points
t=0:dt:(np-1)*dt;           %Time axis
dt0=dt/nt;                  %Time step for calculation (higher resolution than dt)
t0=0:dt0:(nt-1)*dt0;        %Time axis for calculation (nt points)

theta_m=acos(sqrt(1/3));    %Magic angle
voff=v1/tan(theta_m);       %Offset for Lee-Goldburg decoupling

veff=sqrt(v1^2+voff^2);     %Effective field due to Lee-Goldburg decoupling

%Here we adjust veff slightly so that our time step falls exactly when we 
%should be switching the phase of the decoupling
veff=1/(ceil((1/veff)/dt0)*dt0);    

nc=1/veff/dt0;  %Number of time points between phase switches

v1=veff*sin(theta_m);   %We need to re-adjust v1 and voff to be consistent with veff
%Note, in the initial settings, we took v1 to be 65 kHz, here it has been
%adjusted to 64.3 kHz, a minor change.
voff=veff*cos(theta_m);

nH=2;                       %Number of protons to include in the simulation

%% Generate the spin matrics
S=n_spin_system(1/2*ones(1,1+nH));  %Generate spin matrices
rho0=S{1}.x;                        %Initial state of the system (x magnetization on 13C)
detect=rho0';                       %Detection operator (again, x magnetization of 13C)
norm=1/trace(detect*rho0);          %Normalization, so that initial value is 1

%% Setup the powder average
pwd=pwd_avg(5);                     %Powder average (increase for higher quality)
    
AHC=cell(nH,1);                     %Cell for storage of rotating components of H–C dipole couplings
for k=1:nH
    euler=[0 109.5 k*120];          %Assume tetrahedral geometry
%     euler=[0,0,0];
    AHC{k}=rotate_inter([dHC 0],pwd,euler*pi/180);  %Get components for all elements of powder averaged
end

AHH=rotate_inter([20e3 0],pwd,[0,90,0]*pi/180);      % Add a 1H-1H interaction for testing
% This should have no affect, given the homonuclear decoupling, but we can
% verify by setting this to have a non-zero value. The first entry (20
% kHz), is the anisotropy of the H–H coupling (bII=10 kHz coupling).

I0=cell(pwd.N,1);           

%% Loop over powder average
parfor k=1:pwd.N    %Loop over all elements of the powder average
    % Store matrices for each part of the rotor period with attenuated
    % coupling
    Uon=cell(np-1,1);   %Propagators for each sub-division of the rotor period 

    %% Build up the Hamiltonian, separated by rotating component

    Hr=cell(3,1);   %Components of Hamiltonian, for non-rotation, 1xvr rotation and 2xvr rotation components
    for n=1:3
        Hr{n}=zeros(size(S{1}.z));
        for m=1:nH
            Hr{n}=Hr{n}+AHC{m}(k,n)*S{1}.z*S{1+m}.z;    %Heternuclear coupling
            if n==1
                Hr{n}=Hr{n}+v1*S{1+m}.x+voff*S{1+m}.z;  %Applied field and offset (non-rotating)
            end
        end
        if nH>1 %Add a coupling between the first and second proton.
            Hr{n}=Hr{n}+AHH(k,n)*(-1*S{2}.x*S{3}.x-S{2}.y*S{3}.y+2*S{2}.z*S{3}.z)/2;
        end
    end

    %% Build propagators for each part of rotor period (nt parts)
    for m=1:np-1
        Uon{m}=eye(size(rho0));
        for n=1:nt
            H=exp(1i*4*pi*vr*(t(m)+t0(n)))*conj(Hr{3})+...  	%-2x rotating component
                exp(1i*2*pi*vr*(t(m)+t0(n)))*conj(Hr{2})+...    %-1x rotating component
                Hr{1}*(-1)^(mod((m-1)*nt+n,2*nc)>nc)+...        %non-rotating component
                exp(-1i*2*pi*vr*(t(m)+t0(n)))*Hr{2}+...         %1x rotating component
                exp(-1i*4*pi*vr*(t(m)+t0(n)))*Hr{3};            %2x rotating component
            Uon{m}=expm(-1i*2*pi*H*dt0)*Uon{m};                 %Propagate the Hamiltonian
        end
    end

    %% Calculate signal
    I0{k}=zeros(np,1);      %Store the signal for this element of the powder average
%     rho=rho0;
    U=eye(size(Uon{1}));    %Total propagator for 1 rotor period
    Unr=U;                  %Propagator for multiple rotor periods (if nr>1)
    for m=1:np              %Perform for each time point
        I0{k}(m)=trace((Unr*rho0/Unr)*detect)*norm; %Get current magnetization
        if m<np
%             rho=Uon{m}*rho/Uon{m};
            U=Uon{m}*U;     %Now, propagate for dt
            Unr=U^nr;       %Calculate net effect over multiple rotor periods
        end
    end
end

%% Sum up signal from all elements of the powder average
I=zeros(np,1);
for k=1:pwd.N
    I=I+I0{k}*pwd.weight(k);
end

%% Plot the result
figure(1)
hold all
plot(t*vr,I)
axis([0 1 min([0;I]) 1])



