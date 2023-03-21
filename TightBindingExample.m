function [A, B, O] = TightBindingExample(R,sx,sy, numCavitytype)
    %%
    % Load the cavity structs. Each struct has five properties: 
    % struct.Ex: x-component of electric field
    % struct.Ey: y-component of electric field
    % struct.Ez: z-component of electric field
    % struct.r : radius of defect hole
    % struct.w : complex resonant frequency of cavity
    
    centerCavity = load("/Users/bryn/Documents/ENPH 455/0r0760000.mat");
    wingCavity = load("/Users/bryn/Documents/ENPH 455/0r0780000.mat");
    pumpCavity = load("/Users/bryn/Documents/ENPH 455/0r0740000.mat");
    %signalCavity = load("C:\Users\Public\Single Defect Field Profiles\testFiles\0r0890000.mat");
    %idlerCavity = load("C:\Users\Public\Single Defect Field Profiles\testFiles\0r0660000.mat");
    
    a = 480e-9;     % lattice constant [m]
    r0 = 0.192e-6;  % default air hole radius [m]
    h = 0.384e-6;   % thickness of photonic crystal slab [m]
    n = 3.47334;    % index of refraction of PCS (Si @ 191 THz)
    s = 20;         % Number of grid points per lattice constant 
    dV = (a/s)^3;   % volume element 
    z = 8;          % height of computational volume in units of a
    
    % Size of tight-binding computational volume, in units of a.
    % sx = 25;
    % sy = 25;
    
    
    % make blocks of dielectric function
    
    block0 = makeBlock(a,r0,h,z,s,n);
    block1 = makeBlock(a,centerCavity.r,h,z,s,n);
    block2 = makeBlock(a,wingCavity.r,h,z,s,n);
    block3 = makeBlock(a,pumpCavity.r,h,z,s,n);
%     block4 = makeBlock(a,signalCavity.r,h,z,s,n);
%     block5 = makeBlock(a,idlerCavity.r,h,z,s,n);
%     
    blocks = cat(4,block0,block1,block2,block3);
    
    %% normalization
    blank = buildDielectric(zeros(sx,sy),block0); 
    
    % R-matrix for single defect
    Rq = zeros(sx,sy);
    Rq(ceil(sx/2),ceil(sy/2))=1;

    if (numCavitytype >= 1)
        [Ex1,Ey1,Ez1] = padMode(centerCavity.Ex,centerCavity.Ey,centerCavity.Ez,blank);
        epsilonq = buildDielectric(Rq,cat(4,block0,block1));
        A11 = sum((Ex1.*conj(Ex1) + Ey1.*conj(Ey1) + Ez1.*conj(Ez1)).*epsilonq.*dV,'all');
        Ex1 = Ex1./sqrt(A11);
        Ey1 = Ey1./sqrt(A11);
        Ez1 = Ez1./sqrt(A11);
        check1 = sum((Ex1.*conj(Ex1) + Ey1.*conj(Ey1) + Ez1.*conj(Ez1)).*epsilonq.*dV,'all');

    end

    if (numCavitytype >= 2)
        [Ex2,Ey2,Ez2] = padMode(wingCavity.Ex,wingCavity.Ey,wingCavity.Ez,blank);
        epsilonq = buildDielectric(Rq,cat(4,block0,block2));
        A22 = sum((Ex2.*conj(Ex2) + Ey2.*conj(Ey2) + Ez2.*conj(Ez2)).*epsilonq.*dV,'all');
        Ex2 = Ex2./sqrt(A22);
        Ey2 = Ey2./sqrt(A22);
        Ez2 = Ez2./sqrt(A22);
        check2 = sum((Ex2.*conj(Ex2) + Ey2.*conj(Ey2) + Ez2.*conj(Ez2)).*epsilonq.*dV,'all');
    end

    if (numCavitytype >= 3)
        check2 = sum((Ex2.*conj(Ex2) + Ey2.*conj(Ey2) + Ez2.*conj(Ez2)).*epsilonq.*dV,'all');
        [Ex3,Ey3,Ez3] = padMode(pumpCavity.Ex,pumpCavity.Ey,pumpCavity.Ez,blank);
        epsilonq = buildDielectric(Rq,cat(4,block0,block3));
        A33 = sum((Ex3.*conj(Ex3) + Ey3.*conj(Ey3) + Ez3.*conj(Ez3)).*epsilonq.*dV,'all');
        Ex3 = Ex3./sqrt(A33);
        Ey3 = Ey3./sqrt(A33);
        Ez3 = Ez3./sqrt(A33);
        check3 = sum((Ex3.*conj(Ex3) + Ey3.*conj(Ey3) + Ez3.*conj(Ez3)).*epsilonq.*dV,'all');
    end
    % 
    % [Ex4,Ey4,Ez4] = padMode(signalCavity.Ex,signalCavity.Ey,signalCavity.Ez,blank);
    % epsilonq = buildDielectric(Rq,cat(4,block0,block4));
    % A44 = sum((Ex4.*conj(Ex4) + Ey4.*conj(Ey4) + Ez4.*conj(Ez4)).*epsilonq.*dV,'all');
    % Ex4 = Ex4./sqrt(A44);
    % Ey4 = Ey4./sqrt(A44);
    % Ez4 = Ez4./sqrt(A44);
    % check4 = sum((Ex4.*conj(Ex4) + Ey4.*conj(Ey4) + Ez4.*conj(Ez4)).*epsilonq.*dV,'all');
    % 
    % [Ex5,Ey5,Ez5] = padMode(idlerCavity.Ex,idlerCavity.Ey,idlerCavity.Ez,blank);
    % epsilonq = buildDielectric(Rq,cat(4,block0,block5));
    % A55 = sum((Ex5.*conj(Ex5) + Ey5.*conj(Ey5) + Ez5.*conj(Ez5)).*epsilonq.*dV,'all');
    % Ex5 = Ex5./sqrt(A55);
    % Ey5 = Ey5./sqrt(A55);
    % Ez5 = Ez5./sqrt(A55);
    % check5 = sum((Ex5.*conj(Ex5) + Ey5.*conj(Ey5) + Ez5.*conj(Ez5)).*epsilonq.*dV,'all');
    
    % construct the dielectric function of the full system 
    epsilon = buildDielectric(R,blocks);
    

    if (numCavitytype == 1)
        [A,B,O]=TBMatrices(R,[centerCavity.w],cat(4,Ex1),cat(4,Ey1),cat(4,Ez1),blocks,dV,4);
    end
    % compute all required overlap and coupling integrals
    if (numCavitytype == 2)
        [A,B,O]=TBMatrices(R,[centerCavity.w,wingCavity.w],cat(4,Ex1,Ex2),cat(4,Ey1,Ey2),cat(4,Ez1,Ez2),blocks,dV,4);
    end
    if (numCavitytype == 3)
        [A,B,O]=TBMatrices(R,[centerCavity.w,wingCavity.w,pumpCavity.w],cat(4,Ex1,Ex2,Ex3),cat(4,Ey1,Ey2,Ey3),cat(4,Ez1,Ez2,Ez3),blocks,dV,4);
    end
    
    % solve the TB EVP
    %[V_q_nu,N_nu] = TB(A,B,O);
    
    % extract QM frequencies as list
    %N_nu = diag(N_nu);
    
   
end