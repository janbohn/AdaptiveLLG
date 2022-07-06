# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:38:27 2021

@author: Lukas
"""

from __future__ import division
from dolfin import *
import numpy as np
#from fenics import *
import matplotlib.pyplot as plt
#from maxwellrt0 import *
#from bdfllg_func import *
#set_log_level(31)
from scipy.special import comb 

from fenics_error_estimation import mark

#initial, end time and tau0
t = 0 
T=0.5
tau = 0.001


p= 2 # BDF-Ordnung Zeit: only p=2
r= 1  # Degree fem space 

# Time stepping paramters
FS = 5/6 #safety factor
FL=0.1 #lower threshold
FU = 10.0 # upper threshold
Rtol = 1e-3 # tolerance time
taumin=1e-7 # minimal step size
timeestimator=2 # time estimator norm



# Space parameters
theta=0.5 # if dörfler marking is chosen via adaptmodus=1
spacetol=1.0
lowfac=0.5 *0.01
higfac=2 *0.01
coarsenratio= 0.25 # percentage of marked elements until coarsening happens
spaceestimator= 1  # 1 is gradient recovery 
refinemode=2 # refine during timestepping 2 is new stencil with L2 norm, 1 is old stencil with weighted H1norm
adaptmodus=2 # adapt meshes after coarsening:  via dörfler (1) or each error smaller tol (2)
coarsefact= 4**r # the bigger this factor, the coarser is the coarse mesh (refines unitl tol*coarsefact)



dorefinetime=False #True #
dorefinespace=False #True #
tau0=0.001
meshi0=UnitSquareMesh(50, 50) #predefined initial meshe

h0 = 5  # coarse initial mesh

prell=2 # prerefinement 
maxll= 10-prell # maximum refinement 


#Paramters
alpha = 1.51
Ce = 0.25
Dmi= 0.0 #DMI 0== off
d=4 # d=4 is bulk DMI, d=3 is 3D curl (3D domain), d=2 is quasi 3D curl (2D domain)
anis=0.0 #Anisotropy
ani=Constant((0,0,1))  #0== off
#nMAG = 0 

#initial data
class MyExpression0(UserExpression):
    def eval(self, value, x):
        sqnx = (x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5);
        A = (1-2*sqnx**0.5)**4/16;
        if sqnx**0.5<=0.5:
            value[0] = 2*A*(x[0]-0.5)/(A*A + sqnx);
            value[1] = 2*A*(x[1]-0.5)/(A*A + sqnx);
            value[2] = (A*A - sqnx)/(A*A + sqnx);
        else:
            value[0] = 0
            value[1] = 0
            value[2] = -1     
    def value_shape(self):
        return (3,) 
minit=MyExpression0(degree=r+2) 
#minit = Expression(("0","0","-1"), degree = r)

# define Magnetic field
#class MyExpression0(UserExpression):
#    def eval(self, value, x):
#        a = -500*pow(16*x[0]*(1-x[0])*x[1]*(x[1]-1),10)*max(0,1-10*t/15);
#        b = 0;
#        c = -500*pow(16*x[0]*(1-x[0])*x[1]*(x[1]-1),10)-200;#
#
#        value[0] = a
#        value[1] = b
#        value[2] = c   
#    def value_shape(self):
#        return (3,)  

#H = interpolate(MyExpression0(degree=r),V3ref)
H = Constant((0,0,-100.1))


# auxiliary functions

def adaptmesh(mesh, minit, tolerance, modus):
    if modus== 1: 
        errspace=spacetol+1.0
        ll=0
        while (errspace>tolerance and ll<maxll ):
            ll=ll+1
            Pr3 = VectorElement('Lagrange', mesh.ufl_cell(), r, dim=3);
            V3 = FunctionSpace(mesh,Pr3)

            #estimate
            DG0 = FunctionSpace(mesh, "DG", 0)
            cell_residual2 = Function(DG0)
            cell_residual2.vector()[:]=estimspac(interpolate(minit,V3),[],[],[],[],[],spaceestimator) 
            errspace=np.linalg.norm(cell_residual2.vector()[:])

            #mark
            markers = mark.dorfler(cell_residual2, theta)

            #refine
            mesh = refine(mesh, markers, redistribute=True)
            print("Error for current mesh: ", errornorm(minit,interpolate(minit,V3),"H1"))
            if ll == prell:
                meshh0=mesh
    if modus ==2:
        acceptx=False
        ll=0
        meshh0=mesh
        while not acceptx:
            
            Pr3 = VectorElement('Lagrange', mesh.ufl_cell(), r, dim=3);
            V3 = FunctionSpace(mesh,Pr3)

            #estimate
            DG0 = FunctionSpace(mesh, "DG", 0)
            cell_residual2 = Function(DG0)
            cell_residual2.vector()[:]=estimspac(interpolate(minit,V3),[],[],[],[],[],spaceestimator) 
            errspace=np.linalg.norm(cell_residual2.vector()[:])
            
            if refinemode ==1:
                if errspace < tolerance:
                    acceptx=True
                    if errspace < tolerance/2: 
                        docoarse= True
                else: 
                    markers = mark.dorfler(cell_residual2, theta)
                    mesh = refine(mesh, markers, redistribute=True)
                    ll=ll+1
                    # falls weniger als theta halbe markiert sind, dann bitte coarsenen oder sowas
            if refinemode ==2:
                percor, markers, dorefine = mark.threshold(cell_residual2, tolerance*lowfac, tolerance*higfac )
                if dorefine: 
                    mesh = refine(mesh, markers, redistribute=True)
                    ll=ll+1
                    print("Space NOT accepted: refine. Percentage to small: ", percor) 
                else: 
                    acceptx=True 
                    print("Space accepted: all errors smaller. Percentage to small: ", percor)
            if ll==prell:
                meshh0=mesh
    return mesh,meshh0,ll


def estimspac(m,mvecsto,v,lam,TT,tau,mode):
    meshh=m.function_space().mesh()
    DG0 = FunctionSpace(meshh, "DG", 0)
    Pr3 = VectorElement('Lagrange', meshh.ufl_cell(), r, dim=3);
    V3 = FunctionSpace(meshh,Pr3)
    if mode==1:
        w = TestFunction(DG0)
        d1= project(m.dx(0),V3)
        d2= project(m.dx(1),V3)
        residual2 = w*inner(m.dx(0)-d1,m.dx(0)-d1)*dx +w*inner(m.dx(1)-d2,m.dx(1)-d2)*dx 
    return np.asarray(assemble(residual2))**0.5 

def estimtim(mbdf,mvecsto,v,lam,TT,tau,estor):
    mm=interpolate(Expression(('1000','0','0'),degree=r),V3)
    if estor ==1:    
        if len(mvecsto)>2:
            mm.vector()[:] = (0.0*TT[-1] +2* tau) / 6 * ( (mbdf.vector()[:] - mvecsto[-1].vector()[:]) / tau - ( 1 + (tau/TT[-1]))*(mvecsto[-1].vector()[:]-mvecsto[-2].vector()[:]) / TT[-1] + tau/(TT[-1]*TT[-2])*(mvecsto[-2].vector()[:] - mvecsto[-3].vector()[:]) )
        else: # im ersten Schritt sind nur zwei Werte bekannt
            mm.vector()[:] = (0.0*TT[-1] + tau) / 6 * ( (mbdf.vector()[:] - mvecsto[-1].vector()[:]) / tau - ( 1 + (tau/TT[-1]))*(mvecsto[-1].vector()[:]-mvecsto[-2].vector()[:]) / TT[-1] + tau/(TT[-1]))
    
    if estor ==2:
        if len(mvecsto)>2:
            fact  = tau**2*(TT[-1]+tau)/6; 
            tau21 = tau+TT[-1];
            tau10 = TT[-1]+TT[-2];
            tau210= tau+TT[-1]+TT[-2];
            
            mvecsto[-3]=interpolate(mvec[-3],V3)
            
            mm.vector()[:] = 6*fact* mbdf.vector()[:]/(tau*tau21*tau210) - 6*fact* mvecsto[-1].vector()[:]/(tau*TT[-1]*tau10) + 6*fact*mvecsto[-2].vector()[:]/(TT[-1]*TT[-2]*tau21)- 6*fact*mvecsto[-3].vector()[:]/(TT[-2]*tau10*tau210); 
        else:
            return Rtol*0.99
            mm.vector()[:] =(0.0*TT[-1] + tau) / 6 * ( (mbdf.vector()[:] - mvecsto[-1].vector()[:]) / tau - ( 1 + (tau/TT[-1]))*(mvecsto[-1].vector()[:]-mvecsto[-2].vector()[:]) / TT[-1] + tau/(TT[-1]))
    return estimtimnorm(mm,estor)

def estimtimnorm(mm,estor):        
    if estor==1:
        err= 0.01*norm(mm,"H1")
    else:
        err= norm(mm,"L2")
    return err

def tdmicurl(v):
    
    return as_vector((-v[2].dx(0), -v[2].dx(1), v[0].dx(0)+v[1].dx(1)))

def tdcurl(v):
    
    return as_vector((-v[2].dx(1), -v[2].dx(0), v[1].dx(0)-v[0].dx(1)))




#prerefinement

print("Start Precomputation in Space") 

meshh = UnitSquareMesh(h0, h0)

if 'meshi0' in globals():
    meshh=meshi0
    meshh0=meshi0
    ll=0
    print("Use predefined mesh")
else:
    meshh,meshh0, ll= adaptmesh(meshh,minit,spacetol,adaptmodus)
print(" Prerefined mesh: ( ll=", prell," )")    
plot(meshh0)
plt.show()
print(" Initial mesh: ( ll=", ll," )" )
plot(meshh)
plt.show()



#spaces    
Pr = FiniteElement('P',meshh.ufl_cell(), r); #tetrahedron
Pr3 = VectorElement('Lagrange', meshh.ufl_cell(), r, dim=3);
#Pr2 = VectorElement('Lagrange', meshh.ufl_cell(), r, dim=2);
element = MixedElement([Pr3,Pr]);
VV = FunctionSpace(meshh, element)
V3 = FunctionSpace(meshh,Pr3)
#V2 = FunctionSpace(meshh,Pr2)
V = FunctionSpace(meshh,Pr)


print("Dimensions Of Freedom "+str(V3.dim())+ " refinement is " +str(ll))


#storage of results
mvec = []   
mvec.append(interpolate(minit,V3))
mvecsto = []   
mvecsto.append(interpolate(minit,V3))



print("Precomputation in time")
if ( 'tau0' in globals()):
    trk=2
    print("Use predefined tau")
    tau=tau0
else:
    trk=0
for zz in range(0,3-trk):
    delta= [1.0 , -1.0]
    gamma= [1.0]

    mhat = interpolate(Expression(['0','0','0'],degree=r),V3)
    mr = interpolate(Expression(['0','0','0'],degree=r),V3)

    mr.vector()[:]=mvecsto[-1].vector()[:]/float(delta[0])
    mhat=mr/sqrt(dot(mr,mr))


    # define variational problem
    (v,lam) = TrialFunctions(VV)
    (phi,mu) = TestFunctions(VV)

    # define LLG form
    dxr= dx(metadata={'quadrature_degree': 5})
    lhs = ((alpha*inner(v,phi)+inner(cross(mhat,v),phi)+tau/float(delta[0])*Ce*inner(nabla_grad(v),nabla_grad(phi)))*dxr
        + inner( dot(phi,mhat),lam)*dxr   + inner(dot(v,mhat),mu)*dxr)

    rhs = (-Ce*inner(nabla_grad(mr),nabla_grad(phi))+inner(H, phi))*dxr
    if Dmi>0:
        if d==3:
            lhs= lhs -Dmi/2*tau/delta[0]*inner(curl(v),phi)*dxr-Dmi/2*tau/delta[0]*inner(v,curl(phi))*dxr
            rhs=rhs + Dmi/2*inner(curl(mr),phi)*dxr+Dmi/2*inner(mr,curl(phi))*dxr
        elif d==2:
            lhs= lhs -Dmi/2*tau/delta[0]*inner(tdcurl(v),phi)*dxr-Dmi/2*tau/delta[0]*inner(v,tdcurl(phi))*dxr
            rhs=rhs + Dmi/2*inner(tdcurl(mr),phi)*dxr+Dmi/2*inner(mr,tdcurl(phi))*dxr
        else:
            lhs= lhs -Dmi/2*tau/delta[0]*inner(tdmicurl(v),phi)*dxr-Dmi/2*tau/delta[0]*inner(v,tdmicurl(phi))*dxr
            rhs=rhs + Dmi/2*inner(tdmicurl(mr),phi)*dxr+Dmi/2*inner(mr,tdmicurl(phi))*dxr
    if anis>0:
        rhs=rhs+ anis*inner(dot(ani,mhat)*ani,phi)*dxr
    # compute solution
    vlam = Function(VV)
    solve(lhs == rhs, vlam)#,solver_parameters={"linear_solver": "gmres"},form_compiler_parameters={"optimize": True})

    # update magnetization
    (v,lam) = vlam.split(deepcopy=True)

    #mbdf=interpolate(Expression(['0','0','0'],degree=r),V3) ;
    #mbdf.vector()[:]= mr.vector()[:] + tau/float(delta[0]) * v.vector()[:];


    # update  magnetization
    mvecsto.append(interpolate(Expression(['0','0','0'],degree=r),V3) );
    mvecsto[-1].vector()[:]=  mr.vector()[:] + tau/float(delta[0]) * v.vector()[:];    
    if zz ==1:
        d2y=interpolate(Expression(['0','0','0'],degree=r),V3) 
        d2y.vector()[:]= 1/tau**2*(mvecsto[-1].vector()[:]-2*mvecsto[-2].vector()[:]+mvecsto[-3].vector()[:])
        #L= np.linalg.norm(f(t_hilf[0],y_hilf[0])-f(t_hilf[1],y_hilf[1]))/np.linalg.norm(y_hilf[0]-y_hilf[1])
        errtime= estimtimnorm(d2y,timeestimator)
        tau= np.sqrt(Rtol*2/errtime)
        print("Two steps done. Errorestimate: " , errtime, " new step size ", tau  )
        mvecsto=[interpolate(minit,V3)]
    
mvec.append(interpolate(Expression(['0','0','0'],degree=r),V3) );
mvec[-1].vector()[:]= mvecsto[p-1].vector()[:];    


dimvec=[V3.dim()]
llvec=[ll] 
TT = [tau]
tvec=[0,tau]




#if not dorefinespace:
ll=ll-prell

# Time stepping  
jj=p-1 
t=tau

print("Timestepping")
while t < T:
    print("#####################")
    print("step " , jj , " at time ",str(t))
    acceptx=False
    acceptt=False
    while (not acceptx) or (not acceptt): 
        Pr = FiniteElement('P',meshh.ufl_cell(), r); #tetrahedron
        Pr3 = VectorElement('Lagrange', meshh.ufl_cell(), r, dim=3);
        #Pr2 = VectorElement('Lagrange', meshh.ufl_cell(), 1, dim=2);
        element = MixedElement([Pr3,Pr]);
        VV = FunctionSpace(meshh, element)
        V3 = FunctionSpace(meshh,Pr3)
        # proceed one step
        mvecsto[-2]= interpolate(mvec[-2],V3)
        mvecsto[-1]= interpolate(mvec[-1],V3)
        gamma = []
        delta = []

        if p==1: 
            delta= [1.0 , -1.0]
            gamma= [1.0]
        if p==2: 
            wn=tau/TT[-1]
            delta= [ (1+2*wn)/(1+wn),-(1+wn)**2/(1+wn) ,wn**2/(1+wn) ] 
            gamma= [1+tau/TT[-1] , -tau/TT[-1] ]
        #print(delta)
        #print(gamma)
        mhat = interpolate(Expression(['0','0','0'],degree=r),V3)
        mr = interpolate(Expression(['0','0','0'],degree=r),V3)
        for i in range(0,p):
            mhat.vector()[:]=mhat.vector()[:] + float(gamma[i])*mvecsto[-i-1].vector()[:]
            mr.vector()[:]=mr.vector()[:] - float(delta[i+1])*mvecsto[-(i+1)].vector()[:]

        mr.vector()[:]=mr.vector()[:]/float(delta[0])
        mhat=mhat/sqrt(dot(mhat,mhat))


        # define variational problem
        (v,lam) = TrialFunctions(VV)
        (phi,mu) = TestFunctions(VV)

        # define LLG form
        dxr= dx(metadata={'quadrature_degree': 5})
        lhs = ((alpha*inner(v,phi)+inner(cross(mhat,v),phi)+tau/float(delta[0])*Ce*inner(nabla_grad(v),nabla_grad(phi)))*dxr
            + inner( dot(phi,mhat),lam)*dxr   + inner(dot(v,mhat),mu)*dxr)

        rhs = (-Ce*inner(nabla_grad(mr),nabla_grad(phi))+inner(H, phi))*dxr
        if Dmi>0:
            if d==3:
                lhs= lhs -Dmi/2*tau/delta[0]*inner(curl(v),phi)*dxr-Dmi/2*tau/delta[0]*inner(v,curl(phi))*dxr
                rhs=rhs + Dmi/2*inner(curl(mr),phi)*dxr+Dmi/2*inner(mr,curl(phi))*dxr
            elif d==2:
                lhs= lhs -Dmi/2*tau/delta[0]*inner(tdcurl(v),phi)*dxr-Dmi/2*tau/delta[0]*inner(v,tdcurl(phi))*dxr
                rhs=rhs + Dmi/2*inner(tdcurl(mr),phi)*dxr+Dmi/2*inner(mr,tdcurl(phi))*dxr
            else:
                lhs= lhs -Dmi/2*tau/delta[0]*inner(tdmicurl(v),phi)*dxr-Dmi/2*tau/delta[0]*inner(v,tdmicurl(phi))*dxr
                rhs=rhs + Dmi/2*inner(tdmicurl(mr),phi)*dxr+Dmi/2*inner(mr,tdmicurl(phi))*dxr
        if anis>0:
            rhs=rhs+ anis*inner(dot(ani,mhat)*ani,phi)*dxr
        # compute solution
        vlam = Function(VV)
        solve(lhs == rhs, vlam)#,solver_parameters={"linear_solver": "gmres"},form_compiler_parameters={"optimize": True})

        # update magnetization
        (v,lam) = vlam.split(deepcopy=True)

        mbdf=interpolate(Expression(['0','0','0'],degree=r),V3) ;
        mbdf.vector()[:]= mr.vector()[:] + tau/float(delta[0]) * v.vector()[:];
            



        ###################################################################
        ##################### Error Estimation time #######################
        if not dorefinetime:
            acceptt=True
            print("No time refinement, go on !!!!!!")
            taunew=tau
        if not acceptt:    
            err= estimtim(mbdf,mvecsto,v,lam,TT,tau,timeestimator)
            z = 1/FS * (err/Rtol)**(1/(p+1))
            F = FS*(Rtol/err)**(1/(p+1))

            acceptt = np.max(z) < 1.2  
            if tau <1.001*taumin:
                acceptt=True
                print("taumin reached, go on !!!!!!")
            if F > FU: 
                F=FU
            if F < FL:
                F = FL    
            taunew = max(taumin,F*tau)
            
            #print("New Guess zum Zeitpunkt '{0}' : '{1}'".format(t,  tau))
            if acceptt:
                print("Time accepted, F is ", F)
            else: 
                print("Time NOT accepted, F is ", F)
                tau=taunew

        if not dorefinespace:
            acceptx=True
            print("No space refinement, go on !!!!!!")
        if not acceptx:
            #estimate
            DG0 = FunctionSpace(meshh, "DG", 0)
            cell_residual2 = Function(DG0)
            cell_residual2.vector()[:]=estimspac(mbdf,mvec,v,lam,TT,tau,spaceestimator) 
            errspace=np.linalg.norm(cell_residual2.vector()[:])
            
            if refinemode ==1:
                if errspace < spacetol:
                    acceptx=True
                    if errspace < spacetol/2: 
                        docoarse= True
                else: 
                    markers = mark.dorfler(cell_residual2, theta)
                    meshh = refine(meshh, markers, redistribute=True)
                    # falls weniger als theta halbe markiert sind, dann bitte coarsenen oder sowas
            if refinemode ==2:
                percor, markers, dorefine = mark.threshold(cell_residual2, spacetol*lowfac, spacetol*higfac )
                docoarse= percor>coarsenratio 
                print(percor)
                if dorefine: 
                    meshh = refine(meshh, markers, redistribute=True)
                    print("Space NOT accepted: refine. Percentage to small: ", percor) 
                else: 
                    acceptx=True 
                    print("Space accepted: all errors smaller. Percentage to small: ", percor) 
                    
    if not dorefinespace: 
        docoarse= False
    if docoarse:
        print(" COARSENING step") 
        meshh=meshh0
        meshh,notrelevant, ll= adaptmesh(meshh0,mbdf,spacetol*coarsefact,adaptmodus)
                
            
    mvec.append(interpolate(Expression(['0','0','0'],degree=r),V3) );
    mvec[-1].vector()[:]= mbdf.vector()[:];
    mvecsto.append(interpolate(Expression(['0','0','0'],degree=r),V3) );
    mvecsto[-1].vector()[:]= mbdf.vector()[:];               

    TT.append(tau)
    t+=tau
    tvec.append(t)
    jj=jj+1
    dimvec.append(V3.dim())
    llvec.append(ll+prell)
    tau=taunew
    print("Schritt vollendet! Guess für Zeitpunkt '{0}' ist '{1}'".format(t,  tau))
    print("Dimensions Of Freedom "+str(V3.dim())+" refinement is "+ str(ll+prell))
            
         
        
        

  
    
print("fertig")
##### mvec is solution
#for i in range(0,len(mvec)):
#    plot(mvec[i])


    
# Save solution to file in VTK format
vtkfile = File('plots/adaptiveLLG_aprox.pvd')
for i in range(0, len(mvec)-1):
    # compute u at time t by solving the PDE
    mvec[i+1].rename("u", "u") #see the QA reported below.
    vtkfile << mvec[i+1], tvec[i+1]
print("File saved")        
     
#vtkfile = File('plots/loesungbdf.pvd')
#for i in range(0, len(mvec)-1):
#    # compute u at time t by solving the PDE
#    mvecsto[i].rename("u", "u") #see the QA reported below.
#    vtkfile << mvecsto[i], tvec[i]
#print("File saved")     

print("länge = " + str(len(mvec)))
plt.semilogy(tvec[1:],TT)
plt.show()
plt.plot(tvec[1:],dimvec)
plt.show()
plt.plot(tvec[1:],llvec)
plt.show()
