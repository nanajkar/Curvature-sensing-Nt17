#!/usr/bin/env python
#
# This code has been written by Ali Asghar HAKAMI ZANJANI at SDU, Odense, Oct. 2020.
#
# This code is free, and you can redistribute it or modify it for your research purpose.
#
# If you find this code useful for your research please cite [Hakami Zanjani et al., Biophysical Journal (2023),
# https://doi.org/10.1016/j.bpj.2023.04.002] if at all possible.
#
# In this code two-dimensional mean curvature profiles of the surfaces fitted on the upper and
# lower leaflets, and the surface passing through the center of a bilayer membrane,
# are calculated, and their time average is plotted as the output. Time serise of the dimensions of
# the simulation box, area of the fitted surface and projected surface on xy plane are written in log files.
#
# HOW TO USE THIS CODE:
# Using this code is very simple; You just need a structure (.gro), a trajectory (.xtc), and an index (.ndx) file.
#
# Make sure that the specified groups as the upper and lower leaflets, don't pass
# the z boundaries of the main periodic box during the trajectory.
#
# If there is a protein (or any other group) on the membrane and you are going to calculate
# the spatial average of the mean curvature of the membrane inside the area of a circle surrounding
# the projection of the protein on the membrane, during the time, first center the system about
# that group then use this code.

# Run MemCurv.py as:
# ./MemCurv.py structure.gro trajectory.xtc index.ndx
# (Please substitute "structure", "trajectory" and "index" with the names of your files.)
#
# WARNING: Running this code will delete and overwrite all previously generated .dat or .png files. in the
# current folder.
#
# Press any key to continue or ^C (Control+C) to terminate.
#
# Please select a group for upper monolayer, from the list (the list is created from the given index file)
#
# Please select a group for lowr monolayer, from the list (the list is created from the given index file)
#
# Please select a group for protein (or any other group) for the spatial average of the mean curvature
# beneath that group from the list or insert -1 if you don't want to specify any group:
#
# Please enter Nx: 2
# Please enter Ny: 2
# Nx and Ny are the upper limits of the partial sums for Fourier series in fitting step.
# Nx=Ny=2 gives a reasonable surface fitted on a flat bilayer POPC/POPS membrane with a size smaller than 20x20 nm^2.     
#
# Please enter the number of grids in x direction: 200

# Please enter the number of grids in y direction: 200 
# 200 grids in x and y directions, give suitable resolution.
#
# Index of the phosphorus atoms in the upper and lower leaflets can specify the upper and lower
# layers of the membrane.

# Index of different columns in output log.dat files:
#  0: frame number (frame)
#  1: x dimension of bilayer (Lx)
#  2: y dimension of bilayer (Ly)
#  3: z dimension of bilayer (Lz)
#  4: surface projection of bilayer onto xy-plane (Axy = Lx*Ly)
#  5: surface of bilayer (ASur)
#  6: x of protein (Prx)
#  7: y of protein (Pry)
#  8: z of protein (Prz)
#  9: radius of protein (PrR)
# 10: surface projection of protein onto xy-plane (APr = pi*PrR^2)
# 11: surface of bilayer beneath protein (ASurPr)
# 12: average of mean curvature over surface (Hm)
# 13: average of mean curvature over surface beneath protein (HPrm)

from scipy import stats
from MDAnalysis import *
import os
import sys
import numpy as np
import MDAnalysis as mda
import scipy.optimize as optimize
import csv
from scipy.interpolate import RegularGridInterpolator
################################################################################

def GetGroupList(IndexFile):
# Gets the name of the index file, reads the name of the different groups from the
# index file and returns it as a list of [group name, line in index]. 
   GroupList = []
   with open(IndexFile) as fp:
      line = fp.readline()
      cnt  = 1
      while line:
         if '[' in line:
            item = line.strip("[\n ]")
            GroupList.append((item, cnt))
         line = fp.readline()
         cnt += 1
   GroupList.append(('EndOfFile', cnt))
   fp.close()
   return GroupList
################################################################################

def GetGroupName(GroupList,GroupName):
# Gets the GroupList (output of  GetGroupList) and name of a group and asks user
# to select the id of the group from the list.
   print ("\nPlease select a group for {} from the list:".format(GroupName))
   cnt = 0
   for i in GroupList[: -1]:
      print("{}: {}".format(cnt, i[0]))
      cnt += 1
   return "Index of " + GroupName+': '
################################################################################

def GetGroupIndex(GrCode, GrList, IndexFile):
# Gets the code of a group, group list (output of GetGroupList) and the name of
# the index file and returns the index of all atoms of the group in MDA format. 
   GroupIndex = []
   BeginLine  = GrList[GrCode][1]
   EndLine    = GrList[GrCode+1][1]
   with open(IndexFile) as fp:
      line = fp.readline()
      cnt  = 1
      while cnt < EndLine:
         if cnt > BeginLine:
            GroupIndex += [int(i)-1 for i in line.split()]
         line = fp.readline()
         cnt += 1
   fp.close()
   return GroupIndex
################################################################################

def GetGroupIndexMda(GrCode, GrList, IndexFile): # Gets the code of a group and
# group list (output of GetGroupList) and the name of index file and returns the
# index of all atoms of the group in MDA format. 
   GroupIndex = []
   BeginLine  = GrList[GrCode][1]
   EndLine    = GrList[GrCode+1][1]
   with open(IndexFile) as fp:
      line = fp.readline()
      cnt  = 1
      while cnt < EndLine:
         if cnt > BeginLine:
            GroupIndex += [int(i)-1 for i in line.split()]
         line = fp.readline()
         cnt += 1
   fp.close()
   GroupIndexMda = "index"
   for i in range(len(GroupIndex)):
      GroupIndexMda += " "+str(GroupIndex[i])
   return GroupIndexMda
################################################################################

def Zfunc(data, *Amn):
   global Lx, Ly, m_x, n_y
   z = 0
   for m in range (-m_x,m_x+1):
      for n in range (-n_y,n_y+1):
         z += Amn[(m+m_x)*(2*n_y+1)+(n+n_y)] * (np.cos(2*np.pi*m*data[:,0]/Lx+2*np.pi*n*data[:,1]/Ly) + np.sin(2*np.pi*m*data[:,0]/Lx+2*np.pi*n*data[:,1]/Ly))
   return z
################################################################################
################################################################################
################################################################################
# Main 

gro  = sys.argv[1]
xtc  = sys.argv[2]
ndx  = sys.argv[3]
GrLs = GetGroupList(ndx)

print("\nWARNING: Running this code will delete and overwrite all previously generated png and dat files.")
print('\nPlease make sure that the specified groups as the upper and lower leaflets, do not pass the z boundaries of the main periodic box during the trajectory.')
print('\n************************************************************************************************')
input("\nPress any key to continue ... or ^C (Control+C) to terminate")

os.system('rm *.dat')
os.system('rm *.png')


UpLayer  = int(input(GetGroupName(GrLs, 'upper monolayer')))
LowLayer = int(input(GetGroupName(GrLs, 'lower monolayer')))
u = mda.Universe(gro, xtc)

print ("\n{} consists of {} frames.".format(xtc,len(u.trajectory)))

BeginFrame = int(input ("\nPlease enter the begin frame to analysis (0 <= Number <= {}): ".format(len(u.trajectory)-1)))
EndFrame   = int(input ("\nPlease enter the end frame to analysis ({} <= Number <= {}): ".format(BeginFrame,len(u.trajectory)-1)))

m_x        = 2 # int(input ("\nPlease enter Nx: "))
n_y        = 2 # int(input ("\nPlease enter Ny: "))
Resx       = 200 # int(input ("\nPlease enter the number of grids in x direction: "))
Resy       = 200 # int(input ("\nPlease enter the number of grids in y direction: "))

# Making the desired selections. 
UpLayerIndex  = GetGroupIndexMda(UpLayer, GrLs, ndx)
LowLayerIndex = GetGroupIndexMda(LowLayer, GrLs, ndx)
Uplay = u.select_atoms(UpLayerIndex)
Lolay = u.select_atoms(LowLayerIndex)

# Generate grid
X01   = np.linspace(0, 100, Resx+1)
Y01   = np.linspace(0, 100, Resy+1)
X0,Y0 = np.meshgrid(X01, Y01)
ZmU   = 0 * X0
HmU   = 0 * X0
num   = 0

logdata = ['#0.frame', '#1.Lx', '#2.Ly', '#3.Lz', '#4.Axy', '#5.ASur', '#6.Hm']
np.savetxt('Upperlog.dat', np.reshape(logdata, (1, len(logdata))), fmt='%s')

for ts in u.trajectory[BeginFrame:EndFrame+1]:
   Lx     = ts.dimensions[0]/10.0
   Ly     = ts.dimensions[1]/10.0
   Lz     = ts.dimensions[2]/10.0

   UpCoor = Uplay.positions/10.0
   LoCoor = Lolay.positions/10.0
   X1     = np.linspace(0, Lx, Resx+1)
   Y1     = np.linspace(0, Ly, Resy+1)
   X, Y   = np.meshgrid(X1, Y1)
   ZM     = 0 * X
   fxM    = 0 * X
   fxxM   = 0 * X
   fyM    = 0 * X
   fyyM   = 0 * X
   fxyM   = 0 * X
   dx     = Lx/Resx
   dy     = Ly/Resy 
   
   # Calculations for upper leaflet only
   for [data, layname,Zm,Hm] in [[UpCoor,'Upper',ZmU,HmU]] : #, [LoCoor,'Lower',ZmL,HmL]]:
      Par   = open(layname + 'Params.dat','ab') # contains Fourier transform values
      Lg    = open(layname + 'log.dat','ab')
      guess = np.zeros((2*m_x+1)*(2*n_y+1))
      guess[int(((2*m_x+1)*(2*n_y+1)-1)/2)] = np.mean(data[:,2])
      params, pcov = optimize.curve_fit(Zfunc, data[:,:2], data[:,2], guess)
      np.savetxt(Par, np.reshape(params,(1,(2*m_x+1)*(2*n_y+1))), fmt='%.8e', delimiter='\t')
      Z   = 0 * X
      fx  = 0 * X
      fxx = 0 * X
      fy  = 0 * X
      fyy = 0 * X
      fxy = 0 * X

      for n in range (-m_x,m_x+1):
         for m in range (-n_y,n_y+1):
            topimlx  = 2*np.pi*m/Lx
            topimlx2 = topimlx**2
            topinly  = 2*np.pi*n/Ly
            topinly2 = topinly**2
            A_ind    = (m+m_x)*(2*n_y+1)+(n+n_y)
            Arg      = topimlx*X+topinly*Y
            CosArg   = np.cos(Arg)
            SinArg   = np.sin(Arg)
            
            Z   += params[A_ind] * (CosArg + SinArg) # Z=f(x,y)       
            fx  += params[A_ind] * topimlx * (CosArg - SinArg) # fx=dZ/dx
            fxx -= params[A_ind] * topimlx2 * (CosArg + SinArg) # fxx=d2Z/dx2
            fy  += params[A_ind] * topinly * (CosArg - SinArg) # fy=dZ/dy
            fyy -= params[A_ind] * topinly2 * (CosArg + SinArg) # fyy=d2Z/dy2
            fxy -= params[A_ind] * topimlx * topinly * (CosArg + SinArg) # fxy=d2Z/dxdy

      H   = -(fxx*(1+fy**2)-2*fxy*fx*fy+fyy*(1+fx**2))/(2*(1+fx**2+fy**2)**1.5) # mean curvature (We choose the unit vector normal to the surface downward whereas the unit vector of the z axis is upward so there is a negative sign in the formula.) 
      np.savetxt(f'./buckled_membrane_dat/r1_{ts.frame}.txt',H,fmt='%.4e')
   ##################
   # Write the resid, x,y,z, curvature CSV for this frame
   # Interpolator for curvature values
   H_interpolator = RegularGridInterpolator((Y1, X1), H, bounds_error=False, fill_value=None)

   # Get lipid positions and resids
   positions = u.select_atoms('resname PC_u PS_u and name PO4').positions/10
   resids = u.select_atoms('resname PC_u PS_u and name PO4').resids

   # Interpolate curvature at each lipid's x,y position
   curvature_data = []
   for i in range(len(positions)):
      xi, yi, zi = positions[i]
      Hi = H_interpolator((yi, xi))  # note: order is (y, x)
      curvature_data.append([resids[i], xi, yi, zi, Hi])

   # Write to CSV file (append mode)
   with open(f"UpperLeaflet_Curvature_{ts.frame}.csv", "w", newline='') as csvfile:
      writer = csv.writer(csvfile)
      if ts.frame == 0:  # header only once
         writer.writerow(['resid', 'x', 'y', 'z', 'curvature'])
      writer.writerows(curvature_data)

   num += 1
   print("frame ",ts.frame)
   

