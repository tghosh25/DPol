#!/usr/bin/env python

# PURPOSE:
# Code to simulate dust polarization maps at 353 GHz (Vansyngel et al. 2016)
# EXAMPLE:
# MODIFICATION HISTORY:
# 01-06-2017 ; last updated the code
# 28-04-2017 ; modified it to make it compatible with github
# 25-04-2017 ; copied from flavien_test_model_v2.py


import sys
import numpy as np
import healpy as hp
import ConfigParser
import argparse
import datetime


class output(object):
    def __init__(self, config_dict):
        self.output_prefix = str(config_dict['output_prefix'])
        self.nside         = int(config_dict['nside'])
        self.nsim          = int(config_dict['nsim'])
        self.output_dir    = str(config_dict['output_dir'])
        self.input_dir     = str(config_dict['input_dir'])
        self.fwhm          = float(config_dict['fwhm']) 
        self.b0_model      = int(config_dict['b0_model']) 
        self.lonb0         = float(config_dict['lonb0']) 
        self.latb0         = float(config_dict['latb0']) 
        self.b0_filename   = str(config_dict['b0_filename']) 
        self.nlay          = float(config_dict['nlay']) 
        self.fm            = float(config_dict['fm']) 
        self.p0            = float(config_dict['p0']) 
        self.spec          = float(config_dict['spec']) 
        self.lcross        = float(config_dict['lcross']) 
        self.t             = float(config_dict['t']) 
        self.f             = float(config_dict['f']) 
        self.rho           = float(config_dict['rho']) 



parser = argparse.ArgumentParser(description='Code to simulate dust polarization maps.')
parser.add_argument('config_file', help='Main configuration file.')


# Get the input parameters from the configuration file.

Config = ConfigParser.ConfigParser()
a = Config.read(parser.parse_args().config_file)
if a==[] :
    print 'Couldn\'t find file config.ini'
    exit(1)
out = output(Config._sections['CommonParameters'])

#print out.nside, out.p0, out.t, out.lcross, out.spec, out.f, out.rho, out.nlay, out.fm, out.lonb0, out.latb0


# read input map and mask files 

nlmax      = 2*out.nside
npix       = hp.nside2npix(out.nside)
lpix       = np.arange(npix)
los_vec    = hp.pix2vec(out.nside,lpix)
theta, phi = hp.pix2ang(out.nside,lpix)

tmp         = hp.read_map(out.input_dir+'COM_CompMap_Dust-GNILC-F353_2048_R2.00.fits',field=0)
tmp         = (tmp - 0.13)/2.83e-4
if(npix == np.size(tmp)):
    inmap = tmp
else:
    inmap  = hp.ud_grade(tmp, nside_out=out.nside, order_in='RING')

gmask         = hp.read_map(out.input_dir+'GalMaskApo2Fsky81_ns256.fits',field=0)
if(npix != np.size(gmask)):
    gmask  = hp.ud_grade(gmask, nside_out=out.nside, order_in='RING')


# simulating large-scale uniform magnetic field

if(out.b0_model == 1):
    B0vec=np.zeros((3,npix))  
    B0vec[0] = np.cos(np.radians(out.lonb0))*np.cos(np.radians(out.latb0))
    B0vec[1] = np.sin(np.radians(out.lonb0))*np.cos(np.radians(out.latb0))
    B0vec[2] = np.sin(np.radians(out.latb0))
else:
    B0_tmp = hp.read_map(out.input_dir+out.b0_filename, field=(0,1,2))
    if(npix == np.size(B0_tmp[0])):
        B0vec = B0_tmp
    else:
        B0vec  = hp.ud_grade(B0_tmp, nside_out=out.nside, order_in='RING')



north_vec =   [-np.cos(phi)*np.cos(theta),-np.sin(phi)*np.cos(theta),np.sin(theta)]
east_vec  = - np.transpose(np.cross(np.transpose(los_vec), np.transpose(north_vec)))


# APS of turbulent magnetic field field

ell=np.arange(nlmax+1)
cl_mag = ell * 0.
cl_mag[2:nlmax]=(ell[2:nlmax]/50.)**(out.spec)  

# Constructing a filter function 

wl = np.ones(nlmax)

for i in range(nlmax):    
    if(i <= out.lcross - (0.5*out.lcross)): wl[i]= 0.0
    if((i > out.lcross - (0.5*out.lcross)) & (i < out.lcross + (0.5*out.lcross))): wl[i]=0.5 * (1. - np.sin(np.pi*(out.lcross-i)/out.lcross))


# Header of output maps

extra_header = []
date = datetime.datetime.now()
extra_header.append(('DATE', '%s-%s-%s' %(date.day, date.month, date.year)))
extra_header.append(('PIXTYPE', 'HEALPIX'))
extra_header.append(('MTYPE', 'DUST MODEL'))
extra_header.append(('FREQ', '353 GHz'))
extra_header.append(('VERSION', '1.0'))


for relz in range(out.nsim):

    print "Relz:", relz+1

    I_tmp = np.zeros(npix)
    Q_tmp = np.zeros(npix)
    U_tmp = np.zeros(npix)

    for i in np.arange(out.nlay):

        Bturb=np.zeros((3,npix))
        
        Bturb[0]=hp.synfast(cl_mag, out.nside, lmax=nlmax, pol=False)
        Bturb[1]=hp.synfast(cl_mag, out.nside, lmax=nlmax, pol=False)
        Bturb[2]=hp.synfast(cl_mag, out.nside, lmax=nlmax, pol=False)
    
        sig_map=np.mean([np.std(Bturb[0]), np.std(Bturb[1]), np.std(Bturb[2])])
        fm_n=out.fm/(sig_map*np.sqrt(3.))                       

        Bvec = B0vec + fm_n*Bturb
        norm_Bvec=np.sqrt(np.sum(Bvec**2, axis=0))        # normalization to unit length
        Bvec=Bvec/norm_Bvec
   
        B_times_los = los_vec[0]*Bvec[0] + los_vec[1]*Bvec[1] + los_vec[2]*Bvec[2] 
        Bvec_perp=Bvec-(los_vec*B_times_los)
        cos2gamma = np.sum(Bvec_perp**2, axis=0)
            
        buf = (Bvec_perp[0]*north_vec[0]+Bvec_perp[1]*north_vec[1]+Bvec_perp[2]*north_vec[2])/np.sqrt(cos2gamma)
        B_angle = np.arccos(np.clip(buf, -1, 1))        
        buf2 =  Bvec_perp[0]*east_vec[0]+Bvec_perp[1]*east_vec[1]+Bvec_perp[2]*east_vec[2]
        neg_angle=np.where(buf2 < 0.)                              
        B_angle[neg_angle] = - B_angle[neg_angle]          
        psi_ang = B_angle + (np.pi/2.) 
        b=np.where(psi_ang > np.pi)
        psi_ang[b]=  psi_ang[b]- (2.*np.pi)
        
        QoImap =  np.cos(2.*psi_ang)  
        UoImap = -np.sin(2.*psi_ang)                                               

        psi= 0.5 * np.arctan2(- UoImap,  QoImap) 

        I_tmp =   I_tmp +  (1. - out.p0*(cos2gamma - 2./3.))
        Q_tmp =   Q_tmp +  (cos2gamma * np.cos(2.*psi))
        U_tmp =   U_tmp -  (cos2gamma * np.sin(2.*psi))


    dmap=np.zeros((3,npix))
    dmap[0] = inmap
    dmap[1] = Q_tmp/I_tmp * inmap
    dmap[2] = U_tmp/I_tmp * inmap

      
#Smooth final maps with Gaussian fwhm smoothing

    tmpmap=hp.smoothing(dmap*gmask, fwhm=np.radians(np.sqrt((out.fwhm/60.)**2 - (5./60.)**2)), pol=True, lmax=nlmax)

    alms=hp.map2alm(tmpmap, lmax=nlmax, pol=True)
    nalms=(alms[0]*out.t, alms[1]*out.p0 + hp.almxfl(alms[0], wl*out.rho*out.p0), hp.almxfl(alms[2], (1. - (1. - out.f)*wl) * out.p0))
    
    map=hp.alm2map(nalms, out.nside, lmax=nlmax, pol=True, pixwin=True)    
    hp.write_map(out.output_dir + '%s_TQU_353GHz_relz%s.fits' % (out.output_prefix,str(relz+1).zfill(3)), map*gmask,  coord='G', column_names=('TEMPERATURE','Q_POLARISATION','U_POLARISATION'),extra_header=extra_header)



