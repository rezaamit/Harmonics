import numpy 
import lal
import lalsimulation
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from lalsimulation import SimInspiralTransformPrecessingWvf2PE
from lalsimulation import DetectorPrefixToLALDetector
from lalsimulation import FLAG_SEOBNRv4P_HAMILTONIAN_DERIVATIVE_NUMERICAL
from lalsimulation import FLAG_SEOBNRv4P_EULEREXT_QNM_SIMPLE_PRECESSION
from lalsimulation import FLAG_SEOBNRv4P_ZFRAME_L
from lalsimulation import SimNeutronStarEOS4ParameterPiecewisePolytrope
from lalsimulation import SimNeutronStarRadius
from lalsimulation import CreateSimNeutronStarFamily
from lalsimulation import SimNeutronStarLoveNumberK2
from lalsimulation import SimNeutronStarEOS4ParameterSpectralDecomposition
from lal import MSUN_SI, C_SI, MRSUN_SI
from pycbc.types import FrequencySeries




def component_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass1, mass2, f_ref, phase):
    
    data = []
    for i in range(len(theta_jn)):
        iota, S1x, S1y, S1z, S2x, S2y, S2z = SimInspiralTransformPrecessingNewInitialConditions(theta_jn[i], phi_jl[i],
                                                                                                tilt_1[i], tilt_2[i], phi_12[i],
                                                                                                a_1[i], a_2[i], 
                                                                                                mass1[i] *MSUN_SI, 
                                                                                                mass2[i] * MSUN_SI,
                                                                                                float(f_ref[i]), float(phase[i]))
        
        data.append([iota, S1x, S1y, S1z, S2x, S2y, S2z])
        
        return data


def calculate_dpsi(theta_jn, phi_jl, beta):
    
    if theta_jn == 0:
        return -1. * phi_jl
    
    n = numpy.array([numpy.sin(theta_jn), 0, numpy.cos(theta_jn)])
    j = numpy.array([0, 0, 1])
    l = numpy.array([numpy.sin(beta) * numpy.sin(phi_jl), numpy.sin(beta) * numpy.cos(phi_jl), numpy.cos(beta)])
    p_j = numpy.cross(n, j)
    p_j /= numpy.linalg.norm(p_j)
    p_l = numpy.cross(n, l)
    p_l /= numpy.linalg.norm(p_l)
    cosine = numpy.inner(p_j, p_l)
    sine = numpy.inner(n, numpy.cross(p_j, p_l))
    dpsi = numpy.pi / 2 + numpy.sign(sine) * numpy.arccos(cosine)
    
    return dpsi



def calculate_dphi(theta_jn, phi_jl, beta):
    
    if theta_jn == 0:
        return 0.
    
    n = numpy.array([numpy.sin(theta_jn), 0, numpy.cos(theta_jn)])
    j = numpy.array([0, 0, 1])
    l = numpy.array([numpy.sin(beta) * numpy.sin(phi_jl), numpy.sin(beta) * numpy.cos(phi_jl), numpy.cos(beta)])
    e_j = numpy.cross(j,l)
    e_j /= numpy.linalg.norm(e_j)
    e_n = numpy.cross(n,l)
    e_n /= numpy.linalg.norm(e_n)
    cosine = numpy.inner(e_j, e_n)
    sine = numpy.inner(l, numpy.cross(e_j, e_n))
    dphi = numpy.sign(sine)*numpy.arccos(cosine)
    
    return dphi


def calculate_opening_angle(mass1, mass2, chi_eff, chi_p, freq):
    
    theta1 = numpy.arctan2(chi_p, chi_eff)
    chi1 = numpy.sqrt(chi_p**2 + chi_eff**2)

    beta, s1x, s1y, s1z, s2x, s2y, s2z = SimInspiralTransformPrecessingNewInitialConditions(0., 0., 
                                                                                            theta1, 0., 0., chi1, 0., 
                                                                                            mass1*MSUN_SI, mass2*MSUN_SI, 
                                                                                            freq, 0.)
    return beta




def make_single_spin_waveform(waveform, theta_JN, phi_JL, phi0, psi0, chi_eff, chi_p, mass1, mass2, flow, df, flen):
    
    hp, hc = make_single_spin_plus_cross(waveform, theta_JN, phi_JL, phi0, chi_eff, chi_p, mass1, mass2, flow, df, flen)

    beta = calculate_opening_angle(mass1, mass2, chi_eff, chi_p, flow)
    dpsi = calculate_dpsi(theta_JN, phi_JL, beta)
    h = make_waveform(hp, hc, psi0 + dpsi)
    dphi = calculate_dphi(theta_JN, phi_JL, beta)
    h *= numpy.exp(2j * dphi )
    
    return h




def make_single_spin_plus_cross(approximant, theta_JN, phi_JL, phi0, chi_eff, chi_p, mass1, mass2, flow, df, flen):
   
    chi1 = numpy.sqrt(chi_p**2 + chi_eff**2)
    theta1 = numpy.arctan2(chi_p, chi_eff)
    inc, s1x, s1y, s1z, s2x, s2y, s2z = SimInspiralTransformPrecessingNewInitialConditions(theta_JN, phi_JL, 
                                                                                           theta1, 0., 0., chi1, 0., 
                                                                                           mass1*MSUN_SI, mass2*MSUN_SI, 
                                                                                           flow, phi0)
     
    fhigh = 2048
    f_ref = flow
    distance = 1.*1e6
    parameters = {}
    parameters['approximant'] = lalsimulation.GetApproximantFromString(str(approximant))
    
    hp, hc = lalsimulation.SimInspiralChooseFDWaveform(lal.MSUN_SI*mass1, lal.MSUN_SI*mass2, 
                                                s1x, s1y, s1z, s2x, s2y, s2z,
                                                distance*lal.PC_SI, inc, phi0, 
                                                0.0, 0.0, 0.0, df, flow, fhigh,
                                                f_ref, None, parameters['approximant'])
    
    hp = FrequencySeries(hp.data.data[:], delta_f = hp.deltaF, epoch = hp.epoch)
    hc = FrequencySeries(hc.data.data[:], delta_f = hc.deltaF, epoch = hc.epoch)
    
    hp.resize(flen)
    hc.resize(flen)
    
    return hp, hc

def make_waveform(hp, hc, psi0):
    
    fp = numpy.cos(2 * psi0)
    fc = numpy.sin(2 * psi0)
    temp = fp * hp + fc * hc
    
    return temp

