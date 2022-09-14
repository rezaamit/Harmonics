from conversions import component_spins
from scipy.interpolate import interp1d
import lal
import lalsimulation as lalsim
from lal import MSUN_SI, G_SI, C_SI
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
from pycbc.types import FrequencySeries
import numpy 



def _make_waveform_from_spherical(approx, theta_jn, phi_jl, phase, mass1, mass2, 
                                  tilt_1, tilt_2, phi_12, a_1, a_2, distance, f_ref, flow, fhigh, df, flen):
    
    
    if f_ref is None:
        f_ref = flow

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
        _cartesian_spins_from_spherical(
            [theta_jn], [phi_jl], [tilt_1], [tilt_2], [phi_12], [a_1], [a_2],
            [mass1], [mass2], [f_ref], [phase]
        )
    return _make_waveform_from_cartesian(approx, mass1, mass2, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
        spin_2z, iota, phase, distance, flow, df, flen, fhigh, f_ref)



def _cartesian_spins_from_spherical(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass1, mass2, f_ref, phase):
    
    
    data = component_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, mass1, mass2, f_ref, phase)
    data = numpy.array(data).T
    
    return data



def _make_waveform_from_cartesian(approximant, mass1, mass2, spin_1x, spin_1y, spin_1z, 
                                  spin_2x, spin_2y, spin_2z, iota, phase, distance, flow, df, flen, fhigh, f_ref):
    
    
    if f_ref is None:
        f_ref = flow
    
    distance = 1.*1e6
    
    
    parameters = {}
    parameters['approximant'] = lalsim.GetApproximantFromString(str(approximant))
    
    
    hp, hc = lalsim.SimInspiralChooseFDWaveform(lal.MSUN_SI * mass1, lal.MSUN_SI * mass2,
                                                spin_1x[0], spin_1y[0], spin_1z[0], 
                                                spin_2x[0], spin_2y[0], spin_2z[0], distance*lal.PC_SI, 
                                                iota[0], phase, 0.0, 0.0, 0.0, df, flow, 
                                                fhigh, f_ref, None, parameters['approximant'])
    
    
    
    hp = FrequencySeries(hp.data.data[:], delta_f = hp.deltaF, epoch = hp.epoch)

    hc = FrequencySeries(hc.data.data[:], delta_f = hc.deltaF, epoch = hc.epoch)
    
    
    if flen is not None:
        hp.resize(flen)
        hc.resize(flen)
    
    return hp, hc
    

