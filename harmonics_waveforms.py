import numpy 
from waveform import _make_waveform_from_spherical as _make_waveform
from conversions import calculate_dpsi, calculate_dphi, component_spins


def make_waveform( approx, theta_jn, phi_jl, phase, psi_J, mass1, mass2, tilt_1, tilt_2, phi_12, a_1, a_2, beta, distance, f_ref, flow, fhigh, df, flen):
    
    hp, hc = _make_waveform(approx, theta_jn, phi_jl, phase, mass1, mass2, tilt_1, tilt_2, phi_12, a_1, a_2, distance, f_ref,
                            flow, fhigh, df, flen)

    dpsi = calculate_dpsi(theta_jn, phi_jl, beta)
    
    fp = numpy.cos(2 * (psi_J - dpsi))
    fc = -1. * numpy.sin(2 * (psi_J - dpsi))
    h = (fp * hp + fc * hc)
    h *= numpy.exp(2j * calculate_dphi(theta_jn, phi_jl, beta))

    return h



def calculate_precessing_harmonics(mass1, mass2, a_1, a_2, tilt_1, tilt_2, phi_12, beta, distance, harmonics, approx, f_ref, flow, fhigh, df, flen):
    
    harm = {}
    if (0 in harmonics) or (4 in harmonics):
        h0 = make_waveform(approx, 0., 0., 0., 0., mass1, mass2, tilt_1, tilt_2, phi_12, a_1, a_2, beta, 
                           distance, f_ref, flow, fhigh, df, flen)
        hpi4 = make_waveform(approx, 0., 0., numpy.pi / 4, numpy.pi / 4, mass1, mass2, tilt_1, tilt_2,
                             phi_12, a_1, a_2, beta, distance, f_ref, flow, fhigh, df, flen)
        if (0 in harmonics):
            harm[0] = (h0 - hpi4) / 2
        if (4 in harmonics):
            harm[4] = (h0 + hpi4) / 2
    if (1 in harmonics) or (3 in harmonics):
        h0 = make_waveform(approx, numpy.pi / 2, 0., numpy.pi / 4, numpy.pi / 4, mass1, mass2,
                           tilt_1, tilt_2, phi_12, a_1, a_2, beta, distance, f_ref, flow, fhigh, df, flen)
        
        hpi2 = make_waveform(approx, numpy.pi / 2, numpy.pi / 2, 0., numpy.pi / 4, mass1, mass2,
                             tilt_1, tilt_2, phi_12, a_1, a_2, beta, distance, f_ref, flow, fhigh, df, flen)
        
        if (1 in harmonics):
            harm[1] = -1. * (h0 + hpi2) / 4
        if (3 in harmonics):
            harm[3] = -1. * (h0 - hpi2) / 4
            
    if (2 in harmonics):
        h0 = make_waveform(approx, numpy.pi / 2, 0., 0., 0., mass1, mass2, tilt_1, tilt_2, phi_12, a_1, a_2,
                           beta, distance, f_ref, flow, fhigh, df, flen)
        hpi2 = make_waveform(approx, numpy.pi / 2, numpy.pi / 2, 0., 0., mass1, mass2, tilt_1, tilt_2, phi_12, a_1, a_2, 
                             beta, distance, f_ref, flow, fhigh, df, flen)
        harm[2] = -(h0 + hpi2) / 6
        
    return harm



