import lal
import lalsimulation as lalsim



def add_waveforms(h1, h2):
    
    if h1.sample_rate != h2.sample_rate:
        raise ValueError('Sample rate must be the same')

    h_sum = h1.copy()
    h_sum_lal = h_sum.lal()
    lalsim.SimAddInjectionREAL8TimeSeries(h_sum_lal, h2.lal(), None)
    h_sum.data[:] = h_sum_lal.data.data[:]
    
    return h_sum


