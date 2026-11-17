# -*- coding: utf-8 -*-
"""
Optical pulses on a discrete time/frequency grid.

Conventions (see docs/physics.md):

- `t` is a uniform time grid centred on t = 0, in seconds.
- `pulse.a` is the complex envelope in sqrt(W), so |a|^2 is power in W.
- `pulse.A = fft(fftshift(a))` is the spectrum with its time origin at t = 0
  (a transform-limited pulse has flat spectral phase). It is in FFT order:
  `pulse.f = fftfreq(N, dt)` are frequencies relative to the reference
  `pulse.f0`, and `pulse.f_abs = f + f0` are absolute frequencies.
- Energies are in J, PSDs in W/Hz (or dBm/Hz), using the repetition rate
  `frep` to convert pulse energy to average power.

@author: Luis Ledezma
"""
import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq
import matplotlib.pyplot as plt
import scipy.signal
from scipy.constants import pi, c

def coherence_g1(pulse_train):
    '''
    Calculates g1 coherence function for a train of pulses
    It uses all possible pairs of pulses in the train

    Parameters
    ----------
    pulse_train : ARRAY
        ARRAY OF PULSES. THERE SHOULD BE A PULSE PER COLUMN. 
        SO THE NUMBER OF COLUMNS WILL BE EQUAL TO THE NUMBER OF PULSES.
        IT USES ALL N*(N-1)/2 PAIR OF PULSES AVAILABLE

    Returns
    -------
    g : TYPE
        THE COHERENCE FUNCTION IN THE FREQUENCY DOMAIN.

    '''
    E12 = 0
    E11 = 0
    E22 = 0
    
    Npulses = pulse_train.shape[1]
    nn = 0 #Number of pairs

    for k1 in range(Npulses - 1):
        
        E1 = fft(pulse_train[:, k1])
        
        for k2 in np.arange(k1+1, Npulses):
            
            E2 = fft(pulse_train[:, k2])
    
            E12 += np.conj(E1) * E2
            E11 += np.conj(E1) * E1
            E22 += np.conj(E2) * E2
        
            nn += 1
    
    g = np.abs(E12 / np.sqrt(E11 * E22) )   
    print('Calculated coherence with %i pairs of pulses' %(nn) )
    return g

def average_g1(x, g1):
    '''
    Spectrally averaged coherence: g1 weighted by the energy spectral
    density of the time-domain field `x`.
    '''
    Xesd = np.abs( fft(x) )**2
    return np.sum( g1 * Xesd ) / np.sum( Xesd ) 

def gaussian(t, Energy, FWHM, f0=0):
    '''
    Gaussian envelope, |a|^2 = Ppeak exp(-4 ln2 (t/FWHM)^2).

    Parameters
    ----------
    t : array
        Time grid (s).
    Energy : float
        Pulse energy (J). Ppeak = 0.94 Energy / FWHM.
    FWHM : float
        Intensity full width at half maximum (s).
    f0 : float, optional
        Carrier frequency offset from the grid's reference frequency (Hz).

    Returns
    -------
    array
        Complex envelope in sqrt(W).
    '''
    Ppeak = 0.94*Energy/FWHM
    x = np.sqrt(Ppeak) * np.exp(-2*np.log(2)*(t/FWHM)**2)
    return x * np.exp(1j*2*pi*f0*t)

def sech(t, Energy, FWHM, f0=0):
    '''
    Sech envelope, |a|^2 = Ppeak sech^2(1.76 t/FWHM).

    Parameters
    ----------
    t : array
        Time grid (s).
    Energy : float
        Pulse energy (J). Ppeak = 0.88 Energy / FWHM.
    FWHM : float
        Intensity full width at half maximum (s).
    f0 : float, optional
        Carrier frequency offset from the grid's reference frequency (Hz).

    Returns
    -------
    array
        Complex envelope in sqrt(W).
    '''
    Ppeak = 0.88*Energy/FWHM
    x = np.sqrt(Ppeak) / np.cosh( 1.76 * t/ FWHM )
    return x * np.exp(1j*2*pi*f0*t)

def noise(t, Npower):
    '''
    Complex white Gaussian noise with mean power `Npower` (W) per sample.
    Uses numpy's global random generator (seed it for reproducible runs).
    '''
    sigma = np.sqrt(Npower) / np.sqrt(2)
    x = np.random.normal(scale = sigma, size = t.size) + 1j*np.random.normal(scale = sigma, size = t.size)
    return x

def gaussian_pulse(t, FWHM, f_ref, Energy=None, Ppeak=None, Pavg=None, f0=0, Npwr_dB=100, frep=250e6):
    '''
    Gaussian `pulse` object with additive noise.

    Parameters
    ----------
    t : array
        Time grid (s), centred on 0.
    FWHM : float
        Intensity FWHM (s).
    f_ref : float
        Reference frequency of the grid (Hz); becomes `pulse.f0`.
    Energy, Ppeak, Pavg : float, optional
        Give one of: pulse energy (J), peak power (W), or average power (W,
        converted with `frep`). Default 1 pJ.
    f0 : float
        Absolute carrier frequency (Hz). Pass it explicitly: the default 0
        means 0 Hz, not f_ref (#43).
    Npwr_dB : float, optional
        Noise power below the peak power, in dB (sign is ignored).
    frep : float, optional
        Repetition rate (Hz), used for average powers and PSDs.

    Returns
    -------
    pulse
    '''
    
    if Energy != None:
        pass
    elif Ppeak != None:
        Energy = Ppeak * FWHM / 0.94
    elif Pavg != None:
        Energy = Pavg/frep
    else:
        print('Warning: Using default 1pJ of energy for the pulse')
        Energy = 1e-12
        
    x = gaussian(t, Energy, FWHM, f0-f_ref)
    
    Npwr_dB = abs(Npwr_dB)
    Npwr = np.amax(np.abs(x)**2) * 10**( -Npwr_dB/10 )
    n = noise(t, Npwr)
    return pulse(t, x+n, c/f_ref, frep=frep, domain='Time')

def sech_pulse(t, FWHM, f_ref, Energy=None, Ppeak=None, Pavg=None, f0=0, Npwr_dB=100, frep=250e6):
    '''
    Sech `pulse` object with additive noise. Same arguments as
    `gaussian_pulse`; pass `f0` explicitly (#43).
    '''
    
    if Energy != None:
        pass
    elif Ppeak != None:
        Energy = Ppeak * FWHM / 0.88
    elif Pavg != None:
        Energy = Pavg/frep
    else:
        print('Warning: Using default 1pJ of energy for the pulse')
        Energy = 1e-12
        
    x = sech(t, Energy, FWHM, f0-f_ref)
    
    Npwr_dB = abs(Npwr_dB)
    Npwr = np.amax(np.abs(x)**2) * 10**(-Npwr_dB/10 )
    n = noise(t, Npwr)
    return pulse(t, x+n, c/f_ref, frep=frep, domain='Time')

def filter_signal(f_abs, X, f0, bw, type='ideal'):
    '''
    Band-pass filter a spectrum.

    Parameters
    ----------
    f_abs : array
        Absolute frequencies of `X` (Hz), FFT order.
    X : array
        Spectrum to filter (e.g. `pulse.A`).
    f0, bw : float
        Centre frequency and full bandwidth of the pass band (Hz).
    type : {'ideal', 'butterworth'}
        Brick-wall filter, or a 15th-order Butterworth band pass.

    Returns
    -------
    filtered : array
        Time-domain envelope in natural time order, like `pulse.a`.
    h : array
        Filter response on the `f_abs` grid.
    '''
    
    if type == 'ideal':
        f1 = f0 - bw/2
        f2 = f0 + bw/2
        h = np.ones_like(f_abs)
        h[ f_abs < f1 ] = 0
        h[ f_abs > f2 ] = 0
    elif type == 'butterworth':
        f_min = np.amin(f_abs)
        f_max = np.amax(f_abs)
        f1 = (f0 - bw/2 - f_min)/(f_max-f_min)
        f2 = (f0 + bw/2 - f_min)/(f_max-f_min)
        sos = scipy.signal.butter(15, [f1, f2], 'bandpass', output='sos')
        _, h = scipy.signal.sosfreqz(sos, worN = X.size)
        h = fftshift(h)
    
    #X follows the pulse.A convention, A = fft(fftshift(a)), so undo the
    #shift to get the envelope back in natural time order (#41)
    filtered = fftshift( ifft( X * h ) )
    return filtered, h

def energy_td(t, x):
    'Pulse energy (J) integrated in the time domain, sum |x|^2 dt.'
    dt = t[1] - t[0]
    pwr = abs(x)**2
    energy = np.sum(pwr)*dt
    return energy

def energy_fd(t, x):
    'Pulse energy (J) integrated in the frequency domain (equals energy_td by Parseval).'
    dt = t[1] - t[0]
    f = fftfreq(t.size, dt)
    df = f[1]-f[0]
    Xw = get_spectrum(t, x)
    energy = np.sum(Xw)*df
    return energy

def pulse_center(t, x):
    'Intensity-weighted mean time (s).'
    dt = t[1] - t[0]
    I = abs(x)**2
    return np.sum(t*I*dt)/np.sum(I*dt)

def FWHM(t, x, prominence=1):
    '''
    Intensity FWHM (s) of every peak of |x|^2 with at least `prominence` (W).
    Returns an array with one width per peak found.
    '''
    dt = t[1]-t[0] #Sample spacing
    x2 = abs(x)**2 #Intensity
    peaks = scipy.signal.find_peaks(x2, prominence=prominence)
    width = scipy.signal.peak_widths(x2, peaks[0])
    fwhm = width[0]*dt
    return fwhm

def get_spectrum(t, x):
    'Energy spectral density |X(f)|^2 in J/Hz, FFT order.'
    dt = t[1]-t[0] #Sample spacing
    X = fft(x)
    Xmag = abs(X)*dt
    return Xmag**2

def get_psd(t, x, frep):
    'Power spectral density (W/Hz) of a pulse train at repetition rate frep.'
    Xw = get_spectrum(t, x)
    psd = Xw * frep
    return psd

def get_psd_dBm(t, x, frep):
    'Power spectral density in dBm/Hz.'
    Xpsd = get_psd(t, x, frep)
    Xpsd_dBm = 10*np.log10(Xpsd) + 30
    return Xpsd_dBm

def plot_vs_time(t, x, ylabel='', t_unit='ps', ax=None):
    '''
    Private function to plot stuff x vs time (fs)
    '''
    if t_unit=='ps':
        t = t*1e12
    elif t_unit=='fs':
        t = t*1e15
    elif t_unit=='ns':
        t = t*1e9
    else:
        print('What was that? Don\'t know that time unit...')
        
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(t, x)
    ax.grid(True)
    ax.set_xlabel('Time (' + t_unit + ')');
    ax.set_ylabel(ylabel);
    
    return ax
        
def plot_vs_wavelength(wl, x, ylabel='', wl_unit='um', ax=None):
    '''
    Private function to plot stuff x vs wavelength (microns)
    '''        
    if wl_unit=='um':
        wl = wl*1e6
    elif wl_unit=='nm':
        wl = wl*1e9
    else:
        print('What was that? Don\'t know that wavelength unit...')
        
    wl = fftshift(wl)
    x = fftshift(x)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(wl, x)
    ax.grid(True)
    ax.set_xlabel('Wavelength (' + wl_unit + ')');
    ax.set_ylabel(ylabel);
        
    return ax
            
def plot_vs_freq(f, x, ylabel='', f_unit='THz', ax=None):
    '''
    Private function to plot stuff x vs frequency (THz)
    '''
    if f_unit=='THz':
        f = f*1e-12
    elif f_unit=='GHz':
        f = f*1e-9
    elif f_unit=='PHz':
        f = f*1e-15
    else:
        print('What was that? Don\'t know that frequency unit...')
        
    f = fftshift(f)
    x = fftshift(x)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(f, x)
    ax.grid(True)
    ax.set_xlabel('Frequency (' + f_unit + ')');
    ax.set_ylabel(ylabel);
        
    return ax
 
  
def plot_magsq(t, x, label='Pulse Intensity (W)', ax=None, t_unit='ps'):
    '''Plot |x|^2 versus time; t_unit is 'fs', 'ps' or 'ns'. Returns the axis.'''
    x2 = abs(x)**2
    ax = plot_vs_time(t, x2, ylabel=label, ax=ax, t_unit=t_unit)
    
    return ax
        
def plot_PSD(t, x, frep, f0=0, ax=None, f_unit='um', dBm=True):
    '''
    Plot the PSD versus wavelength ('um', 'nm') or frequency ('GHz', 'THz',
    'PHz'), in dBm/Hz or W/Hz. `f0` is the grid's reference frequency.
    Returns the axis.
    '''
    
    dt = t[1] - t[0]
    f = fftfreq(t.size, dt) + f0
    
    if dBm:
        psd = get_psd_dBm(t, x, frep)
        label='Power Spectral Density (dBm / Hz)'
    else:
        psd = get_psd(t, x, frep)
        label='Power Spectral Density (W / Hz)'
    
    if f_unit in ['um', 'nm']:
        ax = plot_vs_wavelength(c/f, psd, ylabel=label, ax=ax, wl_unit=f_unit)
    elif f_unit in ['GHz', 'THz', 'PHz']:
        ax = plot_vs_freq(f, psd, ylabel=label, ax=ax, f_unit=f_unit)
        
    return ax
    
def add_t_offset(pulse, t_offset):
    '''
    Shift a pulse in time by applying a linear spectral phase, in place.
    Note the sign: a positive `t_offset` moves the pulse to earlier (negative)
    time. Returns the same pulse object.
    '''
    p_spec = pulse.A
    p_spec = np.multiply(p_spec,np.exp(1j*t_offset*pulse.Omega))
    pulse.update_fd(p_spec)
    return pulse

def phase(x, unwrap=True, threshold=1e-3):
    '''
    Phase of `x` (rad), NaN where |x| is below `threshold` times its peak.
    '''
    xpeak = np.amax(np.abs(x))
    xnorm = x/xpeak
    p = np.ones( xnorm.size ) * np.nan
    mask = np.abs(xnorm) > threshold
    p[mask] = np.angle(xnorm[mask])
    if unwrap:
        p[mask] = np.unwrap(p[mask])
    return p

class pulse:
    '''
    A pulse on a time/frequency grid.

    Parameters
    ----------
    t : array
        Uniform time grid (s), centred on 0.
    x : array
        Envelope in sqrt(W) (domain='Time') or its spectrum (domain='Freq').
    wavelength : float
        Reference wavelength of the grid (m); sets f0 = c/wavelength.
    frep : float
        Repetition rate (Hz).
    domain : {'Time', 'Freq'}
        Meaning of `x`. With 'Freq', `x` is a spectrum in FFT order with the
        same convention as `pulse.A`.

    Attributes
    ----------
    a, A : array
        Envelope (time order) and spectrum (FFT order), A = fft(fftshift(a)).
    t, dt, NFFT : time grid, spacing and number of points.
    f, Omega, df : relative frequencies (Hz), angular frequencies, spacing.
    f0, wl0 : reference frequency and wavelength.
    f_abs, wl : absolute frequencies and wavelengths, FFT order.
    frep : repetition rate.

    Pulses can be added, and multiplied by scalars or by other pulses
    (point-wise in time).
    '''
    def __init__(self, t, x, wavelength, frep, domain='Time'):

        if not 50e-9 <= wavelength <= 20e-6:
            print('Warning: Wavelength of %0.2f um '%(wavelength*1e6) + 
                  'for pulse looks outside usual range.')

        #Static atributes, they usually don't change
        self.t = t
        self.dt = t[1]-t[0] #Sample spacing
        self.NFFT = t.size
        
        #Relative frequencies
        self.f = fftfreq(self.NFFT, self.dt)
        self.Omega = 2*pi*self.f
        self.df = self.f[1]-self.f[0]
        
        #Reference frequency:
        self.wl0 = wavelength 
        self.f0 = c/wavelength
        
        #Repetition rate
        self.frep = frep
        
        #Absolute frequencies
        self.f_abs = self.f + self.f0
        self.wl = c/self.f_abs

        #Dynamic atributes, they change as pulse propagates
        if domain=='Time':
            self.a = x
            self.A = fft( fftshift(x), self.NFFT)
        elif domain=='Freq':
            self.A = x
            self.a = fftshift( ifft(x, self.NFFT) ) #inverse of A = fft(fftshift(a)) (#42)
        else:
            print('Wrong domain option. Options are "Time" or "Freq"')

        if np.amin(self.f_abs)<=0:
            print('Warning: negative absolute frequencies found. ' + 
                  'Considering reducing the number of points, ' +  
                  'or increasing the total time')   
            
    def __add__(self, pulse2):
        xsum = self.a + pulse2.a
        return pulse(self.t, xsum, self.wl0, self.frep)
    
    def __mul__(self, x):
        if np.isscalar(x):
            return pulse(self.t, x*self.a, self.wl0, self.frep)
        elif isinstance(x, pulse):
            return pulse(self.t, x.a * self.a, self.wl0, self.frep)
    
    def __rmul__(self, x):
        return self.__mul__(x)
    
    def update_td(self, a):
        '''Replace the time-domain envelope and recompute the spectrum.'''
        if np.size(self.t) == np.size(a):
            self.a = a
            self.A = fft( fftshift(a), self.NFFT)
        else:
            raise RuntimeError('Hmm, this pulse seems different. Cannot update.')

    def update_fd(self, A):
        '''Replace the spectrum (FFT order) and recompute the envelope.'''
        if np.size(self.t) == np.size(A):
            self.A = A
            self.a = fftshift( ifft(A, self.NFFT) )
        else:
            raise RuntimeError('Hmm, this pulse seems different. Cannot update.') 

    def energy_td(self):
        t = self.t
        x = self.a
        return energy_td(t, x)

    def energy_fd(self):
        t = self.t
        x = self.a
        return energy_fd(t, x)
    
    def get_psd(self):
        return get_psd(self.t, self.a, self.frep)
    
    def get_psd_dBm(self):
        return get_psd_dBm(self.t, self.a, self.frep)
    
    def spectrum(self, res):
        '''Power (dBm) in a resolution bandwidth `res` (Hz), like an OSA trace.'''
        # rbw = c/(self.wl**2)*res #to frequency
        rbw = res
        spectrum = 10*np.log10( self.get_psd()*rbw ) + 30 #to dBm
        return spectrum

    def time_center(self):
        return pulse_center(self.t, self.a)

    def width_FWHM(self, prominence=None):
        '''
        Intensity FWHM (s). Known issue: returns an array (one entry per peak)
        and takes the default prominence from the field, not the intensity (#44).
        '''
        if prominence is None:
            prominence = 0.25*np.amax(self.a)
        return FWHM(self.t, self.a, prominence=prominence)

    def plot_magsq(self, label='Pulse Intensity (W)', ax=None, t_unit='ps'):
        t = self.t
        x = self.a
        ax = plot_magsq(t, x, label, ax)
        return ax         

    def plot_PSD(self, ax=None, f_unit='um', dBm=True):
        t = self.t
        x = self.a
        frep = self.frep
        f0 = self.f0
        ax = plot_PSD(t, x, frep, f0, ax, f_unit, dBm)
        return ax
    
    def plot_spectrum(self, f_unit='um', res=None, ax=None):
        
        if res is None:
            res = self.df
        
        f = self.f_abs
        spectrum = self.spectrum(res)
        label = 'Power Spectrum (dBm)'
        
        if f_unit in ['um', 'nm']:
            ax = plot_vs_wavelength(c/f, spectrum, ylabel=label, ax=ax, wl_unit=f_unit)
        elif f_unit in ['GHz', 'THz', 'PHz']:
            ax = plot_vs_freq(f, spectrum, ylabel=label, ax=ax, f_unit=f_unit)
        
        return ax
    
    def apply_filter(self, f0, bw, type='ideal'):
        '''
        New pulse band-pass filtered around absolute frequency f0 with full
        bandwidth bw (Hz). `type` is 'ideal' or 'butterworth'.
        '''
        f_abs = self.f_abs
        X = self.A
        filtered, h = filter_signal(f_abs, X, f0, bw, type=type)
        return pulse(self.t, filtered, self.wl0, self.frep)

    def spectral_phase(self, unwrap=True, threshold=1e-3):
        return phase( self.A, unwrap=unwrap, threshold=threshold )
    
    def temporal_phase(self, unwrap=True, threshold=1e-3):
        return phase( self.a, unwrap=unwrap, threshold=threshold )

def test1():
    pass

if __name__ == '__main__':
    test1()
    