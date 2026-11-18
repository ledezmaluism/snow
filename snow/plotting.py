"""
Plotting helpers for pulses and propagation results.

Every function draws on `ax` if given (otherwise on a new figure) and returns
the axis. Frequencies and wavelengths are taken in FFT order, as they come
from a `Pulse`, and sorted for plotting.

    from snow import plotting
    plotting.plot_magsq(p.t, p.a, t_unit='fs')
    plotting.plot_evolution(result, domain='wavelength')

`Pulse.plot_magsq`, `Pulse.plot_PSD` and `Pulse.plot_spectrum` call these.
"""
import numpy as np
from numpy.fft import fftshift, fftfreq
from scipy.constants import c
import matplotlib.pyplot as plt

_T_UNITS = {'fs': 1e15, 'ps': 1e12, 'ns': 1e9}
_WL_UNITS = {'um': 1e6, 'nm': 1e9}
_F_UNITS = {'GHz': 1e-9, 'THz': 1e-12, 'PHz': 1e-15}


def _scale(unit, table, name):
    try:
        return table[unit]
    except KeyError:
        raise ValueError(f"{name} must be {', '.join(repr(u) for u in table)}, not {unit!r}") from None


def _axis(ax):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    return ax


def plot_vs_time(t, x, ylabel='', t_unit='ps', ax=None):
    '''Plot x versus time; t_unit is 'fs', 'ps' or 'ns'.'''
    t = t*_scale(t_unit, _T_UNITS, 't_unit')
    ax = _axis(ax)
    ax.plot(t, x)
    ax.grid(True)
    ax.set_xlabel('Time (' + t_unit + ')')
    ax.set_ylabel(ylabel)
    return ax


def plot_vs_wavelength(wl, x, ylabel='', wl_unit='um', ax=None):
    '''Plot x versus wavelength (FFT order); wl_unit is 'um' or 'nm'.'''
    wl = fftshift(wl*_scale(wl_unit, _WL_UNITS, 'wl_unit'))
    x = fftshift(x)
    ax = _axis(ax)
    ax.plot(wl, x)
    ax.grid(True)
    ax.set_xlabel('Wavelength (' + wl_unit + ')')
    ax.set_ylabel(ylabel)
    return ax


def plot_vs_freq(f, x, ylabel='', f_unit='THz', ax=None):
    '''Plot x versus frequency (FFT order); f_unit is 'GHz', 'THz' or 'PHz'.'''
    f = fftshift(f*_scale(f_unit, _F_UNITS, 'f_unit'))
    x = fftshift(x)
    ax = _axis(ax)
    ax.plot(f, x)
    ax.grid(True)
    ax.set_xlabel('Frequency (' + f_unit + ')')
    ax.set_ylabel(ylabel)
    return ax


def plot_vs_spectral(f_abs, x, ylabel='', f_unit='um', ax=None):
    '''Plot x versus wavelength ('um', 'nm') or frequency ('GHz', 'THz', 'PHz').'''
    if f_unit in _WL_UNITS:
        return plot_vs_wavelength(c/f_abs, x, ylabel=ylabel, ax=ax, wl_unit=f_unit)
    if f_unit in _F_UNITS:
        return plot_vs_freq(f_abs, x, ylabel=ylabel, ax=ax, f_unit=f_unit)
    raise ValueError(f"f_unit must be 'um', 'nm', 'GHz', 'THz' or 'PHz', not {f_unit!r}")


def plot_magsq(t, x, label='Pulse Intensity (W)', ax=None, t_unit='ps'):
    '''Plot |x|^2 versus time.'''
    return plot_vs_time(t, np.abs(x)**2, ylabel=label, ax=ax, t_unit=t_unit)


def plot_PSD(t, x, frep, f0=0, ax=None, f_unit='um', dBm=True):
    '''
    Plot the PSD of envelope x versus wavelength or frequency, in dBm/Hz or
    W/Hz. `f0` is the grid's reference frequency.
    '''
    from . import pulses
    f = fftfreq(t.size, t[1] - t[0]) + f0
    if dBm:
        psd = pulses.get_psd_dBm(t, x, frep)
        label = 'Power Spectral Density (dBm / Hz)'
    else:
        psd = pulses.get_psd(t, x, frep)
        label = 'Power Spectral Density (W / Hz)'
    return plot_vs_spectral(f, psd, ylabel=label, f_unit=f_unit, ax=ax)


def plot_evolution(result, domain='time', unit=None, dB=True, dynamic_range=40,
                   z_unit='mm', ax=None):
    '''
    Map of the pulse along the propagation, from the snapshots of a
    `PropagationResult` (propagate with ``z_save=...``).

    Parameters
    ----------
    result : nlo.PropagationResult
    domain : {'time', 'wavelength', 'frequency'}
        Plot |a(t)|^2 or the energy spectral density.
    unit : str, optional
        Axis unit: 'fs'/'ps'/'ns', 'um'/'nm' or 'GHz'/'THz'/'PHz'. Default
        'ps', 'um' or 'THz'.
    dB : bool
        Colour scale in dB relative to the maximum, spanning `dynamic_range`.
    z_unit : {'mm', 'um', 'm'}
    '''
    if result.z.size == 0:
        raise ValueError('the result has no snapshots; propagate with z_save=[...]')
    p = result.pulse
    zscale = {'m': 1, 'mm': 1e3, 'um': 1e6}[z_unit]
    if domain == 'time':
        unit = unit or 'ps'
        x = p.t*_scale(unit, _T_UNITS, 'unit')
        data = np.abs(result.fields)**2
        xlabel = f'Time ({unit})'
    elif domain in ('wavelength', 'frequency'):
        S = fftshift(np.abs(result.spectra)**2, axes=1)
        f = fftshift(p.f_abs)
        if domain == 'wavelength':
            unit = unit or 'um'
            x = c/f*_scale(unit, _WL_UNITS, 'unit')
            xlabel = f'Wavelength ({unit})'
        else:
            unit = unit or 'THz'
            x = f*_scale(unit, _F_UNITS, 'unit')
            xlabel = f'Frequency ({unit})'
        data = S
    else:
        raise ValueError(f"domain must be 'time', 'wavelength' or 'frequency', not {domain!r}")

    if dB:
        with np.errstate(divide='ignore'):
            data = 10*np.log10(data/np.max(data))
        vmin, vmax = -dynamic_range, 0
        clabel = 'Relative power (dB)'
    else:
        vmin = vmax = None
        clabel = 'Power (a.u.)'
    order = np.argsort(x)
    ax = _axis(ax)
    mesh = ax.pcolormesh(x[order], result.z*zscale, data[:, order], shading='auto',
                         vmin=vmin, vmax=vmax)
    ax.figure.colorbar(mesh, ax=ax, label=clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'z ({z_unit})')
    return ax
