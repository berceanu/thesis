import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
# for colorbar stuff
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

from scipy import optimize
from phcpy.phcpy2c import py2c_set_seed
from phcpy.solver import solve
from phcpy.solutions import strsol2dict
import ConfigParser
import sys

def read_parameters(filename):
    config = ConfigParser.RawConfigParser()
    try:
        config.readfp(open(filename))
    except IOError:
        raise IOError('cannot find parameters.ini, exiting')
    return config

param = read_parameters("/home/berceanu/notebooks/OPODrag/parameters.ini")
param_set = sys.argv[1]

omega_X = param.getfloat(param_set, "omega_X")
omega_C0 = param.getfloat(param_set, "omega_C0")
kz = param.getfloat(param_set, "kz")
Omega_R = param.getfloat(param_set, "Omega_R")
gamma_X = param.getfloat(param_set, "gamma_X")
gamma_C = param.getfloat(param_set, "gamma_C")
k_pmpx = param.getfloat(param_set, "k_pmpx")
k_pmpy = param.getfloat(param_set, "k_pmpy")
k_sigx = param.getfloat(param_set, "k_sigx")
k_sigy = param.getfloat(param_set, "k_sigy")
omega_pmp = param.getfloat(param_set, "omega_pmp")
ip_chosen = param.getfloat(param_set, "ip_chosen")
gv = param.getfloat(param_set, "gv")

nkx = param.getint(param_set, "nkx")
nky = param.getint(param_set, "nky")
ipx_start = param.getfloat(param_set, "ipx_start")
ipx_end = param.getfloat(param_set, "ipx_end")
kxl = param.getfloat(param_set, "kxl")
kxr = param.getfloat(param_set, "kxr")
kyl = param.getfloat(param_set, "kyl")
kyr = param.getfloat(param_set, "kyr")
eigs_threshold = param.getfloat(param_set, "eigs_threshold")
stride_r = param.getint(param_set, "stride_r")
stride_c = param.getint(param_set, "stride_c")
delta_k = param.getfloat(param_set, "delta_k")
ipstabini = param.getfloat(param_set, "ipstabini")
ipstabfin = param.getfloat(param_set, "ipstabfin")


k_idlx = 2 * k_pmpx - k_sigx
k_idly = 2 * k_pmpy - k_sigy

ks = ("{0:.3f}".format(k_sigx)).replace(".", "_")

gp = gamma_C + (1 / np.sqrt(1 + (Omega_R / ((0.5 * ((omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx ** 2 + k_pmpy ** 2) / kz) ** 2)) + omega_X) - 0.5 * np.sqrt(((omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx ** 2 + k_pmpy ** 2) / kz) ** 2)) - omega_X) ** 2 + 4 * Omega_R ** 2)) - (omega_C0 * np.sqrt(1 + (np.sqrt(k_pmpx ** 2 + k_pmpy ** 2) / kz) ** 2)))) ** 2)) ** 2 * (gamma_X - gamma_C)
omega_p_chosen = (omega_pmp - omega_X) / gp

side_k = nkx * delta_k / 2
side_r = np.pi / delta_k
delta_r = np.pi / side_k

x = y = np.arange(-side_r, side_r, delta_r)
kx = ky = np.arange(-side_k, side_k, delta_k)

KX, KY = np.meshgrid(kx, ky)
X, Y = np.meshgrid(x, y)

ipx = np.linspace(ipx_start, ipx_end, 30)

mpl.rcParams.update({'font.size': 28, 'font.family': 'serif'})

x_label_k = r'$k_x - k_n\,[\mu m^{-1}]$'
y_label_k = r'$k_y\,[\mu m^{-1}]$'
x_label_i = r'$x\,[\mu m]$'
y_label_i = r'$y\,[\mu m]$'

letter_spi = ['s', 'p', 'i']

title_k = []
for idx in range(3):
    title_k.append(r'$g \left|\tilde{\Psi}_{LP}^{' + letter_spi[idx] +
                   r'}\left(k+k_{' + letter_spi[idx] +
                   r'}\right)\right|^{2} [\gamma_p \mu m^4]$')

title_i = []
for idx in range(3):
    title_i.append(r'$I^' + letter_spi[idx] + r'$')

color_spi = ['blue', 'red', 'green']
marker_spi = ['^', 'o', 'v']

momentum_spi = np.array([[k_sigx, k_sigy],
                         [k_pmpx, k_pmpy],
                         [k_idlx, k_idly]])


def null(A, eps=1e-10):
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T


def index_mom(mom):
    return int(np.floor((mom + side_k) / delta_k))


def enC(kx, ky):
    return (omega_C0 * np.sqrt(1 + (np.sqrt(kx ** 2 + ky ** 2) / kz) ** 2)
            - omega_X) / gp

def enLP(kx, ky):
    return 0.5 * enC(kx, ky) - 0.5 * np.sqrt(enC(kx, ky) ** 2
                                             + 4 * Omega_R ** 2 / gp ** 2)

def hopf_x(kx, ky):
    return 1 / np.sqrt(1 + ((Omega_R / gp)
                            / (enLP(kx, ky) - enC(kx, ky))) ** 2)

def blue_enLP(kx, ky):
    return enLP(kx, ky) + 2 * hopf_x(kx, ky) ** 2 *\
        (ni_chosen + np_chosen + ns_chosen)

def hopf_c(kx, ky):
    return -1 / np.sqrt(1 + ((enLP(kx, ky) - enC(kx, ky))
                             / (Omega_R / gp)) ** 2)


def gamma(kx, ky):
    return (gamma_C + hopf_x(kx, ky) ** 2 * (gamma_X - gamma_C)) / gp


kxL, kxR = index_mom(kxl), index_mom(kxr)
kyL, kyR = index_mom(kyl), index_mom(kyr)

es = enLP(k_sigx, k_sigy)
ep = enLP(k_pmpx, k_pmpy)
ei = enLP(k_idlx, k_idly)

gs = gamma(k_sigx, k_sigy)
gi = gamma(k_idlx, k_idly)

xs = hopf_x(k_sigx, k_sigy)
xp = hopf_x(k_pmpx, k_pmpy)
xi = hopf_x(k_idlx, k_idly)

cs = hopf_c(k_sigx, k_sigy)
cp = hopf_c(k_pmpx, k_pmpy)
ci = hopf_c(k_idlx, k_idly)

alpha = (xi ** 2 / xs ** 2) * (gs / gi)

def n_hom_mf(ips):
    return np.array([optimize.brentq(lambda n: ((ep - omega_p_chosen
                                                 + xp ** 2 * n) ** 2 + 1./4.) * n - xp ** 4 * ip, 0, 3)
                     for ip in ips])


def L(kx, ky):
    return np.array(
        [[-omega_s_chosen + enLP(
            k_sigx + kx, k_sigy + ky) - 1j * 1 / 2 * gamma(
            k_sigx + kx, k_sigy + ky) + 2 * (
            ni_chosen + np_chosen + ns_chosen) * hopf_x(
            k_sigx + kx, k_sigy + ky) ** 2, 2 * (
            p * np.conjugate(
                i) + s * np.conjugate(
                p)) * hopf_x(
            k_pmpx + kx, k_pmpy + ky) * hopf_x(
            k_sigx + kx, k_sigy + ky), 2 * s * np.conjugate(
            i) * hopf_x(
            k_idlx + kx, k_idly + ky) * hopf_x(
            k_sigx + kx, k_sigy + ky), s ** 2 * hopf_x(
            k_sigx - kx, k_sigy - ky) * hopf_x(
            k_sigx + kx, k_sigy + ky), 2 * p * s * hopf_x(
            k_pmpx - kx, k_pmpy - ky) * hopf_x(
            k_sigx + kx, k_sigy + ky), (
            p ** 2 + 2 * i * s) * hopf_x(
            k_idlx - kx, k_idly - ky) * hopf_x(
            k_sigx + kx, k_sigy + ky)],
         [2 * (i * np.conjugate(p) + p * np.conjugate(s)) * hopf_x(k_pmpx + kx, k_pmpy + ky) * hopf_x(k_sigx + kx, k_sigy + ky), -omega_p_chosen + enLP(k_pmpx + kx, k_pmpy + ky) - 1j * 1 / 2 * gamma(k_pmpx + kx, k_pmpy + ky) + 2 * (ni_chosen + np_chosen + ns_chosen) * hopf_x(k_pmpx + kx, k_pmpy + ky) ** 2, 2 * (p * np.conjugate(i) + s * np.conjugate(p))
          * hopf_x(
              k_idlx + kx, k_idly + ky) * hopf_x(
              k_pmpx + kx, k_pmpy + ky), 2 * p * s * hopf_x(
              k_sigx - kx, k_sigy - ky) * hopf_x(
              k_pmpx + kx, k_pmpy + ky), (
              p ** 2 + 2 * i * s) * hopf_x(
              k_pmpx - kx, k_pmpy - ky) * hopf_x(
              k_pmpx + kx, k_pmpy + ky), 2 * i * p * hopf_x(
              k_idlx - kx, k_idly - ky) * hopf_x(
              k_pmpx + kx, k_pmpy + ky)],
         [2 * i * np.conjugate(s) * hopf_x(k_idlx + kx, k_idly + ky) * hopf_x(k_sigx + kx, k_sigy + ky), 2 * (i * np.conjugate(p) + p * np.conjugate(s)) * hopf_x(k_idlx + kx, k_idly + ky) * hopf_x(k_pmpx + kx, k_pmpy + ky), -omega_i_chosen + enLP(k_idlx + kx, k_idly + ky) - 1j * 1 / 2 * gamma(k_idlx + kx, k_idly + ky) + 2 *
          (ni_chosen + np_chosen + ns_chosen) * hopf_x(k_idlx + kx, k_idly + ky) ** 2, (p ** 2 + 2 * i * s) * hopf_x(k_sigx - kx, k_sigy - ky) * hopf_x(k_idlx + kx, k_idly + ky), 2 * i * p * hopf_x(k_pmpx - kx, k_pmpy - ky) * hopf_x(k_idlx + kx, k_idly + ky), i ** 2 * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_idlx + kx, k_idly + ky)],
         [-(np.conjugate(s) ** 2 * hopf_x(k_sigx - kx, k_sigy - ky) * hopf_x(k_sigx + kx, k_sigy + ky)), -2 * np.conjugate(p) * np.conjugate(s) * hopf_x(k_sigx - kx, k_sigy - ky) * hopf_x(k_pmpx + kx, k_pmpy + ky), -((np.conjugate(p) ** 2 + 2 * np.conjugate(i) * np.conjugate(s)) * hopf_x(k_sigx - kx, k_sigy - ky) * hopf_x(k_idlx + kx, k_idly + ky)), omega_s_chosen -
          enLP(
              k_sigx - kx, k_sigy - ky) - 1j * 1 / 2 * gamma(
              k_sigx - kx, k_sigy - ky) - 2 * (
              ni_chosen + np_chosen + ns_chosen) * hopf_x(
              k_sigx - kx, k_sigy - ky) ** 2, -2 * (
              i * np.conjugate(
                  p) + p * np.conjugate(
                  s)) * hopf_x(
              k_pmpx - kx, k_pmpy - ky) * hopf_x(
              k_sigx - kx, k_sigy - ky), -2 * i * np.conjugate(
              s) * hopf_x(
              k_idlx - kx, k_idly - ky) * hopf_x(
              k_sigx - kx, k_sigy - ky)],
         [-2 * np.conjugate(p) * np.conjugate(s) * hopf_x(k_pmpx - kx, k_pmpy - ky) * hopf_x(k_sigx + kx, k_sigy + ky), -((np.conjugate(p) ** 2 + 2 * np.conjugate(i) * np.conjugate(s)) * hopf_x(k_pmpx - kx, k_pmpy - ky) * hopf_x(k_pmpx + kx, k_pmpy + ky)), -2 * np.conjugate(i) * np.conjugate(p) * hopf_x(k_pmpx - kx, k_pmpy - ky) * hopf_x(k_idlx + kx, k_idly + ky), -2 * (p * np.conjugate(i) + s * np.conjugate(p))
          * hopf_x(
              k_pmpx - kx, k_pmpy - ky) * hopf_x(
              k_sigx - kx, k_sigy - ky), omega_p_chosen - enLP(
              k_pmpx - kx, k_pmpy - ky) - 1j * 1 / 2 * gamma(
              k_pmpx - kx, k_pmpy - ky) - 2 * (
              ni_chosen + np_chosen + ns_chosen) * hopf_x(
              k_pmpx - kx, k_pmpy - ky) ** 2, -2 * (
              i * np.conjugate(
                  p) + p * np.conjugate(
                  s)) * hopf_x(
              k_idlx - kx, k_idly - ky) * hopf_x(
              k_pmpx - kx, k_pmpy - ky)],
         [-((np.conjugate(p) ** 2 + 2 * np.conjugate(i) * np.conjugate(s)) * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_sigx + kx, k_sigy + ky)), -2 * np.conjugate(i) * np.conjugate(p) * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_pmpx + kx, k_pmpy + ky), -(np.conjugate(i) ** 2 * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_idlx + kx, k_idly + ky)), -2 * s * np.conjugate(i) * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_sigx - kx, k_sigy - ky), -2 * (p * np.conjugate(i) + s * np.conjugate(p)) * hopf_x(k_idlx - kx, k_idly - ky) * hopf_x(k_pmpx - kx, k_pmpy - ky), omega_i_chosen - enLP(k_idlx - kx, k_idly - ky) - 1j * 1 / 2 * gamma(k_idlx - kx, k_idly - ky) - 2 * (ni_chosen + np_chosen + ns_chosen) * hopf_x(k_idlx - kx, k_idly - ky) ** 2]])


def L_mats(K_X, K_Y, n_kx, n_ky):
    mats = L(K_X, K_Y)
    new_mats = np.transpose(mats, (2, 3, 0, 1))
    new_mats.shape = (n_kx * n_ky, 6, 6)
    return new_mats


def eigL_mats(mats, n_kx, n_ky):
    res = np.linalg.eigvals(mats)
    res.shape = (n_ky, n_kx, 6)
    return res


def fd(kx, ky):
    return np.array([cs / xs * hopf_c(kx + k_sigx, ky + k_sigy) * s,
                     cp / xp * hopf_c(kx + k_pmpx, ky + k_pmpy) * p,
                     ci / xi * hopf_c(kx + k_idlx, ky + k_idly) * i,
                    -cs / xs * hopf_c(k_sigx - kx, k_sigy - ky) *
                     np.conjugate(s),
                    -cp / xp * hopf_c(k_pmpx - kx, k_pmpy - ky) *
                     np.conjugate(p),
                    -ci / xi * hopf_c(k_idlx - kx, k_idly - ky)
                    * np.conjugate(i)])


def fd_mats(K_X, K_Y, n_kx, n_ky):
    res = fd(K_X, K_Y)
    new_res = np.transpose(res, (1, 2, 0))
    new_res.shape = (n_kx * n_ky, 6)
    return new_res


def bog_coef_mats(mats, fds, n_kx, n_ky):
    res = np.linalg.solve(mats, -fds)
    res.shape = (n_ky, n_kx, 6)
    return res


def eqs(ip):

    #x1 -> omega_s
    #x2 -> ns
    #x3 -> np
       
    eq1 = "{0:+.16f}*x1{1:+.16f}*x2{2:+.16f}*x3{3:+.16f};"          .format((-gi - gs)/(gi*xs**2), -alpha**2 + 1, -2*alpha + 2, (-ei*gs + es*gi + 2*gs*omega_p_chosen)/(gi*xs**2))
    eq2 = "{0:.16f}*x1^2{1:+.16f}*x1*x2{2:+.16f}*x1*x3{3:+.16f}*x1{4:+.16f}*x2^2{5:+.16f}*x2*x3{6:+.16f}*x2{7:+.16f}*x3^2{8:+.16f}*x3{9:+.16f};"          .format(xs**(-4), (-4*alpha - 2)/xs**2, -4/xs**2, -2*es/xs**4, 4*alpha**2 + 4*alpha + 1, 8*alpha + 4, (4*alpha*es + 2*es)/xs**2, -alpha + 4, 4*es/xs**2, (4*es**2 + gs**2)/(4*xs**4))
    eq3 = "{0:.16f}*x1^2*x2^2{1:+.16f}*x1*x2^3{2:+.16f}*x1*x2^2*x3{3:+.16f}*x1*x2^2{4:+.16f}*x1*x2*x3^2{5:+.16f}*x1*x2*x3{6:+.16f}*x2^4{7:+.16f}*x2^3*x3{8:+.16f}*x2^3{9:+.16f}*x2^2*x3^2{10:+.16f}*x2^2*x3{11:+.16f}*x2^2{12:+.16f}*x2*x3^3{13:+.16f}*x2*x3^2{14:+.16f}*x2*x3{15:+.16f}*x3^4{16:+.16f}*x3^3{17:+.16f}*x3^2{18:+.16f}*x3;"          .format(4.0/xs**4, 1.0*(-16.0*alpha - 8.0)/xs**2, 1.0*(8.0*alpha - 8.0)/xs**2, -8.0*es/xs**4, 4.0/xs**2, 1.0*(4.0*ep - 4.0*omega_p_chosen)/(xp**2*xs**2), 16.0*alpha**2 + 16.0*alpha + 4.0, -16.0*alpha**2 + 8.0*alpha + 8.0, 1.0*(16.0*alpha*es + 8.0*es)/xs**2, 4.0*alpha**2 - 16.0*alpha, 1.0*(-8.0*alpha*ep*xs**2 - 8.0*alpha*es*xp**2 + 8.0*alpha*omega_p_chosen*xs**2 - 4.0*ep*xs**2 + 8.0*es*xp**2 + 4.0*omega_p_chosen*xs**2)/(xp**2*xs**2), 1.0*(4.0*es**2 + 1.0*gs**2)/xs**4, 4.0*alpha - 4.0, 1.0*(4.0*alpha*ep*xs**2 - 4.0*alpha*omega_p_chosen*xs**2 - 4.0*ep*xs**2 - 4.0*es*xp**2 + 4.0*omega_p_chosen*xs**2)/(xp**2*xs**2), 1.0*(-4.0*ep*es + 4.0*es*omega_p_chosen + 1.0*gs)/(xp**2*xs**2), 1.00000000000000, 1.0*(2.0*ep - 2.0*omega_p_chosen)/xp**2, 1.0*(1.0*ep**2 - 2.0*ep*omega_p_chosen + 1.0*omega_p_chosen**2 + 0.25)/xp**4, -1.0*ip)
    f = [eq1, eq2, eq3]
    
    #seeding random number generator for reproducible results
    py2c_set_seed(130683)
    
    s = solve(f,silent=True)
    ns = len(s) #number of solutions

    R = []
    for k in range(ns):
        d = strsol2dict(s[k])
        x123 = np.array([d['x1'], d['x2'], d['x3']])
        real = np.real(x123)
        imag = np.fabs(np.imag(x123))
        sol = np.empty((real.size + imag.size,), dtype=real.dtype)
        sol[0::2] = real
        sol[1::2] = imag
        if np.allclose(imag, np.zeros(3)):
            if real[1] > 1e-7:
                R.append((sol[0], sol[2], sol[4], ip))

    return R


solutions = []
for pmp_int in ipx:
    solutions.append(eqs(pmp_int))


solutions_v2 = [sol for sol in solutions if sol != []]


nsnpip = []
for idx in range(1, 4):
    nsnpip.append(np.array([tple[idx]
                  for lst in solutions_v2 for tple in lst]))

ipstart = nsnpip[2][0]
print ipstart

fig_mf, ax = plt.subplots()

ax.plot(ipx[ipx < ipstart], n_hom_mf(ipx[ipx < ipstart]), color='black')
ax.plot(ipx[ipx >= ipstart], n_hom_mf(ipx[ipx >= ipstart]), linestyle='dashed', color='black')

for idx in [0, 1]:
    ax.plot(nsnpip[2], nsnpip[idx], linestyle='none',
            marker=marker_spi[idx], markerfacecolor=color_spi[idx])

ax.plot(nsnpip[2], alpha * nsnpip[0], linestyle='none',
        marker=marker_spi[2], markerfacecolor=color_spi[2])

ax.axvline(x=ip_chosen, color='k', ls='dashed')
ax.fill([ipstabini,ipstabfin,ipstabfin,ipstabini], [0,0,1.8,1.8], 'gray', alpha=0.5, edgecolor='k')

ax.set_xlim(ipx[0], ipx[-1])
ax.set_xlabel(r'$I_p [\gamma_p^3]$')
ax.set_ylabel(r'$\epsilon_s, \epsilon_p, \epsilon_i [\gamma_p]$')
if param_set == "ks0.0":
    ax.xaxis.set_ticks([0, 4, 8, 12, 16])
ax.yaxis.set_ticks([0.4, 0.8, 1.2, 1.6])
fig_mf.savefig('fig_mf_ks_{0:s}'.format(ks), bbox_inches='tight')

[(omega_s_chosen, ns_chosen, np_chosen, ip_chosen)] = eqs(ip_chosen)
omega_i_chosen = 2 * omega_p_chosen - omega_s_chosen
ni_chosen = alpha * ns_chosen


energy_spi = [omega_s_chosen, omega_p_chosen, omega_i_chosen]


p = 1 / np.sqrt(ip_chosen) * (2 / xs ** 2 * (es - 1j * gs / 2 - omega_s_chosen) * ns_chosen + 2 * (2 * np_chosen + ns_chosen + 2 * ni_chosen)
                              * ns_chosen - (1 / xp ** 2 * (ep + 1j * 1 / 2 - omega_p_chosen) * np_chosen + (np_chosen + 2 * ns_chosen + 2 * ni_chosen) * np_chosen))

pr = p.real
pi = p.imag


matSI = np.array(
    [[2 * ni_chosen + 2 * np_chosen + ns_chosen + (
        es - omega_s_chosen) / xs ** 2, gs / (
        2. * xs ** 2), -pi ** 2 + pr ** 2, 2 * pi * pr],
     [-gs / (2. * xs ** 2), 2 * ni_chosen + 2 * np_chosen + ns_chosen +
      (es - omega_s_chosen) / xs ** 2, 2 * pi * pr, pi ** 2 - pr ** 2],
     [-pi ** 2 + pr ** 2, 2 * pi * pr, ni_chosen + 2 * np_chosen +
      2 * ns_chosen + (ei - omega_i_chosen) / xi ** 2, gi / (2. * xi ** 2)],
     [2 * pi * pr, pi ** 2 - pr ** 2, -gi / (2. * xi ** 2), ni_chosen + 2 * np_chosen + 2 * ns_chosen + (ei - omega_i_chosen) / xi ** 2]])


norm = ns_chosen + ni_chosen
N = null(matSI) * np.sqrt(norm)

[sr, si, ir, ii] = N[:, 0]
s = sr + 1j * si
i = ir + 1j * ii

LP = enLP(KX, KY)
BLP = blue_enLP(KX, KY)
G = gamma(KX, KY) / 2

fig_lp, ax = plt.subplots()
ax.plot(KX[nky / 2, kxL:kxR], BLP[nky / 2, kxL:kxR], 'k-.')
ax.fill_between(
    KX[nky / 2, kxL:kxR], BLP[nky / 2, kxL:kxR] - G[nky / 2, kxL:kxR] / 2,
    BLP[nky / 2, kxL:kxR] + G[nky / 2, kxL:kxR] / 2, alpha=0.5)
for idx in range(3):
    ax.scatter(momentum_spi[idx, 0], energy_spi[idx], c=color_spi[idx], marker=marker_spi[idx])
ax.axhline(y=ep + 0.5 * np.sqrt(3), color='r', ls='dashed')
ax.set_title('blue-shifted LP dispersion')
ax.set_xlim(kxl, kxr)
ax.set_xlabel(x_label_k)
ax.set_ylabel(r'$\epsilon-\omega_X[\gamma_p]$')
fig_lp.savefig('fig_lp_ks_{0:s}'.format(ks), bbox_inches='tight')

indices_si_rows = [0, 0, 2, 2, 3, 3, 5, 5]
indices_si_cols = [2, 5, 0, 3, 2, 5, 0, 3]
indices_ps_rows = [0, 0, 1, 1, 3, 3, 4, 4]
indices_ps_cols = [1, 4, 0, 3, 1, 4, 0, 3]
indices_pi_rows = [1, 1, 2, 2, 4, 4, 5, 5]
indices_pi_cols = [2, 5, 1, 4, 2, 5, 1, 4]


matsL = L_mats(KX, KY, nkx, nky)
eigs = eigL_mats(matsL, nkx, nky)

matsL_diag = np.copy(matsL)
matsL_diag[:, indices_si_rows, indices_si_cols] = complex(0, 0)
matsL_diag[:, indices_ps_rows, indices_ps_cols] = complex(0, 0)
matsL_diag[:, indices_pi_rows, indices_pi_cols] = complex(0, 0)
eigs_diag = eigL_mats(matsL_diag, nkx, nky)


y_points, x_points, eig_indices = np.where(np.abs(eigs.real) <= eigs_threshold)
y_points_diag, x_points_diag, eig_indices_diag = np.where(
    np.abs(eigs_diag.real) <= 0.05)

phi_gold = 1.618

fig_excitation, ax = plt.subplots(1, 1, figsize=(6*phi_gold, 6))
for idx in range(3):
#    axes[0].plot(
#        KX[nky / 2, kxL:kxR], eigs[nky / 2,
#                                   kxL:kxR, idx].imag, linestyle='none',
#        marker='o', markerfacecolor='black', markersize=4)
    ax.plot(
        KX[nky / 2, kxL:kxR], eigs[nky / 2,
                                   kxL:kxR, idx].real, linestyle='none',
        marker='o', markerfacecolor='black', markersize=2)
for idx in range(3, 6):
#    axes[0].plot(
#        KX[nky / 2, kxL:kxR], eigs[nky / 2,
#                                   kxL:kxR, idx].imag, linestyle='none',
#        marker='o', markerfacecolor='black', markersize=4)
    ax.plot(
        KX[nky / 2, kxL:kxR], eigs[nky / 2,
                                   kxL:kxR, idx].real, linestyle='none',
        marker='o', markerfacecolor='black', markersize=2)
#for ax in [0, 1]:
#    axes[ax].axhline(y=0, color='black', ls='dashed')
    #axes[ax].axvline(x=0, color='black', ls='dashed')
ax.axhline(y=0, color='black', ls='dashed')
#axes[0].set_ylabel(r'$\Im{(\omega)}[\gamma_p]$')
ax.set_xlabel(x_label_k)
#axes[0].set_xticklabels([])
#axes[0].set_xlim(kx[kxL], kx[kxR])
ax.set_xlim(kx[kxL], kx[kxR])
#axes[0].xaxis.set_ticks([-6, -3, 0, 3, 6])
#axes[0].yaxis.set_ticks([-1.2, -0.8, -0.4, 0])
ax.xaxis.set_ticks([-6, -3, 0, 3, 6])
ax.set_ylim(-5, 5)
ax.set_ylabel(r'$\Re{(\omega)}[\gamma_p]$')
fig_excitation.savefig('fig_excitation_ks_{0:s}'.format(ks), bbox_inches='tight')

vectfd = fd_mats(KX, KY, nkx, nky)
bcoef = bog_coef_mats(matsL, vectfd, nkx, nky)
bcoef_diag = bog_coef_mats(matsL_diag, vectfd, nkx, nky)

matsL_min = -np.fft.fftshift(matsL, axes=(1, 2))
vectfd_min = -np.fft.fftshift(vectfd, axes=(1,))
bcoef_conj_mink = bog_coef_mats(matsL_min, vectfd_min, nkx, nky)

matsL_min_diag = -np.fft.fftshift(matsL_diag, axes=(1, 2))
bcoef_conj_mink_diag = bog_coef_mats(matsL_min_diag, vectfd_min, nkx, nky)


psi_k = gv / 2 * (bcoef[:, :, 0:3] + bcoef_conj_mink[:, :, 3:6])
psi_k_diag = gv / 2 * (bcoef_diag[:, :, 0:3] + bcoef_conj_mink_diag[:, :, 3:6])


[imax, jmax] = np.unravel_index(
    np.argmax(np.abs(psi_k[:, :, 0])), psi_k[:, :, 0].shape)


l1 = psi_k[imax - 2:imax + 3, jmax - 2, :]
l2 = psi_k[imax - 2:imax + 3, jmax + 2, :]
l3 = psi_k[imax - 2, jmax - 1:jmax + 2, :]
l4 = psi_k[imax + 2, jmax - 1:jmax + 2, :]

l = np.concatenate((l1, l2, l3, l4))
averages = np.mean(l, axis=0)
psi_k[imax - 1:imax + 2, jmax - 1:jmax + 2, :] = averages


psi_k[nky / 2, nkx / 2, :] += np.sqrt(nkx * nky) * \
    np.array([s / xs, p / xp, i / xi])
psi_k_diag[nky / 2, nkx / 2, :
           ] += np.sqrt(nkx * nky) * np.array([s / xs, p / xp, i / xi])

res_k = np.log10(np.abs(psi_k) ** 2)  # logscale

fig_mom_S, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(res_k[kyL:kyR, kxL:kxR, 0],
                 cmap=cm.gray, origin=None, extent=[kx[kxL], kx[kxR], ky[kyL], ky[kyR]])
ax.set_ylabel(y_label_k)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.xaxis.set_ticks([-6, -3, 0, 3, 6])
ax.yaxis.set_ticks([-6, -3, 0, 3, 6])
ax.set_xticklabels([])
fig_mom_S.savefig('fig_mom_ks_{0:s}_{1:s}'.format(ks, letter_spi[0]), bbox_inches='tight')

fig_mom_P, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(res_k[kyL:kyR, kxL:kxR, 1],
                 cmap=cm.gray, origin=None, extent=[kx[kxL], kx[kxR], ky[kyL], ky[kyR]])
ax.set_ylabel(y_label_k)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.xaxis.set_ticks([-6, -3, 0, 3, 6])
ax.yaxis.set_ticks([-6, -3, 0, 3, 6])
ax.set_xticklabels([])
fig_mom_P.savefig('fig_mom_ks_{0:s}_{1:s}'.format(ks, letter_spi[1]), bbox_inches='tight')

fig_mom_I, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(res_k[kyL:kyR, kxL:kxR, 2],
                 cmap=cm.gray, origin=None, extent=[kx[kxL], kx[kxR], ky[kyL], ky[kyR]])
ax.set_ylabel(y_label_k)
ax.set_xlabel(x_label_k)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.xaxis.set_ticks([-6, -3, 0, 3, 6])
ax.yaxis.set_ticks([-6, -3, 0, 3, 6])
fig_mom_I.savefig('fig_mom_ks_{0:s}_{1:s}'.format(ks, letter_spi[2]), bbox_inches='tight')

res_k_diag = np.log10(np.abs(psi_k_diag) ** 2)  # logscale


kl3d, kr3d = -5, 5
kxl3d, kxr3d = index_mom(kl3d), index_mom(kr3d)

######
######
fig_3d = plt.figure(figsize=(10, 8))
ax = fig_3d.add_subplot(111, projection='3d')

tmp_planes = ax.zaxis._PLANES 
ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                     tmp_planes[0], tmp_planes[1], 
                     tmp_planes[4], tmp_planes[5])
view_1 = (25, -135)
view_2 = (25, -45)
init_view = view_2
ax.view_init(*init_view)


ax.plot_surface(
    KX[kxl3d:kxr3d, kxl3d:kxr3d], KY[kxl3d:kxr3d,
                                     kxl3d:kxr3d], BLP[
        kxl3d:kxr3d, kxl3d:kxr3d],
    rstride=stride_r, cstride=stride_c, alpha=0.4)
for idx in range(3):
    ax.plot(
        kx[x_points] + momentum_spi[idx, 0], ky[y_points] +
        momentum_spi[idx, 1], energy_spi[idx],
        linestyle='none', marker=marker_spi[idx], markerfacecolor=color_spi[idx],
        markersize=5)

fig_3d.tight_layout()

ax.set_xlim(kl3d, kr3d)
ax.set_ylim(kl3d, kr3d)

ax.set_xlabel('\n' + r'$k_x[\mu m^{-1}]$', linespacing=2)
ax.set_ylabel('\n' + r'$k_y[\mu m^{-1}]$', linespacing=2)

ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel(r'$\omega-\omega_X(0)[\gamma_p]$', rotation=90, labelpad=10)
if param_set == "ks0.0":
    ax.zaxis.set_ticks([-20, -10, 0])
else:
    ax.zaxis.set_ticks([-40, -20, 0])

fig_3d.savefig('fig_3d_ks_{0:s}'.format(ks), bbox_inches='tight')
#####
#####

psi_r = np.fft.fftshift(
    np.fft.ifft2(np.sqrt(nkx * nky) * psi_k, axes=(0, 1)), axes=(0, 1))
res_r = np.abs(psi_r) ** 2 / \
    np.array([ns_chosen / xs ** 2, np_chosen / xp ** 2, ni_chosen / xi ** 2])

rango = 400 / (1024/nkx)


#use np.clip for truncating the Z range
fig_real_S, ax = plt.subplots(1, 1, figsize=(5, 5))
im = ax.imshow(res_r[rango:-rango, rango:-rango, 0], cmap=cm.gray,
                 origin=None, extent=[x[rango], x[-rango], y[rango], y[-rango]])
ax.set_ylabel(y_label_i)
ax.set_xticklabels([])

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
ticks_at = np.linspace(np.amin(res_r[rango:-rango, rango:-rango, 0]),
                       np.amax(res_r[rango:-rango, rango:-rango, 0]), 5) 
cb = fig_real_S.colorbar(im, cax=cax, ticks=ticks_at, format='%1.2f')
cb.ax.tick_params(labelsize=20)

fig_real_S.savefig('fig_real_ks_{0:s}_{1:s}'.format(ks, letter_spi[0]), bbox_inches='tight')


ls = ['--','-',':']
col = ['blue', 'red', 'green']
leg = ['signal', 'pump', 'idler']
mrk = ['^', 'o', 'v']
fig_ranges, ax = plt.subplots(1, 1, figsize=(6*phi_gold, 6))
for idx in range(3):
    ax.plot(x[rango:-rango], res_r[nky/2, rango:-rango, idx], linestyle=ls[idx], color=col[idx],
            linewidth=3.0, label=leg[idx])
ax.axhline(y=1, color='black', ls='dashed')
#ax.axvline(x=0, color='black', ls='dashed')

ax.set_xlabel(x_label_i)

if (ks=='0_000') or (ks=='0_700'):
    ax.set_yticklabels([])
else:
    plt.legend(prop={'size':18})
    ax.set_ylabel(r'$I(x,y=0,\omega_n)$')

yloc = plt.MaxNLocator(6)
ax.yaxis.set_major_locator(yloc)
ax.set_xlim(x[rango], x[-rango])
ax.set_ylim(0.7,1.2)


fig_ranges.savefig('fig_ranges_ks_{0:s}'.format(ks), bbox_inches='tight')



fig_real_P, ax = plt.subplots(1, 1, figsize=(5, 5))
im = ax.imshow(res_r[rango:-rango, rango:-rango, 1], cmap=cm.gray,
                 origin=None, extent=[x[rango], x[-rango], y[rango], y[-rango]])
ax.set_ylabel(y_label_i)
ax.set_xticklabels([])

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
ticks_at = np.linspace(np.amin(res_r[rango:-rango, rango:-rango, 1]),
                       np.amax(res_r[rango:-rango, rango:-rango, 1]), 5) 
cb = fig_real_P.colorbar(im, cax=cax, ticks=ticks_at, format='%1.2f')
cb.ax.tick_params(labelsize=20)

fig_real_P.savefig('fig_real_ks_{0:s}_{1:s}'.format(ks, letter_spi[1]), bbox_inches='tight')


fig_real_I, ax = plt.subplots(1, 1, figsize=(5, 5))
im = ax.imshow(res_r[rango:-rango, rango:-rango, 2], cmap=cm.gray,
                 origin=None, extent=[x[rango], x[-rango], y[rango], y[-rango]])
ax.set_ylabel(y_label_i)
ax.set_xlabel(x_label_i)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
ticks_at = np.linspace(np.amin(res_r[rango:-rango, rango:-rango, 2]),
                       np.amax(res_r[rango:-rango, rango:-rango, 2]), 5) 
cb = fig_real_I.colorbar(im, cax=cax, ticks=ticks_at, format='%1.2f')
cb.ax.tick_params(labelsize=20)

fig_real_I.savefig('fig_real_ks_{0:s}_{1:s}'.format(ks, letter_spi[2]), bbox_inches='tight')



# ranges of the above plots
for idx in range(3):
    print np.amin(res_r[:,:,idx]), np.amax(res_r[:,:,idx])

psi_r_diag = np.fft.fftshift(
    np.fft.ifft2(np.sqrt(nkx * nky) * psi_k_diag, axes=(0, 1)), axes=(0, 1))
res_r_diag = np.abs(psi_r_diag) ** 2 / \
    np.array(
        [ns_chosen / xs ** 2, np_chosen / xp ** 2, ni_chosen / xi ** 2])


fig_real_diag_S, ax = plt.subplots(1, 1, figsize=(5, 5))
im = ax.imshow(res_r_diag[rango:-rango, rango:-rango, 0], cmap=cm.gray,
                 origin=None, extent=[x[rango], x[-rango], y[rango], y[-rango]])

ax.set_ylabel(y_label_i)
#ax.set_yticklabels([])
if ks=='0_700':
    ax.set_xlabel(x_label_i)
else:
    ax.set_xticklabels([])

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
ticks_at = np.linspace(np.amin(res_r_diag[rango:-rango, rango:-rango, 0]),
                       np.amax(res_r_diag[rango:-rango, rango:-rango, 0]), 5) 
cb = fig_real_diag_S.colorbar(im, cax=cax, ticks=ticks_at, format='%1.2f')
cb.ax.tick_params(labelsize=20)

fig_real_diag_S.savefig('fig_real_diag_ks_{0:s}_{1:s}'.format(ks, letter_spi[0]), bbox_inches='tight')

fig_real_diag_P, ax = plt.subplots(1, 1, figsize=(5, 5))
im = ax.imshow(res_r_diag[rango:-rango, rango:-rango, 1], cmap=cm.gray,
                 origin=None, extent=[x[rango], x[-rango], y[rango], y[-rango]])

#ax.set_ylabel(y_label_i)
ax.set_yticklabels([])
if ks=='0_700':
    ax.set_xlabel(x_label_i)
else:
    ax.set_xticklabels([])

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
ticks_at = np.linspace(np.amin(res_r_diag[rango:-rango, rango:-rango, 1]),
                       np.amax(res_r_diag[rango:-rango, rango:-rango, 1]), 5) 
cb = fig_real_diag_P.colorbar(im, cax=cax, ticks=ticks_at, format='%1.2f')
cb.ax.tick_params(labelsize=20)

fig_real_diag_P.savefig('fig_real_diag_ks_{0:s}_{1:s}'.format(ks, letter_spi[1]), bbox_inches='tight')


fig_real_diag_I, ax = plt.subplots(1, 1, figsize=(5, 5))
im = ax.imshow(res_r_diag[rango:-rango, rango:-rango, 2], cmap=cm.gray,
                 origin=None, extent=[x[rango], x[-rango], y[rango], y[-rango]])

#ax.set_ylabel(y_label_i)
ax.set_yticklabels([])
if ks=='0_700':
    ax.set_xlabel(x_label_i)
else:
    ax.set_xticklabels([])

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
ticks_at = np.linspace(np.amin(res_r_diag[rango:-rango, rango:-rango, 2]),
                       np.amax(res_r_diag[rango:-rango, rango:-rango, 2]), 5) 
cb = fig_real_diag_I.colorbar(im, cax=cax, ticks=ticks_at, format='%1.2f')
cb.ax.tick_params(labelsize=20)

fig_real_diag_I.savefig('fig_real_diag_ks_{0:s}_{1:s}'.format(ks, letter_spi[2]), bbox_inches='tight')

print 'diagonal ranges'
for idx in range(3):
    print np.amin(res_r_diag[:, :, idx]), np.amax(res_r_diag[:, :, idx])


np.save("/home/berceanu/notebooks/OPODrag/psi_k", psi_k)
np.save("/home/berceanu/notebooks/OPODrag/res_r", res_r)


print "done!"

