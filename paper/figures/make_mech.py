#!/usr/bin/env python3
"""Generate W3 data and standalone pgfplots, offline (Python 3 + numpy only).

Run: python3 -I ../paper/figures/make_mech.py
Formula sources: contact-limited-transfer.md; mechanism-temperature-incubation.md;
mechanism-dispersal/mechanism_note.md; mechanism-linked-hosts/mechanism_note.md.
Finite-noise points are extracted from saved independent verification output.
No new simulations or fitted/interpolated thresholds are introduced.
"""
from pathlib import Path
import re
import numpy as np

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
ROOT = (__import__('pathlib').Path(__file__).resolve().parent / '../../code').resolve()
DATA = HERE / 'data'
DATA.mkdir(exist_ok=True)


def save(name, columns, header):
    values = np.column_stack(columns)
    assert np.isfinite(values).all(), name
    np.savetxt(DATA / name, values, delimiter=',', header=header,
               comments='', fmt='%.12g')


def factor(stay, z):
    return z / (stay - (stay - 1) * z)


b = np.geomspace(1, 1000, 301)
phi = 0.5
qj, qb = factor(3, phi), factor(b, phi)
r = qj * qb
theta = phi / (1 - phi)
kg = (1 + 2 * theta) / (2 * (3 + b)) - r / (1 - r)
l = (qj - r) / (1 - r)
k01 = 0.1**2 * kg + 0.1 * 0.9 * l
k05 = 0.5**2 * kg + 0.5 * 0.5 * l
save('mech-1.csv', [b, k01, k05, kg, b * 0 + .1 * .9 * qj,
                    b * 0 + .5 * .5 * qj, b * 0],
     'stay,d01,d05,gain,limit01,limit05,limitgain')
assert np.isclose(k01[0], 17 / 1120)

# Existing Monte Carlo estimates and their reported standard errors.
dispersal = (ROOT / 'results/verify-mechanisms-routes-dispersal_20261114.txt').read_text()
block = dispersal[dispersal.index('sigma = 0.10  D ='):]
# The first ten formatted lines cover the published example through sigma = 2.
rows = re.findall(r'sigma = ([\d.]+)  D = ([+\-\d.]+) \(SE ([\d.]+)\)', block)[:10]
rows = np.asarray(rows, dtype=float)
assert rows.shape == (10, 3) and np.array_equal(rows[:, 0], [.1, .2, .3, .5, .7, 1, 1.2, 1.4, 1.7, 2])
s = .5
phi, m, patches = .8, .15, 4
ra = factor(2, phi)**2
rb = ra * (1 - m) * (1 - phi * s) / (1 - phi * (1 - m) * s)
S = (1 + phi) / (2 * (1 - phi) * 4) - ra / (1 - ra)
H = rb / (1 - rb)
F = S / patches - (1 - 1 / patches) * H
assert np.isclose(F, -.3110577, atol=1e-7)
save('mech-2.csv', [rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 0]**2 * F], 'sigma,D,SE,quadratic')
noise = np.linspace(0, 2, 301)
save('mech-2-curve.csv', [noise, noise**2 * F], 'sigma,quadratic')

linked = (ROOT / 'results/verify-mechanism-linked-hosts_20261114.txt').read_text()
block = linked.split('HEAD. The headline:', 1)[1].split('PASS  HEAD1', 1)[0]
linked_rows = []
for line in block.splitlines():
    parts = line.split('|')
    if len(parts) != 3 or not re.fullmatch(r'\s*\d+\.\d+\s*', parts[0]):
        continue
    matches = [re.search(r'([+\-]\d+\.\d+) \((\d+\.\d+)\)', p) for p in parts[1:]]
    assert all(matches)
    linked_rows.append([float(parts[0]), *[float(x) for match in matches for x in match.groups()]])
linked_rows = np.asarray(linked_rows)
assert linked_rows.shape == (10, 5)

# Covariance-mode formula, with later-host/earlier-host lag convention.
z = .9 * np.exp(1j * np.pi / 10)
J = np.array([[0., -1.], [1., 0.]])
Hmode = (np.eye(2) - 1j * J) / 2
stays = np.array([5., 50., 5.])
q = factor(stays, z)
r = np.prod(q)
arc = np.array([[r, q[2] * q[0]], [q[1], r]])  # destination row, preceding transfer column
bracket = (1 + z) / (2 * stays.sum() * (1 - z)) - arc / (1 - r)
k0 = 2 * np.real(np.sum(Hmode * bracket))
k1 = 2 * np.real(np.sum(Hmode[::-1, ::-1] * bracket))
assert np.allclose([k0, k1], [.208123, -.094786166994], atol=1e-6)
save('mech-3.csv', [linked_rows[:, i] for i in range(5)] + [linked_rows[:, 0]**2 * k0, linked_rows[:, 0]**2 * k1], 'sigma,D0,SE0,D1,SE1,Q0,Q1')
save('mech-3-curve.csv', [noise, noise**2 * k0, noise**2 * k1], 'sigma,Q0,Q1')

# Equal means: geometric, two-substage chain, and fixed integer delay.
b = np.arange(2, 201, dtype=float)
phi = .9
theta = phi / (1 - phi)
qa = factor(4, phi)
transforms = [factor(b, phi), factor(b / 2, phi)**2, phi**b]
ks = [(1 + 2 * theta) / (2 * (4 + b)) - qa * t / (1 - qa * t) for t in transforms]
assert np.all(ks[0] <= ks[1]) and np.all(ks[1] <= ks[2] + 1e-12)
save('mech-4.csv', [b, *ks, b * 0 + 1 / 8], 'stay,geometric,chain,fixed,baseline')

print('Wrote mechanism CSVs; compile the shipped TeX with build_figs.py.')
