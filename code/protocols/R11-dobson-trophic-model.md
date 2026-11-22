# Protocol R11: Dobson's own model, with a predator and its prey as the two hosts

tertiary, 14 November 2026 (the file's time stamp is the record of when; written before any code for this run existed). One run, one report (`reports/R11-dobson-trophic-model.md`), then stop.

## Why

The chapter's nonlinear model (Dobson 2020, eqs 1.6 to 1.11, pp. 59 to 60) has never been analysed in this project: `README.md` says so, and draft 7 lists "cycles in which the parasite drives its hosts' numbers" under Limits. The equations as printed are not dimensionally consistent (the parasite's effect on the predator is written as a multiple of H; predation on the prey is written with C² and C_I²). This protocol fixes a corrected reading and asks what it does. The two hosts interact directly, by predation, so the model is not a ring of blocks in the sense of `notes/sealed/tertiary-coupled_as-sealed_20261114.md`, and the loop equation there does not apply to it as it stands. Part of the question is how far that theory's classification still predicts the outcome.

**This is published territory.** Trophically transmitted parasites in predator–prey models have been analysed by Hadeler & Freedman (1989), Fenton & Rands (2006) and Rogawa, Ogata & Mougi (2018) among others (cards in `notes/literature/coupled/`, most from abstracts). Nothing found here is to be called new before those are read in full.

## The model (the corrected reading; every departure from the printed equations is listed below)

State: H predators (definitive hosts), P adult parasites in predators, W free-living infective stages, C uninfected prey (intermediate hosts), I infected prey.

    dH/dt = e a H (C + I)/(C50 + C + I) − d H − α P
    dP/dt = ν a H I/(C50 + C + I) − (μ + d + α) P − α ((k + 1)/k) P²/H
    dW/dt = λ P − σ W − β (C + I) W
    dC/dt = f (C + I)(1 − (C + I)/K) − d_C C − a H C/(C50 + C + I) − β C W
    dI/dt = β C W − d_C I − a H I/(C50 + C + I)

Departures from the printed equations, each a choice of mine:

1. The parasite's effect on the predator is −αP (deaths of hosts in proportion to parasites carried, as in Anderson & May 1978), not −(α + δ)H. The effect δ on fecundity is left out.
2. Predation on prey is a H C/(C50 + C + I) and a H I/(C50 + C + I), not with C² and C_I².
3. The predator's births are e times its intake, with e a conversion efficiency (the printed "a b").
4. Parasites in infected prey are not tracked separately (the chapter says their distribution is ignored); each infected prey eaten establishes ν adult parasites.
5. The clumping term is α((k + 1)/k)P²/H, the negative-binomial form of Anderson & May (1978). The printed αP²/H is the case in which that factor is 1; it is one of the settings below, labelled "printed".
6. Prey births are logistic with carrying capacity K; K = ∞ gives the printed equation. Infected prey breed as uninfected ones do.

## Questions and hypotheses

| # | Statement | Basis | Kind |
|---|---|---|---|
| H0 | The code reproduces known results before it is used: (a) with no parasite and K finite, the predator–prey equilibrium loses stability exactly at the Rosenzweig–MacArthur condition (unstable when the prey's equilibrium density is below the peak of its isocline, C* < (K′ − C50)/2 with K′ = K(1 − d_C/f) the prey's carrying capacity net of deaths; the script derives the condition and compares it with eigenvalues); (b) with K = ∞ and no parasite the equilibrium is never stable. | textbook (recalled; the script derives the condition and tests it against the Jacobian, which does not depend on the recollection) | check, tolerance 1e-6 on the sign of the largest real part at ±1% of the boundary |
| H1 | Across parameter sets for which both a parasite-free and an endemic equilibrium exist and are positive, the fraction of endemic equilibria that are locally stable differs from the fraction of parasite-free ones. Direction not predicted. | open; the literature reports both |
| H2 | At a fixed endemic equilibrium, lengthening the free-living stage (σ and β divided by the same factor c, which leaves the equilibrium of every other variable unchanged) changes stability non-monotonically in some parameter sets and never restores stability at very long stages in sets where the prey is not self-limited (K = ∞). | items 4 and 5 of `board/0046`: a host with its own regulation against one without | open; a test of whether the block theory's classification carries over |
| H3 | Whether the endemic equilibrium is stable is predicted by where the parasite and the predator hold the prey on its production curve: stable more often when C* + I* > K/2. | item 3 of `board/0046` | open |
| H4 | Compared with a parasite of the predator with a direct life cycle (predators pick up W directly at rate β_H W H; no prey stage), matched for R0 and for the mean time round the cycle, the trophic cycle's endemic equilibrium is stable in a different fraction of parameter sets. Direction not predicted. | Dobson's claim | open |

## Grid, with the reason for each range

Time is in units of the predator's life expectancy (d = 1). 20,000 parameter sets drawn log-uniformly:

- prey turnover d_C in [2, 50] and f/d_C in [1.5, 10] (prey shorter-lived than predators, able to grow);
- C50 = 1 (sets the unit of prey density); K in {∞} for a fifth of the sets and log-uniform in [2, 200] for the rest;
- a in [5, 100], e in [0.02, 0.5] (so that predators can persist on prey: e a > d required, sets failing it are redrawn);
- parasite: μ in [0.2, 20], α in [0.001, 1], k in {0.2, 1, 5, ∞ meaning the printed form}, λ in [1, 10⁴], σ in [1, 200], β in [10⁻³, 10], ν in [0.1, 10].

A set is kept if the parasite-free predator–prey equilibrium is positive and R0 > 1. Endemic equilibria are found by integrating from the parasite-free state with a small inoculum and then Newton's method; sets where the integration does not settle are classified by the eigenvalues at every positive equilibrium Newton finds from 20 random starts, and are reported separately as "no stable equilibrium found".

**H2:** for 2,000 of the kept sets, c in {0.01, 0.1, 0.3, 1, 3, 10, 100, 10⁴}.
**H4:** the direct cycle uses the same H, P, W equations with uptake β_H W H in place of the prey route and the same clumping; β_H and λ are set so that R0 and Σ(mean stage times) match the trophic set.

## What is measured

Sign of the largest real part of the Jacobian's eigenvalues (finite differences, checked against an analytic Jacobian on 200 sets, tolerance 1e-5 relative); for H2 the pattern across c; for H3 the position (C* + I*)/K; counts with binomial standard errors. Twelve integrations of the nonlinear model, six on each side of a boundary found, to confirm the linear classification (a perturbation of 1e-4 relative over a time 6/|Re z| shrinks tenfold or grows tenfold).

## Budget and stop

Forty minutes of computing. Stop there and report what finished. No second run with other ranges; if a range turns out to exclude most sets, say so in the report.

## What is not tested

Manipulation of prey behaviour by the parasite; reduced fecundity of infected hosts (δ; castration); more than one prey or predator species (Dobson's eqs 1.14 to 1.17); stochasticity; the global dynamics beyond the twelve integrations.
