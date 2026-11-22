# Protocol R14: across sites, is infection lower where hosts vary more? (NEON)

tertiary, 14 November 2026 (the file's time stamp is the record of when). Written before any NEON data file was downloaded or opened; only the catalogue of products and site-months had been read (`data/INVENTORY.md`, section 2). One pass, one report (`reports/R14-neon-across-sites.md`), then stop.

## Why

The theory's main prediction is that host variability costs a parasite reproduction number: the loss D of ln R0 rises with the variance of log host abundance (Results 1 to 4; exact values in run R10). One landscape cannot test that. The US National Ecological Observatory Network samples small mammals, questing ticks and the pathogens in those ticks at the same sites with one protocol; its catalogue shows 13 sites with five or more years of all three. This is the only open data found that can set infection against host variability across sites.

Rule 10 of `study rules` governs: data set ranges and check patterns. Nothing is fitted and no parameter of the theory is tuned.

## Data

NEON products, "basic" package, all available months, for the sites BLAN, DELA, HARV, KONZ, MLBS, ORNL, OSBS, SCBI, SERC, STEI, TALL, TREE, UKFS:
- small mammal box trapping (DP1.10072.001), table of captures per trap-night;
- ticks sampled using drag cloths (DP1.10093.001), field counts and taxonomy;
- tick-borne pathogen status (DP1.10092.001).
The product documentation (readme and variables files that come with each download) is to be read first, and the definitions below adjusted only where a column does not exist; any adjustment is reported.

## Definitions, fixed now

- **Mouse index, site and year:** the number of distinct tagged individuals of the genus *Peromyscus* captured, per 100 trap-nights of effort, pooled over all plots and bouts of the calendar year. A site-year counts if it has at least 500 trap-nights.
- **Host variability σ_site:** the standard deviation across years of ln(mouse index), with n − 1; years with a zero index are left out and counted. A site needs at least five years. **Memory r₁:** the correlation between consecutive years.
- **Infection:** among nymphs of *Ixodes scapularis* tested for *Borrelia burgdorferi* (sensu lato or sensu stricto, whichever the product reports; both if both, reported separately and the broader one used for the tests), the fraction positive. **NIP_site** pools all years. A yearly value needs at least 20 nymphs tested.
- **Sites included in the across-site tests:** those with at least 100 such nymphs tested in total over at least four years, and a σ_site.

## Predictions

| # | Prediction | Basis | Outcome rule |
|---|---|---|---|
| T1 (main) | Across the included sites, NIP_site falls as σ_site rises: Spearman's rank correlation is negative. | D rises with σ; lower R0* means lower prevalence in a regulated cycle (K8: R0 Π(1 − p_k) = 1). | "Supported" if negative with a one-sided permutation p below 0.05; "contradicted" if positive with p below 0.05 for the opposite side; otherwise "not decided". With fewer than five included sites the test is not run and that is the result. |
| T2 | At each included site with at least four yearly values, the standard deviation of ln(yearly NIP) is below that of ln(mouse index) over the same years. | K3, as P1 of run R12. | fails if more than a quarter of the sites violate it |
| T3 | The pooled within-site slope of ln(NIP in year t) on ln(mouse index in year t − 1) lies between 0 and 1 − p̄, within two standard errors. | K3, as P2 of run R12. | fails otherwise |

**Reported without a hypothesis:** σ_site and r₁ for every one of the 13 sites that has five years of the mouse index, whether or not ticks were tested there (the range of host variability across a continent, to set beside the 0.46 to 0.87 of the literature and the 0.77 to 1.03 of the Cary plots); latitude of each site and its rank correlation with σ_site and with NIP_site, to show how far T1 is confounded with geography. No adjustment is made for it.

## What a result would mean

Thirteen sites at most, and probably fewer than eight with infected *I. scapularis*, across a gradient of climate and host community that also affects infection. A rank correlation from so few sites cannot settle the prediction either way: with six sites it reaches p = 0.05 only if it is 0.83 or beyond. The result is one honest look, reported whatever it is.

## Not done

Other pathogens and other ticks; rodent blood tests (available only from 2020); mosquitoes; any model of the host community.

## Budget

Download: as long as it takes within one hour. Analysis: thirty minutes, one pass.
