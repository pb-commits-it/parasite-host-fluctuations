#!/usr/bin/env python3
"""Assemble the study report from saved numerical outputs, progressively.
Use --complete only when declared fits and artifact checks have finished.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
from scipy import stats
import numpy as np

def read(n):return json.loads((a.RES/(n+'.json')).read_text())
def fmt(x):return f'{x:.3f}'
def main():
 complete='--complete' in sys.argv;v=read('validation');g=read('reproduction_gate');md=read('meta_decision');mm=read('meta_models');un=read('unified_trait_summary');wd=read('weather_descriptive_summary');wp=read('weather_pre_analysis');th=read('thermal_summary');qa=read('weather_qc');aud=read('artifact_audit');sv=read('solver_check');power=read('pre_analysis_power');fn=read('figure_numbers');wf=read('weather_fit_summary') if (a.RES/'weather_fit_summary.json').exists() else {}
 phases={n:read(n) for n in ['M1_null','M2_year','M3_nhosts','M4_partype']}
 bmeta=mm['M3_nhosts_tax']['REML']['beta'][2];jmeta=mm['M7_joint_tax']['REML']['beta'][2]
 ub=un['M3_nhosts_tax'];uj=un['M7_joint_tax'];uno=un['M6_joint_notax'];within=mm['M8_within_3plus']['REML']
 mde=next(z for z in power['mde_summary'] if z['sigma_gamma']==.3 and z['sigma_tax']==.15)
 mde_vals=list(mde['full_85'].values());mw=list(mde['within_3plus_44'].values())
 summary={'timestamp':a.stamp(),'registered_outcome':'not decided','delta_meta_aicc':md['delta_aicc'],'bootstrap_p':md['bootstrap_p'],'bootstrap_B':md['B'],'bootstrap_mc_se':np.sqrt(md['bootstrap_p']*(1-md['bootstrap_p'])/(md['B']+1)),'unified_delta_aicc':uj['aicc']-ub['aicc'],'unified_notax_delta_aicc':uno['aicc']-phases['M3_nhosts']['aicc'],'meta_host_number_baseline':bmeta,'meta_host_number_joint':jmeta,'meta_host_number_attenuation':1-jmeta/bmeta,'unified_host_number_attenuation':1-uj['beta'][2]/ub['beta'][2],'thresholds':{'aicc':-2,'alpha':.05,'material_attenuation':.2},'n_original_models':len(phases),'n_unified_models_done':len(un),'n_weather_sets_done':sum('delta_aicc' in x for x in wf.values()),'weather_adjusted':{k:{'delta_aicc':w.get('delta_aicc'),'n_rows':w['n_rows'],'n_fish':int(w['n_fish']),'n_years':int(w['n_years'])} for k,w in wf.items()}}
 if complete:
  assert len(un)==13, len(un)
  assert len(wf)==5 and all('delta_aicc' in w or 'not_fit' in w for w in wf.values())
  assert all(z.get('converged',True) and z.get('outer_grad_max',0)<.02 and z.get('inner_grad_max',0)<1e-5 for z in un.values())
  assert md['bootstrap_failure_count']==0
 lines=[f'# Wood: parasite decline and life-cycle traits\n\nReport assembled {summary["timestamp"]}. Original registration: see the preserved [protocol](protocol.md). Environmental results below are an amendment.\n',
 f'**Registered outcome: not decided.** The coded traits do not improve the registered host-number comparison: meta-regression ΔAICc = {md["delta_aicc"]:.3f}, bootstrap p = {md["bootstrap_p"]:.4f}. The direct count-model comparison also worsens AICc by {summary["unified_delta_aicc"]:.3f}. None of the four primary trait contrasts is significant. This does not refute the mathematical theory; the required change in host variability and its link to parasite burden are unmeasured.\n',
 '## Question, data and provenance\n',
 f'Do fluctuation exposure, storage, alternative hosts and transmission mode explain temporal decline beyond obligate host number? One data row is one parasite-taxon count in one preserved fish. The analysis uses {v["n_observations"]:,} rows, {v["n_fish"]} fish of {v["n_fish_species"]} species, {v["n_taxa"]} published parasite codes and {v["n_host_parasite"]} host–parasite pairs, collected {v["year_min"]}–{v["year_max"]}. The retained counts sum to {sv["count_sum"]:,}. Host-number groups contain {sv["nhost_taxon_counts"]["1"]}, {sv["nhost_taxon_counts"]["2"]} and {sv["nhost_taxon_counts"]["3"]} codes. Some codes distinguish the same named parasite across fish hosts; they are not necessarily independent biological species.\n',
 'The primary dataset is [Dryad, doi:10.5061/dryad.fqz612jwf](https://doi.org/10.5061/dryad.fqz612jwf), version recorded in the inherited manifest, obtained from its official [Zenodo mirror](https://zenodo.org/records/7448240), under CC0. The authors’ GitHub code was read as text, not executed. The inherited `data/raw/wood/MANIFEST.json` and new `data/raw/wood/weather/MANIFEST.json` record provenance and request failures. This copied folder is not a Git repository, so the original git-ignore claim cannot be reverified here.\n',
 f'The locked trait CSV and both pre-analysis outputs retain their registered hashes. Every trait row has an inherited source string; {aud["manifest_files_audited"]} local files were checked against available manifest hashes, with no mismatch. The blind table was not rebuilt after outcomes. Its sources include the Benesh life-cycle data, GMPD, WoRMS and GloBI; generic family/genus rules also fill unresolved taxa. Published host associations establish possible hosts, not their competence, transfer shares, local abundance variance or synchrony. The label “fluctuating” is a guild proxy, and encystment does not measure a long, variable residence. These coding assumptions are a substantial limit.\n',
 'The original trait ledger lacks full bibliographic and read-status documentation for many monographs and papers. [Per-reference cards](literature_cards/) now explicitly mark those as unverified in this continuation. The article’s results, discussion and relevant methods were read; no novelty clearance is claimed from unread sources. See [query log](literature_query_log.md) and [trait source audit](results/trait_source_audit.csv).\n',
 '## Theory and pre-analysis power\n',
 'The theory compares a stationary fluctuating life cycle with the same arithmetic-mean rates. Its loss is D = log(R0) − log(R0*). For independent exposed transfers in a primitive geometric-stage ring, the quadratic approximation adds their variance-weighted coefficients: D2 = Σ σ²[(1+2θ)/(2T) − R/(1−R)], where T is total mean residence, θ = φ/(1−φ), φ is host memory, and R = E[φ^L] for generation interval L. At φ = 0 the coefficient is 1/(2T). Host labels do not determine the complete survival-and-timing kernel.\n',
 'The registered predictions are more decline with more fluctuating obligate transfers; less decline with storage; less decline with alternative/paratenic hosts; and more decline with active free-living transmission, subject to the registered mortality/storage interaction caveat. The inherited engine results quantify illustrative losses in log reproduction number. They do not predict an absolute century-scale count slope. That extra prediction assumes increasing host variability or narrowing persistence margins, a positive burden response, and comparable unmeasured mean rates. None is established by these museum counts.\n',
 'The pre-analysis calculation used the locked sample design and illustrative noise. The following detectable effects are per collection-year SD, for residual slope SD 0.30 and taxonomic SD 0.15.\n',
 '| Contrast | Expected sign | 80% detectable effect, all taxa | Within 3+ hosts |\n|---|---:|---:|---:|']
 design=read('categorical_design_audit');summary['categorical_design_audit']=design
 effect=read('theory_effect_sizes');summary['illustrative_theory_effect_sizes']=effect
 # Insert the theoretical magnitude paragraph before the power table, so all
 # estimated loss changes are distinguished from empirical slope magnitudes.
 locations=[i for i,line in enumerate(lines) if line.startswith('The pre-analysis calculation')]
 lines.insert(locations[0],f'The full categorical design has rank {design["rank"]} with {design["columns"]} columns. Its maximum VIF is {max(design["VIF"].values()):.3f}, correcting the protocol’s overstated low-collinearity claim. The pre-analysis power calculation already used this full design and is unchanged.\n')
 ee=[effect[k] for k in ['P1_added_fluctuating_transfer','P2_storage','P3_alternative_hosts','P4_active_vs_low_death_waiting_short_other_stay']]
 lines.insert(locations[0], f'The inherited simulator outputs give illustrative changes in D: an added fluctuating transfer, {ee[0][0]:+.3f} to {ee[0][1]:+.3f}; storage, {ee[1][0]:+.3f} to {ee[1][1]:+.3f}; alternative hosts, {ee[2][0]:+.3f} to {ee[2][1]:+.3f}; active waste versus low-death waiting with the shorter other stay, {ee[3][0]:+.3f} to {ee[3][1]:+.3f}. These are log reproduction-number losses from the inherited illustrative runs, read rather than rerun here. The registered long-storage/waiting-mortality counterexample limits the last sign. They cannot be converted to a measured decline rate.\n')
 names=['Fluctuating transfers','Storage','Spreading','Active fraction'];signs=['−','+','+','−']
 for i,name in enumerate(names):lines.append(f'| {name} | {signs[i]} | {mde_vals[i]["mde80"]:.3f} | {mw[i]["mde80"]:.3f} |')
 strong=[r['daicc_le_minus2_power'] for r in power['mc_summary'] if 'Strong' in r['scenario']];half=[r['daicc_le_minus2_power'] for r in power['mc_summary'] if 'Moderate' in r['scenario']]
 lines += [f'\nAt the illustrative full effect scale, power of the **AICc part alone** was {100*min(strong):.1f}–{100*max(strong):.1f}%; at half scale it was {100*min(half):.1f}–{100*max(half):.1f}%. These are not calibrated power estimates for the full conjunction of bootstrap, AICc, individual-sign and unified-model criteria. Small effects, especially spreading and transmission mode, remain hard to distinguish from no effect. All original scenario outputs are retained in `results/pre_analysis_power.*`.\n',
 '## Reproduction and registered analysis\n',
 f'The independent negative-binomial Laplace model passed the original reproduction gate. Its group slopes, transformed to percentage change per decade, are below. The year SD is {v["year_sd"]:.6f} years; intervals are transformed normal Wald intervals.\n',
 '| Obligatory hosts | Change per decade | 95% interval |\n|---|---:|---:|']
 for i,r in enumerate(fn['host_number_plot']):lines.append(f'| {i+1 if i<2 else "3+"} | {r["percent_per_decade"]:.3f}% | [{r["lower"]:.3f}%, {r["upper"]:.3f}%] |')
 lines+=['\n| Published-model reproduction | AICc | Difference from best |\n|---|---:|---:|']
 for n in g['aicc_order']:lines.append(f'| {n} | {phases[n]["aicc"]:.3f} | {phases[n]["aicc"]-phases["M3_nhosts"]["aicc"]:.3f} |')
 table_audit=read('published_table_audit');summary['published_table_audit']=table_audit
 lines.append(f'\nA source arithmetic discrepancy is retained explicitly: SI Table S4 reports null-model ΔAIC = 62.45, whereas the reproduced value against the best model is {table_audit["reproduced_null_minus_best_AICc"]:.3f}. The reported 62.45 agrees with comparing the null model against M2 ({table_audit["reproduced_null_minus_M2_AICc"]:.3f}), rather than M3. All four negative log likelihoods agree with the SI at its printed precision. This does not affect the registered reproduction criterion.')
 lines += ['\nThe model retains length effects by host–parasite pair, latitude effects by parasite code, fish-specimen random effects and taxon-specific overdispersion. Main trait models add taxonomic random slopes across the original broad groups. The registered secondary meta-regression uses M2 taxon slopes and their sampling variances, with ML for model comparison and REML/t intervals for contrasts. These are shrunken slopes and the meta-regression ignores their cross-taxon estimation covariance; the one-stage analysis is therefore an essential check.\n',
 f'The omnibus null bootstrap used {md["B"]:,} draws with {md["bootstrap_failure_count"]} unresolved numerical fits after optimizer checks. Its Monte Carlo standard error for p is {summary["bootstrap_mc_se"]:.4f}. The likelihood-ratio statistic is {md["lrt"]:.3f}. ΔAICc must be at most −2 and bootstrap p below 0.05, with an individually supported registered direction and the count-model check. Neither omnibus threshold is met. The no-taxonomy count comparison M6 versus original M3 also worsens AICc by {summary["unified_notax_delta_aicc"]:.3f}.\n',
 '| Primary contrast | REML coefficient | 95% interval | Two-sided p | Holm p | Registered decision |\n|---|---:|---:|---:|---:|---|']
 for i,name in enumerate(names):lines.append(f'| {name} | {md["traits_beta"][i]:.3f} | [{md["traits_ci_lo"][i]:.3f}, {md["traits_ci_hi"][i]:.3f}] | {md["traits_p"][i]:.3f} | {md["traits_holm"][i]:.3f} | not decided |')
 lines+=['\n| Primary direct count-model contrast | Coefficient | 95% normal Wald interval |\n|---|---:|---:|']
 for i,name in enumerate(names):
  j=i+3;bb=uj['beta'][j];ss=uj['se'][j];lines.append(f'| {name} | {bb:.3f} | [{bb-1.96*ss:.3f}, {bb+1.96*ss:.3f}] |')
 lines+=['\n| Within 3+ hosts | REML coefficient | 95% interval | Two-sided p | Holm p |\n|---|---:|---:|---:|---:|']
 from meta_analysis import holm
 wh=holm(within['pvalues'][1:5]);summary['within_3plus_holm']=wh
 for i,name in enumerate(names):
  j=i+1;lines.append(f'| {name} | {within["beta"][j]:.3f} | [{within["ci_lo"][j]:.3f}, {within["ci_hi"][j]:.3f}] | {within["pvalues"][j]:.3f} | {wh[i]:.3f} |')
 lines += [f'\nThe meta-regression host-number contrast changes from {bmeta:.3f} to {jmeta:.3f}; the count-model contrast attenuates by {100*summary["unified_host_number_attenuation"]:.1f}%. The registered stronger falsifier is **not met**: the primary intervals still include the illustrative full-scale effects −0.20 for fluctuating transfers and +0.30 for storage. No confidence interval here proves an absent mechanism.\n',
 '## Sensitivities and numerical checks\n',
 'The alternative spreading definition, storage coding, waterborne/active definitions, storage-by-active interaction, terminal-stage adjustment, resolved-taxon restriction and family grouping are retained as registered sensitivities. Their complete estimates are in `meta_models.json`, `unified_trait_summary.json` and the per-model checkpoints. The summary below gives the fluctuating-transfer and storage coefficients; subset rows are not AIC-compared to the full dataset.\n',
 '| Sensitivity | Meta taxa | Meta fluctuation / storage | Count fluctuation / storage |\n|---|---:|---:|---:|']
 for key in [k for k in mm if k.startswith('S')]:
  mr=mm[key]['REML'];ur=un.get(key);mt=f'{mr["beta"][3]:.3f} / {mr["beta"][4]:.3f}' if mr.get('estimable') else 'not estimable';ut=f'{ur["beta"][3]:.3f} / {ur["beta"][4]:.3f}' if ur and ur.get('estimable') else 'pending or not estimable'
  lines.append(f'| {key} | {mr["n"]} | {mt} | {ut} |')
 # Secondary registered effects are reported rather than hidden by the null primary.
 secondary=[]
 for key in ['S2b_terminal','S4_storage_active','S6_resolved']:
  z=mm[key]['REML'];secondary.append({'model':key,'storage_beta':z['beta'][4],'storage_ci':[z['ci_lo'][4],z['ci_hi'][4]],'storage_p':z['pvalues'][4],'storage_holm_p':z['holm_traits'][1]})
 summary['secondary_storage_results']=secondary
 for z in secondary:
  lines.append(f'\nIn {z["model"]}, the secondary storage estimate is {z["storage_beta"]:.3f} (95% interval {z["storage_ci"][0]:.3f} to {z["storage_ci"][1]:.3f}; nominal p = {z["storage_p"]:.4f}, four-trait Holm p = {z["storage_holm_p"]:.4f}).')
 interaction=mm['S4_storage_active']['REML']
 summary['secondary_interaction']={'beta':interaction['beta'][7],'p':interaction['pvalues'][7],'ci':[interaction['ci_lo'][7],interaction['ci_hi'][7]],'ml_delta_aicc_to_baseline':mm['S4_storage_active']['ML']['aicc']-mm['M3_nhosts_tax']['ML']['aicc']}
 lines.append(f'\nThe storage-by-active interaction is {interaction["beta"][7]:.3f} (p = {interaction["pvalues"][7]:.4f}); its storage main effect is conditional on zero active transfers. Its ML model still worsens AICc against the host baseline by {summary["secondary_interaction"]["ml_delta_aicc_to_baseline"]:.3f}. ML and REML select markedly different taxonomic variances in the interaction and resolved-subset models, so these conditional associations are unstable across estimation methods. They do not replace the original omnibus test, and no correction across all sensitivity models was registered. Subset AICc values are not compared with the full-data baseline.')
 lines += [f'\nThe family table actually has {sv["family_count"]} coded groups, including unresolved bins; the protocol’s prose count was corrected by a dated note without changing codes. The terminal adjustment omits {sv["missing_terminal_codes"]} unknown stage codes. The resolved subset retains {sv["resolved_taxa"]} codes. Family bins are a coarse control for relatedness, not a reconstructed phylogeny.\n',
 f'The supplied solver had an extra exponentiation in the dispersion SD. The original file is kept; `glmm_checked.py` fixes it and checks damped Newton convergence. Its full Hessian agrees with directional finite differences to relative error {sv["hessian_directional_relative_error"]:.2g}; the Schur solve residual is {sv["schur_solve_residual"]:.2g}; the final helper gives the same saved reproduction likelihood within {sv["nll_agreement"]:.1g}. Failed harness/solver attempts remain in files with `attempt` names. Variance components can lie near zero; AICc counts them as the registered model specifies.\n',
 '## Weather and environment — amendment\n',f'Dated {summary["timestamp"]}. The amendment was registered before any environmental association analysis. [Field records](results/field_records.csv) list all {wp["n_field_records"]} fish, original coordinates/localities and collection-year intervals. There are {wp["n_known_sites"]} distinct known coordinates and {wp["n_missing_coordinates"]} fish without coordinates. Every record lacks collection month and day. Positional uncertainty is unreported; some published coordinates were inferred from descriptive labels.\n',
 f'[NOAA ERSSTv6](https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.ersst.v6/sst.mnmean.nc/dataset.html) provides monthly SST on a 2-degree ocean grid across the full historical period. The nearest valid cell for all {wp["n_matched_fish"]} geolocated fish is 48° N, 124° W, matched to {wp["n_matched_years"]} sampled years ({qa["matched_year_min"]}–{qa["matched_year_max"]}). Distances are {wp["grid_distance_km"]["min"]:.1f}–{wp["grid_distance_km"]["max"]:.1f} km (median {wp["grid_distance_km"]["50%"]:.1f}). This cannot resolve local estuarine conditions. Annual mean SST and within-year SD of monthly means measure regional warmth and seasonality, respectively. They do not measure daily extremes. Every match, distance, annual window and missing value is in [weather_matches.csv](results/weather_matches.csv).\n',
 f'The archived Race Rocks annual SST covers {wd["coverage"]["temp_na_rm"]["fish"]} fish and {wd["coverage"]["temp_na_rm"]["years"]} sampled years. The station is {wp["race_rocks_distance_km"]["min"]:.1f}–{wp["race_rocks_distance_km"]["max"]:.1f} km from known capture locations. Its point observations are described by [DFO](https://www.dfo-mpo.gc.ca/science/data-donnees/lightstations-phares/index-eng.html). The refetched observer table contains a physically invalid 99.9°C entry; the first unfiltered comparison is retained. A dated post hoc QC comparison marks it missing and yields mean absolute disagreement {qa["mean_abs_difference"]:.3f}°C over {qa["overlap_years"]} overlapping years. It never replaces the published annual model input. Sediment axes are spatially averaged, interpolated and smoothed core proxies; both core distances are supplied in the matches.\n',
 'Land weather products do not measure seawater; the marine reconstruction has better temporal coverage than modern satellite SST. Daily field weather, local salinity, oxygen, pH, depth, freshwater inputs and collection season remain unaddressed. DFO/Dryad access refusals are recorded; no credentials or bypass were used. NOAA’s failing subset request was replaced by its public OPeNDAP representation of the same data.\n',
 f'The environmental design was full rank before outcome analysis; its standardized condition number was {wp["design_condition"]:.3f}. SST–year correlation was {wp["year_sst_correlation"]:.3f}. The saved pre-analysis environmental power simulation is an illustrative Gaussian design diagnostic, not NB-model power.\n',
 'Descriptive Pearson correlations below use annual, species-centered summaries, not repeated parasite rows. The outcome is mean log(1 + total retained parasites per fish), pooled across fish species after centering. This pooled descriptive outcome differs from the conditional group-specific count models. Intervals resample years and do not account for serial dependence.\n',
 '| Environmental covariate | Outcome r (years) | Host-density level r (years) | Absolute host-change r (years) |\n|---|---:|---:|---:|']
 correlations={(r['x'],r['y']):r for r in wd['pooled_correlations']}
 for c in ['sst_mean','sst_monthly_sd','temp_na_rm','ord1','ord2']:
  vals=[]
  for yy in ['log1p_burden','sfish_density','host_abs_change']:
   z=correlations[(c,yy)];vals.append(f'{z["r"]:.3f} ({z["n"]})' if z['r'] is not None else 'unavailable')
  lines.append('| '+c+' | '+' | '.join(vals)+' |')
 lines += ['\nThe absolute-change column is only a proxy from a standardized, interpolated fish-density index. It is based on sparse consecutive sampled years and does **not** estimate log-abundance variance of obligate hosts. The apparently strong SST association with that proxy has little independent information. Per-species correlations and intervals are in `environment_correlations.csv`. Missing host guilds and unknown transmission elasticities prevent quantifying the theory’s host-variance mechanism.\n',
 'The adjustment models keep year, length, latitude, specimen effects and taxonomic slopes, allowing each environmental effect to differ by host-number group. Each baseline/joint comparison uses exactly the same available rows. Values below are amended comparisons; their AICc values are not comparable across subsets.\n',
 '| Adjustment | Fish / rows | Years | Joint minus baseline AICc |\n|---|---:|---:|---:|']
 for k,w in wf.items():lines.append(f'| {k} | {w["n_fish"]} / {w["n_rows"]} | {w["n_years"]} | {w["delta_aicc"]:.3f} |' if 'delta_aicc' in w else f'| {k} | {w["n_fish"]} / {w["n_rows"]} | {w["n_years"]} | {w.get("not_fit","pending")} |')
 # Report group-specific adjusted thermal associations, which the pooled
 # descriptive correlation cannot capture, and explicit material shifts.
 env_effects=[]
 for key in ['W1_ersst','W2_race']:
  w=wf.get(key,{})
  if 'baseline' in w:
   c=w['covariates'][0];scale=w['covariate_sd'][c];bb=w['baseline']['environment_beta'][2]/scale;ee=w['baseline']['environment_se'][2]/scale
   env_effects.append({'model':key,'covariate':c,'log_count_per_C':bb,'se':ee,'percent_per_C':100*np.expm1(bb),'ci_percent_lo':100*np.expm1(bb-1.96*ee),'ci_percent_hi':100*np.expm1(bb+1.96*ee)})
 summary['adjusted_thermal_effects_3plus']=env_effects
 for e in env_effects:
  lines.append(f'\nFor the 3+-host group, {e["model"]} estimates {e["percent_per_C"]:.1f}% change per degree C (95% Wald interval {e["ci_percent_lo"]:.1f}% to {e["ci_percent_hi"]:.1f}%), conditional on year and the other model terms. This is an association, not a causal temperature response.')
 shifts=[]
 for key,w in wf.items():
  if 'joint' in w and w['joint'].get('estimable',True):
   beta=w['joint']['beta'];shifts.append({'model':key,'host_3plus':beta[2],'fluctuating':beta[3],'storage':beta[4],'relative_changes_from_full':[abs((beta[j]-uj['beta'][j])/uj['beta'][j]) if abs(uj['beta'][j])>.01 else abs(beta[j]-uj['beta'][j]) for j in [2,3,4]]})
 summary['environment_coefficient_shifts']=shifts
 if shifts:
  lines.append('\n| Amended joint model | 3+ host contrast | Fluctuating transfers | Storage |\n|---|---:|---:|---:|')
  for z in shifts:lines.append(f'| {z["model"]} | {z["host_3plus"]:.3f} | {z["fluctuating"]:.3f} | {z["storage"]:.3f} |')
  changed=[z['model'] for z in shifts if max(z['relative_changes_from_full'])>=.2]
  summary['material_coefficient_shifts']=changed
  if changed:lines.append('\nThe registered material-shift warning is triggered for '+', '.join(changed)+'. These coefficient comparisons combine adjustment with restriction to available records; they do not isolate an environmental causal effect.')
 finished=[w['delta_aicc'] for w in wf.values() if 'delta_aicc' in w]
 if complete and all(z>-2 for z in finished):
  lines.append('\n**The study conclusion does not change.** Every completed amended joint model fails the registered AICc improvement threshold. A separate environmental omnibus bootstrap was not run because that necessary support condition already fails. No new supports/contradicts verdict is assigned; the original not-decided result stands. Environmental associations and material coefficient shifts limit a causal reading of the original trend.')
 elif complete:
  lines.append('\nThe original not-decided result is unchanged. Some amended comparisons improve AICc; without the separately required environmental bootstrap these do not establish support. See the explicit completion limitation below.')
 lines += [f'\nThe temperature-input illustration uses nu(T)/nu(10) = Q10^((T−10)/10), the exponential-rate relation described in `notes/mechanism-temperature-incubation.md`, with illustrative Q10 values 1.5, 2 and 3. It compares exposed-only versus all-stage acceleration and annual-mean temperature versus mean monthly rate. The engine and closed-form coefficients agreed before use. Across {th["cases"]:,} computed combinations, storage ordering reversed in {th["storage_order_reversals"]}; its D2 ratio to no storage ranged {th["storage_ratio_min"]:.3f}–{th["storage_ratio_max"]:.3f}. Development-rate multipliers ranged {th["development_rate_ratio_min"]:.3f}–{th["development_rate_ratio_max"]:.3f}. These results are illustrative second-order calculations, not measured thermal responses or finite-noise forecasts. Rescaling all stages acknowledges that fish tissues are not constant-temperature shelters. This changes mean residence times; it does not simulate a temperature-driven mortality history or estimate the thermal curvature term separately.\n',
 '## What this says, and what remains undone\n',
 'The original registered result is **not decided**. The data reproduce the host-number pattern, but these inherited trait proxies do not earn added support. They neither identify increasing host variability nor separate it from mean host decline, food-web change or phenological mismatch. The stationary fluctuation theory remains conditional; failure of these proxies is not a counterexample to its proved statements. This study limits the usefulness of this empirical test and says nothing decisive about the stationary mathematical claims.\n',
 'Unaddressed processes include densities of unobserved obligate hosts; host memory and cross-host covariance; actual residence-time distributions; whether waiting loses parasites or delays transfer; nonlinear host/parasite regulation and feedback; stage-specific death, packets and demographic extinction; migration and alternative-host shares; evolving resistance; preservation/detection differences; and local weather at collection. The inherited engine illustrations explore only some of these. No life-history parameter was fitted to force the empirical pattern.\n',
 'These data can feed future studies with a reproducible count-design benchmark, temporal slopes and uncertainty, a frozen trait-proxy table, specimen-level spatial/year metadata and regional SST series. They cannot supply measured whole-cycle R0, obligate-host variance, residence-time spreads or validated thermal-development curves. Independent full-source trait curation and measured host histories are still needed for a publishable causal mechanism test. No novelty claim is made.\n',
 '## Files and reproduction\n',
 'Entry point: `python3 -I casestudies/wood/analysis.py validate`, then `analysis.py phase1`; `meta_analysis.py`; `unified_traits.py`; `weather.py prepare`; `weather_analysis.py descriptives` and `weather_analysis.py fits`; `thermal_sensitivity.py`; `weather_qc.py`; `check_solver.py`; `audit_artifacts.py`; `make_figures.py`; `theory_summary.py`; `check_meta_optima.py`; `final_checks.py`; and finally `build_report.py --complete`. Use `python3 -I` for every Python command. Existing model checkpoints are reused. The weather fetcher is `data/fetch/wood-weather.py`. Software versions are in `results/software_versions.json`; the local dependency installation is logged in `results/dependency_install.txt`. Original theory/power/trait scripts are retained but should not be rerun over the locked artifacts.\n',
 'Figures: [host-number reproduction](figures/host_number_decline.pdf) and [trait intervals](figures/trait_effects.pdf), with standalone pgfplots sources, CSVs and [plotting notes](figures/PLOTTING.md). Numerical JSON/CSV files and logs are under `results/`; [continuation audit](continuation.md) records departures.\n']
 if complete:lines.append('Status: done')
 else:lines.append('Status: not done — remaining registered sensitivity and environmental fits are running')
 (a.HERE/'report.md').write_text('\n'.join(lines)+'\n')
 a.save('report_numbers',summary);print(json.dumps(a.plain(summary),indent=2),flush=True)

if __name__=='__main__':main()
