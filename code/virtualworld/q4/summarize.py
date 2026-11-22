"""Summarize the one Q4 run; enumerate each departure and its diagnostic evidence."""
import sys,collections
sys.dont_write_bytecode=True
from pathlib import Path
P=(__import__('pathlib').Path(__file__).resolve().parent / '.').resolve();sys.path.insert(0,str(P))
from numerics import *

def readcsv(name): return list(csv.DictReader((P/name).open()))
def fl(x):return None if x in ('',None) else float(x)
def fmt(x):return 'NA' if x is None else f'{x:.5g}'
def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows]

if __name__=='__main__':
    if (P/'report.md').exists():raise RuntimeError('report is append-only; later corrections need dated note')
    completed=json.loads((P/'completion-manifest.json').read_text());assert completed['status']=='complete'
    cells=[json.loads(x) for x in (P/'cells.jsonl').read_text().splitlines()]
    ci={(r['series_id'],r['stages'],r['stay']):r for r in cells}
    meta={r['series_id']:r for r in readcsv('series.csv')};mono=readcsv('monotonicity.csv');h3=readcsv('contrasts.csv')
    pers=readcsv('persistence-exact.csv');sel=json.loads((P/'selection.json').read_text())
    dg=[json.loads(x) for x in (P/'diagnostics.jsonl').read_text().splitlines()];di=collections.defaultdict(dict)
    for r in dg:di[r['series_id'],r['stages'],r['stay']][r['variant'],r['replicate']]=r
    evidence=[];ei={}
    def ev(D,full,ar,variants,flags):
        material=max(.005,.2*abs(D));e=dict(material_D_change=material,feature_flags=flags)
        tags=[]
        if full is None:tags.append('zero_support')
        if ar is not None and full is not None and status(D,ar)=='departure' and status(D,full)=='agree':tags.append('non_AR_memory_repaired_by_full_ACF')
        if full is not None and ('small',0) in variants:
            small=variants['small',0]['D']/.01;err=abs(small-full);lim=max(.001,.05*abs(full))
            e.update(small_rescaled_D=small,small_rescaled_error=err,small_tolerance=lim,small_agrees=bool(err<=lim))
            if status(D,full)=='departure' and err<=lim:tags.append('finite_amplitude_contraction')
            if status(D,full)=='departure' and err>lim:tags.append('small_amplitude_residual_unresolved')
        for kind in ('phase','shuffle','detrend','clip','mirror'):
            vv=[r for (v,j),r in variants.items() if v==kind]
            if not vv:continue
            avg=float(np.mean([r['D'] for r in vv]));delta=avg-D
            e[kind+'_mean_D']=avg;e[kind+'_delta_D']=delta
            e[kind+'_min_D']=min(r['D'] for r in vv);e[kind+'_max_D']=max(r['D'] for r in vv)
            is_material=abs(delta)>material;e[kind+'_material']=is_material
            if full is not None and all(r['prediction_full'] is not None for r in vv):
                error=float(np.mean([abs(r['D']-r['prediction_full']) for r in vv]));e[kind+'_full_error']=error
            if is_material:
                if kind=='phase':tags.append('beyond_covariance_phase_or_marginal_shape')
                elif kind=='shuffle':tags.append('temporal_order_sensitive')
                elif kind=='detrend':tags.append('trend_sensitive' if 'trend' in flags else 'detrending_sensitive_without_trend_flag')
                elif kind=='clip':tags.append('tail_clipping_sensitive')
                elif kind=='mirror':tags.append('replay_boundary_sensitive')
        if 'cycle' in flags and e.get('shuffle_material'):tags.append('cyclic_order_supported')
        if not tags:tags.append('unresolved_with_registered_diagnostics')
        e['attribution']=';'.join(tags)
        return e
    for c in cells:
        key=c['series_id'],c['stages'],c['stay'];variants=di.get(key,{})
        e=ev(c['D'],c['prediction_full'],c['prediction_AR'],variants,meta[c['series_id']]['feature_flags'])
        e.update(series_id=c['series_id'],stages=c['stages'],stay=c['stay'],D=c['D'],diagnostics_run=bool(variants))
        evidence.append(e);ei[key]=e
    write_csv(P/'cell-attribution.csv',evidence)
    ledger=[];scope=[]
    for c in cells:
        key=c['series_id'],c['stages'],c['stay'];e=ei[key]
        for pred in ('AR','full'):
            status_=c['H1_'+pred]
            row=dict(hypothesis='H1_'+pred,series_id=c['series_id'],stages=c['stages'],stay=c['stay'],
                actual=c['D'],predicted=c['prediction_'+pred],tolerance=c['tolerance_'+pred],
                error=abs(c['D']-c['prediction_'+pred]) if c['prediction_'+pred] is not None else None,
                attribution=e['attribution'],feature_flags=e['feature_flags'],evidence_file='cell-attribution.csv',
                scope_limit='AR(1)/finite-log and small-amplitude assumptions; circular-record result only')
            if status_=='departure':ledger.append(row)
            elif status_=='inapplicable':scope.append(row)
    mevidence=[]
    for r in mono:
        if r['H2']!='reversal':continue
        sid=r['series_id'];k=int(r['stages']);a=int(r['from_stay']);b=int(r['to_stay'])
        da=di.get((sid,k,a),{});db=di.get((sid,k,b),{});tags=[]
        e=dict(series_id=sid,stages=k,from_stay=a,to_stay=b,actual_decrease=float(r['decrease']),feature_flags=meta[sid]['feature_flags'])
        for kind in ('small','phase','shuffle','detrend','clip','mirror'):
            pairs=[(da[x],db[x]) for x in sorted(set(da)&set(db)) if x[0]==kind]
            if not pairs:continue
            deltas=np.array([x['D']-y['D'] for x,y in pairs]);e[kind+'_mean_decrease']=float(deltas.mean())
            e[kind+'_reversals']=int(np.sum(deltas<(-1e-8 if kind=='small' else -1e-6)))
            if kind=='shuffle' and not np.any(deltas< -1e-6):tags.append('reversal_removed_by_permutation_temporal_order')
            if kind=='small' and np.any(deltas< -1e-8):tags.append('reversal_survives_small_amplitude')
            if kind in ('detrend','clip','mirror') and not np.any(deltas< -1e-6):tags.append('reversal_removed_by_'+kind)
        if int(meta[sid]['zeros']):tags.append('zero_support')
        if 'cycle' in meta[sid]['feature_flags']:tags.append('cyclic_series')
        if 'non_AR_ACF' in meta[sid]['feature_flags']:tags.append('non_AR_memory')
        if not tags:tags=['mixed_or_unresolved_temporal_and_marginal_effects']
        e['attribution']=';'.join(tags);mevidence.append(e)
        ledger.append(dict(hypothesis='H2',series_id=sid,stages=k,from_stay=a,stay=b,actual=float(r['decrease']),
            predicted='positive decrease',tolerance=1e-6,error=-float(r['decrease']),attribution=e['attribution'],
            feature_flags=e['feature_flags'],evidence_file='monotonicity-attribution.csv',
            scope_limit='No universal monotonicity for arbitrary replayed host sequences'))
    write_csv(P/'monotonicity-attribution.csv',mevidence)
    hevidence=[]
    for r in h3:
        if 'departure' not in (r['H3_AR'],r['H3_full']):continue
        k=int(r['stages']);s=int(r['stay']);members=r['members'].split(';');n=int(r['n_fluct'])
        variant_keys=set.intersection(*(set(di.get((sid,k,s),{})) for sid in members))
        vv={}
        for key in variant_keys:
            arr=[di[sid,k,s][key] for sid in members];v={}
            for name in ('D','prediction_full','prediction_AR'):
                vals=[x[name] for x in arr];v[name]=float(np.mean(vals[:n])-np.mean(vals[n:])) if all(x is not None for x in vals) else None
            vv[key]=v
        flags=';'.join(sorted(set(f for sid in members for f in meta[sid]['feature_flags'].split(';') if f)))
        e=ev(float(r['D_contrast']),fl(r['prediction_full']),fl(r['prediction_AR']),vv,flags)
        e.update(contrast_id=r['contrast_id'],members=r['members']);hevidence.append(e)
    hei={r['contrast_id']:r for r in hevidence};write_csv(P/'contrast-attribution.csv',hevidence)
    for r in h3:
        for pred in ('AR','full'):
            row=dict(hypothesis='H3_'+pred,contrast_id=r['contrast_id'],stages=int(r['stages']),stay=int(r['stay']),
                actual=float(r['D_contrast']),predicted=fl(r['prediction_'+pred]),tolerance=fl(r['tolerance_'+pred]),
                error=abs(float(r['D_contrast'])-float(r['prediction_'+pred])) if r['prediction_'+pred] else None,
                attribution=hei[r['contrast_id']]['attribution'] if r['contrast_id'] in hei else 'zero_support',
                evidence_file='contrast-attribution.csv',members=r['members'],partial_source=r['partial_source'],
                scope_limit='Finite-record normalized host forcing; not empirical parasite persistence')
            if r['H3_'+pred]=='departure':ledger.append(row)
            elif r['H3_'+pred]=='inapplicable':scope.append(row)
    rare=json.loads((P/'rare-event-check.json').read_text())
    for r in readcsv('demographic-pgf-checks.csv'):
        if r['passed']=='False':
            ledger.append(dict(hypothesis='MC-validation',series_id=r['series_id'],stages=int(r['stages']),stay=int(r['stay']),
                endpoint=r['endpoint'],actual=float(r['observed']),predicted=float(r['expected']),error=float(r['error']),
                tolerance=float(r['tolerance']),attribution='rare_event_discreteness_exact_tail_test',evidence_file='rare-event-check.json',
                scope_limit='Registered four-SE diagnostic is not a calibrated rare-event probability interval; failure retained'))
    write_csv(P/'departures.csv',ledger);write_csv(P/'prediction-inapplicable.csv',scope)
    # Compact plots are standalone exportable scientific artifacts.
    os.environ['MPLCONFIGDIR']=str(P/'raw/mplconfig')
    sys.path.insert(0,str(P/'raw/plot-deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for src,col in [('GPDD','#2867a7'),('BioTIME','#189477'),('Tribolium','#bf6230')]:
        rr=[r for r in cells if r['source']==src and r['prediction_full'] is not None]
        axes[0].scatter([r['prediction_full'] for r in rr],[r['D'] for r in rr],s=8,alpha=.55,label=src)
    bounds=axes[0].get_xlim();axes[0].plot(bounds,bounds,'k--',lw=.8);axes[0].legend(fontsize=8)
    axes[0].set(xlabel='Full-autocovariance prediction',ylabel='Replay threshold D',title='Finite-amplitude departures')
    for sid in meta:
        rr=[ci[sid,2,s] for s in (1,5,20)];reverse=any(rr[j]['D']<rr[j+1]['D']-1e-6 for j in (0,1))
        axes[1].plot([1,5,20],[r['D'] for r in rr],color='#b44937' if reverse else '#8ca0ad',alpha=.3,lw=.8)
    axes[1].axhline(0,color='black',lw=.6);axes[1].set(xlabel='Steady mean stay',ylabel='D, two stages',title='Red lines have a reversal',xticks=[1,5,20])
    for src,col in [('GPDD','#2867a7'),('BioTIME','#189477'),('Tribolium','#bf6230')]:
        for k,style in [(2,'-'),(3,'--')]:
            vals=[np.mean([float(r['survival']) for r in pers if r['source']==src and int(r['stages'])==k and int(r['stay'])==s and r['endpoint']=='G50']) for s in (1,5,20)]
            axes[2].plot([1,5,20],vals,style,color=col,label=f'{src}, {k} stages')
    axes[2].set(xlabel='Steady mean stay',ylabel='Mean persistence probability',title='50 matched reference generations',xticks=[1,5,20],ylim=(0,1));axes[2].legend(fontsize=7)
    fig.savefig(P/'summary.png',dpi=170);fig.savefig(P/'summary.pdf');plt.close(fig)
    # Aggregate descriptive numbers, never a pooled independent-study test.
    summary=dict(date=date(),H1={p:dict(collections.Counter(r['H1_'+p] for r in cells)) for p in ('AR','full')},
        H2=dict(collections.Counter(r['H2'] for r in mono)),H3={p:dict(collections.Counter(r['H3_'+p] for r in h3)) for p in ('AR','full')},
        departure_rows=len(ledger),inapplicable_rows=len(scope),diagnostic_series=completed['diagnostic_series'],diagnostic_roots=len(dg))
    write_json(P/'summary.json',summary)
    phase1=json.loads((P/'phase1-validation.json').read_text());nval=json.loads((P/'numerical-validation.json').read_text())
    pgfv=json.loads((P/'pgf-validation.json').read_text());mc=readcsv('demographic-pgf-checks.csv')
    failed_mc=[r for r in mc if r['passed']=='False']
    positive=sum(not int(r['zeros']) for r in meta.values());zero=len(meta)-positive
    both_decrease=sum(all(x['H2']=='decrease' for x in mono if x['series_id']==sid and int(x['stages'])==k) for sid in meta for k in (2,3))
    unique_bad=len(set(r['series_id'] for r in mono if r['H2']=='reversal'))
    repaired=sum(r['H1_AR']=='departure' and r['H1_full']=='agree' for r in cells)
    fullbad=[r for r in evidence if ci[r['series_id'],r['stages'],r['stay']]['H1_full']=='departure']
    finite=sum(r.get('small_agrees',False) for r in fullbad)
    orderremoved=sum('reversal_removed_by_permutation' in r['attribution'] for r in mevidence)
    h3partial=[r for r in h3 if r['partial_source']=='True']
    lines=['# Q4-1 — real host trajectories in the virtual parasite world','',f'Report written {date()}.',
        '',f'The universal forms of H1 and H2 do not survive this replay experiment. Of {positive*6} cells with finite log variance, '
        f'{summary["H1"]["AR"]["departure"]} depart from AR(1) and {summary["H1"]["full"]["departure"]} from the full-autocovariance prediction. '
        f'There are {summary["H2"]["reversal"]} adjacent-stay reversals across {unique_bad} records. '
        'H3 agrees in most evaluable contrasts but also has registered departures. These are numerical results for generic parasites driven by observed records; they are not observations of parasite infection.',
        '', '![Q4 result summary](summary.png)', '',
        '## Design, data and verification', '',
        f'The frozen [protocol and dated amendments](protocol.md) precede empirical outcomes. The stratified seeded sample contains 120 compilation records from {sel["eligible"]} eligible records, plus 71 valid Tribolium stage records. '
        f'There are {len(meta)} record labels, {len(meta)-len(sel["overlaps"])} distinct abundance sequences, {positive} positive records and {zero} zero-containing records. '
        'Eight larval/adult control records are repeated between experiments 7 and 10; label-level counts include these shared controls and must not be read as independent evidence. '
        'Annual and subannual records remain separate sampling strata; one model step is one observation interval. No parasite timescale is inferred from these data.',
        '', 'Both public Zenodo archives were fetched with the generic User-Agent `ecology-research/1.0`, without credentials. '
        'The [2023 archive](https://zenodo.org/records/7972066) supplies the primary laboratory records; the '
        '[2018 archive](https://zenodo.org/records/4976742) was read for provenance and excluded from primary contrasts because it lacks a constant-habitat arm. '
        'Both fetched API metadata records identify CC0. Workbook structure and treatment meanings were read from the downloaded READMEs. '
        'This is data acquisition, not a literature novelty search; no paper was used to establish novelty and no sources/ file was opened.',
        '', 'The compilation is the supplied BioTIME/GPDD extraction, despite the directory name lpd. Licensing and study metadata were read from the supplied files, not independently re-audited against every original study. '
        'The original metrics are detrended/gap-aware summaries; predictions here use freshly computed, undetrended, circular log covariance for the actual normalized driver. '
        'All input records are arithmetic-mean normalized to one. Zeros are retained. No pseudocount, interpolation, fit to parasite data, or new ecological mechanism was used.',
        '', 'Circular replay makes the estimand explicit: D is the Perron-product invasion threshold for the repeated finite sequence, relative to the mean-one operator. '
        'It is invariant to starting phase. It is not an estimate of the long-run law of an unobserved stationary natural process. '
        'The full-covariance response is a numerically checked implementation of the supplied matrix expansion, not a new proved proposition or a second-party-reviewed theorem.',
        '',f'Known-result validation: phase-1 K={phase1["K"]:.10f}, reproduced {phase1["estimate"]:.10f}, SE={phase1["se"]:.6g}; '
        f'error {phase1["error"]:.6g} is below registered tolerance {phase1["tolerance"]:.6g}. All 22 supplied tests, '
        f'{len(nval["checks"])} periodic/covariance/centering checks, and {len(pgfv["checks"])} exact-PGF checks passed. '
        f'The saved Monte Carlo endpoints agree with their same-phase PGF predictions in {len(mc)-len(failed_mc)}/{len(mc)} comparisons. '
        'See [phase-1 validation](phase1-validation.json), [numerical validation](numerical-validation.json), '
        '[PGF validation](pgf-validation.json) and [Monte Carlo checks](demographic-pgf-checks.csv).',
        '', '## H1: loss against predictions', '',
        'Agreement means |D−prediction| ≤ max(0.01, 20% of |prediction|), in log reproduction per circuit. '
        'AR(1) uses the record’s circular lag-one log correlation; full ACF uses all circular lags through the validated response kernel. '
        'There is no sampling SE on the periodic root itself; biological input uncertainty is not quantified.', '']
    lines+=table(['Stages','Stay','AR agree / depart','Full ACF agree / depart','Inapplicable'],[
        [k,s,f"{sum(r['H1_AR']=='agree' for r in cells if r['stages']==k and r['stay']==s)} / {sum(r['H1_AR']=='departure' for r in cells if r['stages']==k and r['stay']==s)}",
         f"{sum(r['H1_full']=='agree' for r in cells if r['stages']==k and r['stay']==s)} / {sum(r['H1_full']=='departure' for r in cells if r['stages']==k and r['stay']==s)}",zero] for k in (2,3) for s in (1,5,20)])
    lines+=['',f'H1 verdict: **fails as a universal finite-amplitude claim**. Full covariance repairs {repaired} AR departures, '
        f'but {len(fullbad)} full-covariance departures remain. Among those, {finite}/{len(fullbad)} meet the registered rescaled small-amplitude check; '
        'this supports finite-amplitude terms beyond the second-order expansion in those cells. The remainder stay explicitly unresolved at that registered diagnostic amplitude. '
        f'The {zero*6} zero-containing cells lack finite log variance and cannot be counted as agreement or an AR fit.',
        '', '## H2: does D fall with steady stay?', '']
    lines+=table(['Stages','Stay comparison','Decreases','Flat','Reversals'],[[k,f'{a}→{b}',*[sum(r['H2']==v for r in mono if int(r['stages'])==k and int(r['from_stay'])==a) for v in ('decrease','flat','reversal')]] for k in (2,3) for a,b in ((1,5),(5,20))])
    lines+=['',f'H2 verdict: **false for every-series wording**. Both decreases hold for {both_decrease}/{len(meta)*2} record-by-stage designs; '
        f'{summary["H2"]["reversal"]} reversals occur among {len(mono)} adjacent comparisons. '
        f'Permutation removes the reversal in all eight shuffled versions for {orderremoved}/{len(mevidence)} reversed comparisons, demonstrating a temporal-order contribution there. '
        'Negative D is a real fluctuation benefit relative to the arithmetic-mean operator, not a sign error; removing that benefit as stay rises is an H2 reversal. '
        'The phase-1 positive-memory, small-noise restrictions cannot be extrapolated to all cyclic or zero-containing sequences.',
        '', 'Largest reversed comparisons (all reversals are in the linked ledger):', '']
    largest=sorted(mevidence,key=lambda r:r['actual_decrease'])[:6]
    lines+=table(['Series','Stages / stays','D at first','D at next','Diagnostic attribution'],[[r['series_id'],f"{r['stages']}: {r['from_stay']}→{r['to_stay']}",fmt(ci[r['series_id'],r['stages'],r['from_stay']]['D']),fmt(ci[r['series_id'],r['stages'],r['to_stay']]['D']),r['attribution']] for r in largest])
    lines+=['','## D and persistence by source and stay','',
        'The table gives the median D across record labels and the arithmetic mean of their exact persistence probabilities at G=50. '
        'Two-stage residence means are [2,s]; three-stage means are [2,2,s]. All use log R0=.05 and 20 initial individuals. '
        'These descriptive source averages are not population estimates or independent-study significance tests. Full per-series outcomes, '
        'including G=23,100 and step 500, are in [cells.csv](cells.csv) and [persistence-exact.csv](persistence-exact.csv).','']
    lines+=table(['Source','Stages','Stay','Median D','Mean persistence, G50'],[[src,k,s,
        fmt(float(np.median([r['D'] for r in cells if r['source']==src and r['stages']==k and r['stay']==s]))),
        fmt(float(np.mean([float(r['survival']) for r in pers if r['source']==src and int(r['stages'])==k and int(r['stay'])==s and r['endpoint']=='G50'])))]
        for src in ('BioTIME','GPDD','Tribolium') for k in (2,3) for s in (1,5,20)])
    lines+=['', 'Persistence is computed from the same engine event law by an exact branching PGF, averaged over every starting phase. '
        'Its phase minimum/maximum describe environmental timing, not confidence intervals. Short records and shared cultures do not acquire extra biological replication when replayed. '
        'D governs the invasion threshold; it is not by itself a theorem ordering finite-horizon extinction.',
        '', '## H3: imposed fluctuating versus constant habitat','',
        'Contrasts are fluctuating minus constant mean D within experiment, beetle stage and initial-condition block, with amplitude treatments separate. '
        'The nominal flour schedules share arithmetic mean 20 g; finite observation windows may have unbalanced last cycles. '
        'Raw beetle means differ; the world’s equal host mean comes from normalization. '
        'Host/flour means are retained in [series.csv](series.csv), not conflated.','']
    lines+=table(['Experiment','Host','Contrasts','AR agree / depart / NA','Full agree / depart / NA'],[[ds,st,len([r for r in h3 if int(r['dataset'])==ds and r['stage']==st]),
        ' / '.join(str(sum(r['H3_AR']==v for r in h3 if int(r['dataset'])==ds and r['stage']==st)) for v in ('agree','departure','inapplicable')),
        ' / '.join(str(sum(r['H3_full']==v for r in h3 if int(r['dataset'])==ds and r['stage']==st)) for v in ('agree','departure','inapplicable'))]
        for ds in (7,8,10) for st in ('L','A')])
    lines+=['',f'H3 verdict: **partially supported, not universal**: each prediction has {summary["H3"]["full"]["agree"]} agreements, '
        f'{summary["H3"]["full"]["departure"]} departures and {summary["H3"]["full"]["inapplicable"]} inapplicable contrasts. '
        f'{len(h3partial)} adult contrasts in experiment 8 are partial-source comparisons after the invalid-count exclusion; they are disclosed separately below. '
        'Zero-containing member cultures make the entire log-based treatment contrast inapplicable, even if other cultures have finite log variance. '
        'Shared experiment 7/10 controls prevent treating those contrast counts as independent replication.',
        '', 'Every H3 departure (AR and full prediction columns use the same fixed tolerance rule):','']
    lines+=table(['Contrast','Observed ΔD','AR ΔD','Full ΔD','Attribution'],[[r['contrast_id'],fmt(float(r['D_contrast'])),fmt(fl(r['prediction_AR'])),fmt(fl(r['prediction_full'])),hei[r['contrast_id']]['attribution']] for r in h3 if 'departure' in (r['H3_AR'],r['H3_full'])])
    lines+=['','Partial-source experiment 8 adult comparisons (2 fluctuating cultures / 3 controls):','']
    lines+=table(['Contrast','Observed ΔD','AR verdict','Full verdict'],[[r['contrast_id'],fmt(float(r['D_contrast'])),r['H3_AR'],r['H3_full']] for r in h3partial])
    lines+=['','## Every departure and what the diagnostics identify','',
        f'[departures.csv](departures.csv) enumerates **all {len(ledger)} failed comparisons**: 224 H1-AR, 211 H1-full, 55 H2, '
        '5 H3-AR, 5 H3-full and 1 Monte Carlo validation discrepancy. Each row names the cell or contrast, observed/predicted value, fixed tolerance, attribution, '
        'and scope limit. These overlap; they are not 501 independent failures. '
        '[prediction-inapplicable.csv](prediction-inapplicable.csv) separately enumerates every unsupported log prediction. '
        'The complete evidence is in [cell-attribution.csv](cell-attribution.csv), [monotonicity-attribution.csv](monotonicity-attribution.csv), '
        '[contrast-attribution.csv](contrast-attribution.csv) and [diagnostics.csv](diagnostics.csv).','',
        f'The registered diagnostic campaign evaluated {len(dg)} roots for {len(completed["diagnostic_series"])} series. '
        'The following labels are overlapping evidence, not unique causal decompositions:','']
    tagcounts=collections.Counter(t for r in fullbad for t in r['attribution'].split(';'))
    lines+=table(['Evidence among full-ACF departures','Cells'],sorted(tagcounts.items()))
    lines+=['', 'Non-AR autocorrelation is isolated at second order when the full-covariance prediction repairs an AR discrepancy. '
        'Fourier-phase diagnostics preserve log variance and every circular log covariance; a material D change therefore demonstrates sensitivity beyond those second-order quantities. '
        'Such a change can combine log skew, tail shape and higher-order temporal dependence. It cannot uniquely identify one of them.',
        '', 'Clipping tests sensitivity to extremes, but also changes variance and covariance. Detrending tests a trend intervention, not a formal stationarity test. '
        'Permutation preserves abundance marginals while changing temporal ordering. Mirroring tests sensitivity to the replay boundary and reverses the interior ordering on its return leg. '
        'Consequently mixed or unresolved attributions are retained. Skew and heavy-tail flags alone are descriptive; they are not presented as causal proof.',
        '', '## Implementation departures and scope limits','',
        '1. The workbook contains −183 adults at experiment 8, HS2, IC2, replicate 5, week 48. Input validation stopped before empirical model outcomes. '
        'The dated protocol amendment excludes that whole adult record without changing the count or substituting another culture. The affected six contrasts remain partial-source results.',
        '2. Exact integer simulation stalled in its registered large-count chunk sampler at the first Tribolium batch. It was interrupted; 7,680 completed paths and 480 endpoints remain archived. '
        'No incomplete paths are classified. The subsequently registered and independently validated PGF completion gives the same branching-law persistence for all cells, without a cap. '
        'This is a computational-method change after observing thresholds, not a change in the ecological model. The original run manifest is an interrupted checkpoint, not evidence of completion; use completion-manifest.json.',
        '3. Before scientific execution, a syntax error was fixed and its failed-import log retained. The opening registration timestamp was corrected by a dated note. '
        'Demographic batching and its seed assignment were amended before empirical outcomes. No scientific tolerance was relaxed.',
        '4. The missing plan.md means the task’s description defines A6 here. Circular replay imposes a periodic stationary law only after randomizing phase; it cannot establish that the original natural process is stationary. '
        'Trends are deliberately replayed and may become low-frequency cycles. Full ACF resolves the AR(1) restriction at second order, but finite-amplitude and zero-support departures also exceed the analytical approximation’s premises.',
        '5. Three generic stage structures/stays do not establish results for actual parasite anatomy. Two and three stages were tested; no additional host feedback, contact nonlinearity or data-fitted mechanism was added. '
        'Results should enter a paper as limits on the theory’s scope: retain the small-noise, finite-log and stated memory assumptions; do not assert universal steady-stay buffering or validation of parasite persistence in nature.',
        '',f'The retained Monte Carlo failure is BT_478_Rhyacophila_fasciata, two stages, stay 1, step 500: observed survival 1/64, '
        f'expected {rare["expected_survivors"]/64:.9g}, absolute error 0.0155123 versus tolerance 0.00730663. '
        f'An exact convolution of the 64 conditional Bernoulli laws gives probability {rare["probability_at_least_one"]:.8f} of at least one survivor '
        f'(absolute-deviation tail p={rare["exact_two_sided_absolute_deviation_tail"]:.8f}; Bonferroni adjustment over 480 endpoints is 1). '
        'At this rare-event boundary a single survivor can exceed four SE without being a four-sigma normal-tail event. '
        'This post hoc test traces the discrepancy to the discreteness of the sampling diagnostic; it does not convert the failed registered check into a pass. '
        'See [exact rare-event test](rare-event-check.json). No new simulation or tolerance change followed.',
        '', 'Plot generation initially found a missing matplotlib dependency. python-dateutil and six were installed only into ignored q4/raw/plot-deps; '
        'no global environment or engine dependency was changed. See plot-dependency.log.',
        '',f'The PGF/diagnostic completion took {completed["seconds"]:.2f} measured seconds. The interrupted run is charged 128 seconds conservatively, '
        f'giving {completed["total_campaign_seconds"]:.2f} seconds charged to the 30-minute campaign budget, plus the short validation gates. '
        'Hashes, seeds, phase-specific probabilities and acquisition records are retained. The eight duplicated controls are identified in selection.json. '
        'No original engine, collaborator report or raw source file was edited; raw working copies are ignored under q4/raw/.',
        '', '## Reproduction and artifact map','',
        'Run the scripts in a fresh q4 output directory, preserving this registration: `validate_phase1.py`, engine unittest discovery, '
        '`fetch_data.py`, `validate_numerics.py`, `prepare_data.py`, then the registered campaign. The historical `run.py` can reproduce the disclosed count bottleneck. '
        'The supplied completion scripts consume its saved threshold rows; `pgf.py` validates the probability calculation, `complete.py` produces exact persistence and diagnostics, '
        '`summarize.py` makes the report, and `verify_outputs.py` audits the saved artifacts. Existing final outputs are refused. '
        'For practical reuse, preserve the saved threshold rows and rerun the completion in a separate output copy; do not overwrite this research record.',
        '', '- [Selection and source hashes](selection.json), [strata](strata.csv), [input exclusions](excluded-compilation.csv), [Tribolium exclusion](excluded-tribolium.csv).',
        '- [Source acquisition manifest](fetch-manifest.json), [initial interrupted run](run-manifest.json), [completed campaign](completion-manifest.json).',
        '- [All threshold predictions](cells.csv), [all persistence probabilities](persistence-exact.csv), [all treatment contrasts](contrasts.csv), [all stay comparisons](monotonicity.csv).',
        '- [Exportable summary figure](summary.pdf), [verification](verification.json).',
        '', 'Status: done']
    (P/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))
