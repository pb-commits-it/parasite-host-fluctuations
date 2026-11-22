"""Fix1 claims: registered two-month binning and host-independent calendar weather lags.

Synthetic checks only. Dates/counts must agree exactly; independently computed
least-squares residual variance and lag-1 correlation must agree within 1e-10.
Run: python3 -I casestudies/lpd/test_fix1.py
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calendar
import csv
import datetime as dt
import io
import tempfile
import unittest
import zipfile
import numpy as np
import helpers_continued as h
from weather_continued import MatchedArchive, calendar_windows


def series():
    t = np.arange(2000, 2022)
    return dict(series_id='synthetic', environment_type='field', latitude=40.,
                longitude=-100., realm_detail='Terrestrial', database='BioTIME',
                interval_class='annual', samples_per_year=1., steps=t,
                step_years=t, step_months=np.zeros(len(t), int),
                subperiod=np.zeros(len(t), int), bin_days=0, step_dates_iso=None,
                values=np.exp(.3*np.sin(np.arange(len(t))*1.7)+.02*np.arange(len(t))))


def archive():
    a = MatchedArchive.__new__(MatchedArchive)
    a.daymet = {}; a.ghcn_stations = []; a.ghcn_data = {}
    lat, lon = 40., -100.
    i = round((90-lat)*24-.5); j = round((lon+180)*24-.5)
    a.terraclimate = {(i,j): dict(cell_lat=lat, cell_lon=lon,
        ym={(y,m): (8+np.sin((y-1999)*.91)+.3*np.cos(m),
                     30+np.cos((y-1999)*1.3))
            for y in range(1999,2022) for m in range(1,13)})}
    return a


class Binning(unittest.TestCase):
    def test_registered_boundaries(self):
        for counts, expected in [([], 'annual'), ([2,2], 'annual'),
                ([1,1,2,2], 'annual'), ([2,2,2], 'subannual_seasonal'),
                ([2,2,3,3], 'subannual_seasonal'), ([5,5,5], 'subannual_seasonal'),
                ([6,6,6], 'subannual_monthly')]:
            with self.subTest(counts=counts):
                self.assertEqual(h.biotime_bin_mode(counts)[0], expected)

    def test_raw_two_month_assemblage_rebuilds_quarters_and_effort(self):
        # Only ten years: annual aggregation cannot yield 15 observations.
        # Two seasons with two sampling events each must produce twenty bins.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'gpdd').mkdir(); (root/'biotime').mkdir()
            for name, key in [('taxon','TaxonID'), ('biotope','BiotopeID'),
                              ('location','LocationID'), ('main','MainID'), ('data','MainID')]:
                (root/'gpdd'/f'{name}.csv').write_text(key+'\n')
            (root/'biotime/BioTIMEMetadata_24_06_2021.csv').write_text(
                'STUDY_ID,TAXA,REALM,LICENSE\n1,Birds,Terrestrial,CC-BY\n')
            out=io.StringIO(); fields=['STUDY_ID','YEAR','MONTH','SAMPLE_DESC',
                'PLOT','sum.allrawdata.ABUNDANCE','GENUS_SPECIES','GENUS']
            w=csv.DictWriter(out,fieldnames=fields); w.writeheader()
            for year in range(2000,2010):
                for month, amount in [(2,10), (8,30)]:
                    for event, species_name in [('a','Anas platyrhynchos'), ('b','Other bird')]:
                        w.writerow(dict(STUDY_ID=1,YEAR=year,MONTH=month,
                            SAMPLE_DESC=f'{year}_{month}_{event}',PLOT='p',
                            **{'sum.allrawdata.ABUNDANCE':amount,
                               'GENUS_SPECIES':species_name,'GENUS':species_name.split()[0]}))
            with zipfile.ZipFile(root/'biotime/BioTIMEQuery_24_06_2021.zip','w') as z:
                z.writestr('BioTIMEQuery_24_06_2021.csv',out.getvalue())
            rows,_=h.load_all_raw_series(root)
            rows=[r for r in rows if r['species']=='Anas platyrhynchos']
            self.assertEqual(len(rows),2)  # pooled and fixed plot
            for r in rows:
                self.assertEqual(r['interval_class'],'subannual_seasonal')
                np.testing.assert_array_equal(r['steps'],[y*4+q for y in range(2000,2010) for q in [0,2]])
                np.testing.assert_array_equal(r['values'],[5.,15.]*10)


class Weather(unittest.TestCase):
    def check_fit(self,s,a):
        mask=s['values']>0; result,records=a.match(s,mask)
        self.assertTrue(result['matched'],result)
        cur={r['host_step']:r for r in records if r['weather_role']=='current'}
        lag={r['host_step']:r for r in records if r['weather_role']=='lag'}
        steps=[int(t) for t in s['steps'][mask] if int(t) in cur and int(t) in lag]
        t=np.array(steps,float); pos={int(t):i for i,t in enumerate(s['steps'])}
        y=np.log([s['values'][pos[int(t)]] for t in steps])
        X=np.column_stack([np.ones(len(t)),t-t.mean(),
                           [cur[k]['temperature_c'] for k in steps],
                           [lag[k]['temperature_c'] for k in steps],
                           [cur[k]['precipitation_mm'] for k in steps]])
        e=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
        rank=np.linalg.matrix_rank(X)
        adjacent=np.diff(t)==1
        r1=np.mean((e[:-1]*e[1:])[adjacent])/np.mean(e*e)
        self.assertEqual(result['n_adjusted'],len(t))
        self.assertAlmostEqual(result['residual_metrics']['sigma2'],e@e/(len(t)-rank),delta=1e-10)
        self.assertAlmostEqual(result['residual_metrics']['r1'],r1,delta=1e-10)
        return result,cur,lag

    def test_complete_grid_known_ols_and_first_observation(self):
        r,_,lag=self.check_fit(series(),archive())
        self.assertEqual(r['n_adjusted'],22)
        self.assertEqual(lag[2000]['start_date'],'1999-01-01')

    def test_zero_host_predecessor_keeps_weather(self):
        s=series();s['values'][7]=0
        r,cur,lag=self.check_fit(s,archive())
        self.assertEqual(r['n_adjusted'],21)
        self.assertNotIn(2007,cur)
        self.assertEqual(lag[2008]['start_date'],'2007-01-01')

    def test_absent_host_predecessor_does_not_compress_time(self):
        s=series()
        for k in ['steps','step_years','step_months','subperiod','values']:
            s[k]=np.delete(s[k],7)
        r,_,lag=self.check_fit(s,archive())
        self.assertEqual(r['n_adjusted'],21)
        self.assertEqual(lag[2008]['start_date'],'2007-01-01')

    def test_missing_weather_excludes_current_and_next_host(self):
        a=archive();ym=next(iter(a.terraclimate.values()))['ym']
        for m in range(1,13):del ym[2007,m]
        r,_,_=self.check_fit(series(),a)
        self.assertEqual(r['n_adjusted'],20)

    def test_calendar_rollover_month_quarter_week_and_fortnight(self):
        for iv,spy,bd,start,expected in [
            ('annual',1,0,'2020-01-01',('2019-01-01','2019-12-31')),
            ('subannual_monthly',12,0,'2020-03-01',('2020-02-01','2020-02-29')),
            ('subannual_seasonal',4,0,'2020-01-01',('2019-10-01','2019-12-31')),
            ('subannual_biweekly_weekly',52,7,'2020-01-06',('2019-12-30','2020-01-05')),
            ('subannual_biweekly_weekly',26,14,'2020-01-06',('2019-12-23','2020-01-05'))]:
            with self.subTest(iv=iv,bd=bd):
                s=series();d=dt.date.fromisoformat(start)
                s.update(interval_class=iv,samples_per_year=spy,bin_days=bd,
                    steps=np.array([0]),step_years=np.array([d.year]),step_months=np.array([d.month]),
                    subperiod=np.array([0]),step_dates_iso=[start] if bd else None)
                _,lag=calendar_windows(s,np.array([True]))
                self.assertEqual(tuple(d.isoformat() for d in lag[0]),expected)

    def test_gpdd_representative_month_uses_host_interval(self):
        s=series();s.update(database='GPDD',interval_class='subannual_seasonal',samples_per_year=2,
            steps=np.array([1]),step_years=np.array([2020]),step_months=np.array([7]),subperiod=np.array([7]))
        cur,lag=calendar_windows(s,np.array([True]))
        self.assertEqual(cur[0],(dt.date(2020,7,1),dt.date(2020,7,31)))
        self.assertEqual(lag[0],(dt.date(2020,1,1),dt.date(2020,1,31)))

    def test_station_eligibility_counts_current_windows_only(self):
        a=archive();a.ghcn_stations=[(40.,-100.,'unused',0)]
        # Actual tuple contract: station id, latitude, longitude, elevation.
        a.ghcn_stations=[('near',40.,-100.,0),('far',40.1,-100.,0)]
        a.ghcn_data={sid:{(y,m):float(y%7+m) for y in years for m in range(1,13)}
                     for sid,years in [('near',range(1999,2008)),('far',range(1999,2022))]}
        s=series();cur,lag=calendar_windows(s,s['values']>0)
        # The near station has 8 current + 9 lag windows, but cannot qualify.
        got=a.station_exact(40.,-100.,cur+lag,current_count=len(cur))
        self.assertEqual(got['station_id'],'far')
        a.terraclimate={}
        r,records=a.match(s,s['values']>0)
        self.assertTrue(r['matched']);self.assertEqual(r['source'],'NOAA_GHCNm_v4')
        self.assertEqual(r['n_adjusted'],22)
        self.assertEqual({v['station_id'] for v in records},{'far'})


if __name__=='__main__':unittest.main(verbosity=2)
