"""Decision, gateway and local UI contracts; no public pages required."""
import copy
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from detector_runtime import browser_network as network
from detector_runtime.behavior import combine, rules_for
from detector_runtime.capture import CaptureError
from detector_runtime.ui import make_server
from detector_runtime import browser_evaluation as evaluation


class StubDetector:
    def __init__(self,root):
        self.root=root
        self.spec={"model_name":"test"}

    def predict_html(self,html,url,basis):
        score=.7 if '<form' in html else .2
        return {"status":"prediction_complete","display_result":"Potential phishing" if score>=.5 else "No phishing predicted",
                "features":{"form_count":int('<form' in html)},"quality_flags":{},"active_quality_flags":[],
                "needs_review":False,"phishing_score":score,"static_model_prediction":"phish" if score>=.5 else "benign",
                "page_url":url,"capture_basis":basis,"html_characters":len(html),"model":{"feature_count":106}}


def observation():
    return {"status":"browser_capture_complete","source":{"html":"<p>Wait</p>","page_url":"https://example.com/"},
            "rendered_html":"<form></form>","page_url":"https://example.com/",
            "signals":{"forms_added_after_ready":1,"maximum_passwordlike_controls":1,"main_frame_navigations":1,
                "test_password_request_attempts":1,"cross_origin_test_password_attempts":1,"base64_decode_calls":0,
                "external_scripts_loaded":0},
            "quality":{"critical_resources_failed":False,"event_storage_cut":False,"request_storage_cut":False,
                       "field_storage_cut":False,"subframes_present":False},
            "capture":{"interaction_requests_forwarded":False},"critical_resource_failures":[]}


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.detector=StubDetector(Path('.'))

    def test_rendered_model_decides_and_source_is_retained(self):
        r=combine(self.detector,observation())
        self.assertEqual(r['source_analysis']['static_model_prediction'],'benign')
        self.assertEqual(r['evaluated_prediction'],'phish')
        self.assertTrue(r['capture_basis_verified'])
        self.assertEqual(r['feature_changes']['form_count'],{'source':0,'rendered':1})

    def test_cross_origin_does_not_override_benign_label(self):
        c=observation();c['rendered_html']='<p>Ordinary login control</p>'
        r=combine(self.detector,c)
        self.assertEqual(r['evaluated_prediction'],'benign')
        self.assertTrue(r['needs_review'])
        self.assertTrue(all(rule['effect']=='review_evidence' for rule in r['behavior']['rules']))

    def test_incomplete_required_resource_abstains(self):
        c=observation();c['quality']['critical_resources_failed']=True
        r=combine(self.detector,c)
        self.assertEqual(r['status'],'inconclusive_capture')
        self.assertIsNone(r['evaluated_prediction'])
        self.assertEqual(r['static_model_prediction'],'phish')

    def test_truncated_observation_abstains(self):
        for flag in ['event_storage_cut','request_storage_cut','field_storage_cut']:
            c=observation();c['quality'][flag]=True
            self.assertIsNone(combine(self.detector,c)['evaluated_prediction'])

    def test_browser_failure_never_becomes_benign(self):
        r=combine(self.detector,{'status':'browser_capture_failed','error':'Timeout'})
        self.assertIsNone(r['static_model_prediction'])
        self.assertFalse(r['model_prediction_made'])


class GatewayTests(unittest.TestCase):
    def test_private_addresses_rejected_before_connect(self):
        with patch.object(network,'_PinnedHTTP') as connection:
            with self.assertRaises(CaptureError):
                network.fetch_public_resource('http://127.0.0.1/')
            connection.assert_not_called()

    def test_nonweb_port_rejected(self):
        with self.assertRaises(CaptureError):
            network.fetch_public_resource('https://example.com:22/')

    def test_no_browser_credentials_forwarded_and_private_redirect_rejected(self):
        class Response:
            status=302
            def getheader(self,name):
                return 'http://127.0.0.1/internal' if name.lower()=='location' else None
        class Connection:
            sent=None
            def __init__(self,*args):pass
            def request(self,method,target,headers):Connection.sent=(method,headers)
            def getresponse(self):return Response()
            def close(self):pass
        def resolver(host,port):
            if host=='127.0.0.1':raise CaptureError('private')
            return ['93.184.216.34']
        with patch.object(network,'_PinnedHTTPS',Connection),patch.object(network,'_public_addresses',resolver):
            with self.assertRaises(CaptureError):network.fetch_public_resource('https://example.com/')
        self.assertEqual(Connection.sent[0],'GET')
        self.assertNotIn('Cookie',Connection.sent[1])
        self.assertNotIn('Authorization',Connection.sent[1])


class InterfaceTests(unittest.TestCase):
    def test_demo_and_browser_reach_integrated_pipeline_with_csrf(self):
        with tempfile.TemporaryDirectory() as temp:
            detector=StubDetector(Path(temp))
            server=make_server(detector,0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            port=server.server_port
            try:
                connection=http.client.HTTPConnection('127.0.0.1',port)
                connection.request('GET','/')
                text=connection.getresponse().read().decode()
                token=text.split('name="detector-token" content="')[1].split('"')[0]
                headers={'Content-Type':'application/json','Origin':f'http://127.0.0.1:{port}','X-Detector-Token':token}
                for mode,target in [('demo','detector_runtime.behavior_examples.analyze_demo'),('browser','detector_runtime.behavior.analyze_browser_url')]:
                    with patch(target,return_value={'status':'prediction_complete','display_result':'Test','phishing_score':.4}) as call:
                        connection.request('POST','/api/analyze',json.dumps({'mode':mode,'url':'https://example.com/'}),headers)
                        response=connection.getresponse();r=json.loads(response.read())
                        self.assertEqual(response.status,200)
                        call.assert_called_once()
                        self.assertTrue((Path(temp)/'reports'/'webpage_analysis_v1'/r['saved_report_filename']).is_file())
                connection.request('POST','/api/analyze',json.dumps({'mode':'browser','url':'https://example.com/'}),{'Content-Type':'application/json'})
                response=connection.getresponse();response.read()
                self.assertEqual(response.status,403)
            finally:
                server.shutdown();server.server_close();thread.join()


class BatchTests(unittest.TestCase):
    def rows(self):
        return [{'ordinal':i+1,'csv_row':i+2,'url':f'https://example.com/{i}',
                 'normalized_url':f'https://example.com/{i}','label':label,
                 'label_source':'Independent fixture label','label_checked_at':'2026-10-11'}
                for i,label in enumerate(['benign','phish','benign','phish'])]

    def test_metrics_count_failures_separately(self):
        rows=self.rows()
        records={1:{'analysis':{'status':'prediction_complete','evaluated_prediction':'benign'}},
                 2:{'analysis':{'status':'prediction_complete','evaluated_prediction':'benign'}},
                 3:{'analysis':{'status':'browser_capture_failed','evaluated_prediction':None}},
                 4:{'analysis':{'status':'inconclusive_capture','evaluated_prediction':None}}}
        report=evaluation.summarize(rows,records,True)
        self.assertEqual(report['scored'],2)
        self.assertEqual(report['scored_coverage'],.5)
        self.assertEqual(report['classification_metrics_on_scored_pages']['confusion_matrix'],{'TN':1,'FP':0,'FN':1,'TP':0})
        self.assertEqual(report['classification_metrics_on_scored_pages']['accuracy'],.5)

    def test_paired_comparison_uses_the_same_rows(self):
        rows=self.rows()
        records={1:{'analysis':{'status':'prediction_complete','evaluated_prediction':'benign',
                     'source_analysis':{'status':'prediction_complete','static_model_prediction':'phish'}}},
                 2:{'analysis':{'status':'prediction_complete','evaluated_prediction':'phish',
                     'source_analysis':{'status':'prediction_complete','static_model_prediction':'benign'}}},
                 3:{'analysis':{'status':'browser_capture_failed'}},
                 4:{'analysis':{'status':'prediction_complete','evaluated_prediction':'phish',
                     'source_analysis':{'status':'inconclusive_capture','static_model_prediction':'benign'}}}}
        paired=evaluation.summarize(rows,records,True)['paired_source_rendered_comparison']
        self.assertEqual(paired['paired_pages'],2)
        self.assertEqual(paired['source_metrics']['accuracy'],0)
        self.assertEqual(paired['rendered_metrics']['accuracy'],1)

    def test_completed_rows_are_not_requested_twice_and_changed_input_rejected(self):
        class Detector:
            def describe(self):return {'model':'test'}
        with tempfile.TemporaryDirectory() as temp:
            calls=[]
            def analyzer(detector,url):
                calls.append(url)
                return {'status':'prediction_complete','evaluated_prediction':'benign','phishing_score':.2}
            rows=self.rows();root=Path(temp)
            evaluation.run_evaluation(Detector(),rows,b'input',root,{'code':'v1'},analyzer)
            evaluation.run_evaluation(Detector(),rows,b'input',root,{'code':'v1'},analyzer)
            self.assertEqual(len(calls),4)
            with self.assertRaises(ValueError):
                evaluation.run_evaluation(Detector(),rows,b'changed',root,{'code':'v1'},analyzer)
            self.assertEqual(len(calls),4)

    def test_interrupted_capture_is_retained_without_retry(self):
        class Detector:
            def describe(self):return {'model':'test'}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=self.rows()
            with self.assertRaises(KeyboardInterrupt):
                evaluation.run_evaluation(Detector(),rows,b'input',root,{},lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
            calls=[]
            report=evaluation.run_evaluation(Detector(),rows,b'input',root,{},
                lambda detector,url:calls.append(url) or {'status':'browser_capture_failed','evaluated_prediction':None})
            self.assertEqual(len(calls),3)
            self.assertEqual(report['outcomes']['interrupted_capture'],1)
            self.assertEqual(report['attempted'],4)
            self.assertIsNone(report['classification_metrics_on_scored_pages']['accuracy'])

    def test_checkpoint_tampering_rejected(self):
        import sqlite3
        from contextlib import closing
        class Detector:
            def describe(self):return {'model':'test'}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=self.rows()
            evaluation.run_evaluation(Detector(),rows,b'input',root,{},lambda *args:{'status':'browser_capture_failed','evaluated_prediction':None})
            with closing(sqlite3.connect(root/'checkpoint.sqlite3')) as conn:
                ordinal,payload=conn.execute('SELECT ordinal,payload FROM attempts LIMIT 1').fetchone()
                record=json.loads(payload);record['analysis']['status']='changed'
                conn.execute('UPDATE attempts SET payload=? WHERE ordinal=?',(json.dumps(record),ordinal))
                conn.commit()
            with self.assertRaises(ValueError):
                evaluation.run_evaluation(Detector(),rows,b'input',root,{},lambda *args:self.fail('Unexpected retry'))


if __name__=='__main__':
    unittest.main()
