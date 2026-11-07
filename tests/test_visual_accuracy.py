"""Offline evidence/size/duration boundaries, with real-model transport forbidden."""
import copy
import json
import subprocess
import unittest
from unittest.mock import Mock, patch

from pipeline import providers as p, visual_contract as vc, openai_provider as adapter
from pipeline.output_checks import output_errors
import test_openai_upgrade as fixtures


def check(result, modality='image', **kwargs):
    return output_errors(result, modality) + vc.errors(result, modality, **kwargs)


class VisualAccuracyTests(unittest.TestCase):
    setUp = fixtures.OpenAIUpgradeTests.setUp

    def test_literal_observations_with_referenced_evidence_pass(self):
        self.assertEqual(check(fixtures.valid_result(True)), [])

    def test_visible_body_not_mappable_to_state_is_not_erased(self):
        r = fixtures.valid_result(True)
        r.update(insufficient_evidence=True, candidates=[], insufficient_reason='可见身体，但张力与耳基不明',
                 translation='当前不能确定状态；身体与四肢仍可见。')
        r['input_assessment']['observable_main_axes'] = []
        self.assertEqual(check(r), [])
        self.assertEqual(r['literal_observations'][0]['status'], 'observed')

    def test_image_intent_emotion_and_motion_contamination_rejected(self):
        for text in ('放松侧卧', '它想要吃东西', 'mouth open, panting', '反复摆尾', '开心地张嘴'):
            r = fixtures.valid_result(True)
            r['literal_observations'][0]['description'] = text
            r['candidates'][0]['evidence'][0]['observation'] = text
            self.assertTrue(any('description 混入' in e for e in check(r)), text)

    def test_unobservable_requires_null_and_specific_reason(self):
        r = fixtures.valid_result(True)
        r['literal_observations'].append({'id':'o3','source':'visual','channel':'ear_position',
            't':0,'status':'unobservable','description':None,'reason':'耳基被长毛遮住'})
        self.assertEqual(check(r), [])
        r['literal_observations'][-1]['description'] = '耳朵向前'
        self.assertTrue(check(r))

    def test_candidate_cannot_cite_missing_or_unobservable_fact(self):
        for change in ('missing', 'unobservable'):
            r = fixtures.valid_result(True)
            if change == 'missing': r['candidates'][0]['evidence'][0]['observation_ids']=['fabricated']
            else: r['literal_observations'][0].update(status='unobservable', description=None, reason='裁切')
            self.assertTrue(any('不存在/不可观测' in e for e in check(r)))

    def test_candidate_cannot_rephrase_fact_as_intent(self):
        r = fixtures.valid_result(True)
        r['candidates'][0]['evidence'][0]['observation'] = '它在等待关注'
        self.assertTrue(any('不能新增推断' in e for e in check(r)))

    def test_first_person_or_definitive_summary_rejected(self):
        for text in ('我这会儿很开心。', '这只猫很放松。'):
            r = fixtures.valid_result(True); r['translation'] = text
            self.assertTrue(check(r))

    def test_alternative_and_knowledge_basis_required(self):
        for field, value in (('alternative_explanations', []), ('alternative_explanations', ['休息可能'])):
            r = fixtures.valid_result(True); r['candidates'][0][field] = value
            self.assertTrue(check(r))
        r = fixtures.valid_result(True); r['candidates'][0]['evidence'][0]['knowledge_basis']=''
        self.assertTrue(check(r))

    def test_main_axis_requires_same_channel_literal_observation(self):
        r = fixtures.valid_result(True); r['input_assessment']['observable_main_axes']=['ear_position']
        r['literal_observations'] = r['literal_observations'][:1]
        self.assertTrue(any('缺少同通道' in e for e in check(r)))

    def test_no_target_can_have_good_quality_without_candidates(self):
        r = fixtures.valid_result(True)
        r.update(insufficient_evidence=True,candidates=[],literal_observations=[],
                 insufficient_reason='没有猫',translation='没有目标，无法判断。')
        r['input_assessment'].update(target_species_present=False,animal_count=0,target_unambiguous=False,
                                      target_description=None,observable_main_axes=[])
        self.assertEqual(check(r), [])
        r['input_assessment']['target_unambiguous'] = True
        self.assertTrue(check(r))

    def test_multiple_animals_require_target_localization(self):
        r = fixtures.valid_result(True); r['input_assessment'].update(animal_count=2,target_description=None)
        self.assertTrue(any('target_description' in e for e in check(r)))
        r['input_assessment']['target_description']='左侧靠近白门的黑猫'
        self.assertEqual(check(r), [])

    def test_video_must_use_actual_frame_times_and_available_audio(self):
        r = fixtures.valid_result(True)
        for o in r['literal_observations']: o['t']=0.5
        self.assertEqual(check(r,'video',frame_times=[0.5,1.0]), [])
        self.assertTrue(check(r,'video',frame_times=[0,1.0]))
        r['literal_observations'][0].update(source='audio',channel='vocalization')
        self.assertTrue(check(r,'video',frame_times=[0.5],audio_available=False))

    def test_unvalidated_pain_scores_rejected(self):
        for value in ({'normalized_score':0}, {'status':'not_assessed','reason':'未知','total':0}, None):
            r = fixtures.valid_result(True); r['pain_indicators']=value
            self.assertTrue(any('pain_indicators' in e for e in check(r)))

    def test_malformed_nested_types_fail_closed_not_internal_error(self):
        changes = [('literal_observations', [None, {'id':{},'source':{},'channel':[],'status':[]}]),
                   ('candidates',[{'label':'x','confidence':0.6,'evidence':[{'observation_ids':[{}]}]}])]
        for field,value in changes:
            r=fixtures.valid_result(True);r[field]=value
            with patch.object(p,'_post_json',return_value=fixtures.response_body(r)):
                out=self.provider.call_omni(str(self.image),'image',self.prompt)
            self.assertEqual(out['_error_kind'],'contract_error')

    def test_visual_errors_do_not_retry_or_repair_original_response(self):
        r=fixtures.valid_result(True);r['candidates'][0]['confidence']=0.2
        raw=json.dumps(r,ensure_ascii=False)
        with patch.object(p,'_post_json',return_value=fixtures.response_body(r,raw=raw)) as post:
            out=self.provider.call_omni(str(self.image),'image',self.prompt)
        self.assertEqual(post.call_count,1)
        self.assertEqual(out['_error_kind'],'contract_error')
        self.assertEqual(self.provider.last_raw_text,raw)
        self.assertNotIn('_contract_repairs',out)
        self.assertNotIn('_dropped_candidates',out)

    def test_audio_prompt_does_not_receive_visual_revision(self):
        with patch.object(p,'_post_json',return_value=fixtures.response_body(fixtures.audio_result(),audio=True)) as post:
            self.provider.call_omni(str(self.audio),'audio',self.prompt)
        self.assertNotIn(vc.VERSION,json.dumps(post.call_args.args[1],ensure_ascii=False))

    def test_arbitrary_json_field_types_never_raise_in_validator(self):
        values=[None,True,1,[],{},'unknown']
        for field in ('literal_observations','candidates','input_assessment','translation','pain_indicators'):
            for value in values:
                r=fixtures.valid_result(True);r[field]=value
                check(r)
        for field in ('observable_main_axes','limitations','target_description','animal_count'):
            for value in values:
                r=fixtures.valid_result(True);r['input_assessment'][field]=value
                check(r)


class VideoLimitsTests(unittest.TestCase):
    setUp = fixtures.OpenAIUpgradeTests.setUp

    def metadata(self, **change):
        stream={'codec_type':'video','width':1920,'height':1080,'avg_frame_rate':'30/1'}
        stream.update(change)
        return {'format':{'duration':'15'},'streams':[stream]}

    def probe(self, data):
        with patch.object(adapter.subprocess,'run',return_value=Mock(stdout=json.dumps(data))):
            return adapter.media_info(str(self.audio),30,kind='video')

    def test_exact_limits_and_portrait_pass(self):
        self.probe(self.metadata());self.probe(self.metadata(width=1080,height=1920))

    def test_over_duration_pixels_and_fps_rejected(self):
        data=self.metadata();data['format']['duration']='15.01'
        cases=[data,self.metadata(width=3840,height=2160),self.metadata(avg_frame_rate='60/1'),self.metadata(width=2500,height=100)]
        for case in cases:
            with self.assertRaises(ValueError):self.probe(case)

    def test_unknown_nonfinite_and_disagreeing_duration_rejected(self):
        for duration in ('nan','inf','0','-1','garbled'):
            data=self.metadata();data['format']['duration']=duration
            with self.assertRaises(ValueError):self.probe(data)
        with self.assertRaises(ValueError):self.probe(self.metadata(duration='16'))

    def test_oversize_rejected_before_probe_or_api(self):
        with self.audio.open('r+b') as f:f.truncate(10*1024*1024+1)
        with patch.object(adapter.subprocess,'run') as probe,patch.object(p,'_post_json') as post:
            out=self.provider.call_omni(str(self.audio),'video',self.prompt)
        probe.assert_not_called();post.assert_not_called()
        self.assertEqual(out['_meta']['attempts'],0)

    def test_audio_limit_remains_thirty_seconds(self):
        data={'format':{'duration':'30'},'streams':[{'codec_type':'audio'}]}
        with patch.object(adapter.subprocess,'run',return_value=Mock(stdout=json.dumps(data))):
            adapter.media_info(str(self.audio),30)
            with self.assertRaises(ValueError):adapter.media_info(str(self.audio),30,kind='video')

    def test_bad_video_stops_before_frame_decode_and_model(self):
        with patch.object(adapter.subprocess,'run',return_value=Mock(stdout=json.dumps(self.metadata(width=3840,height=2160)))), \
             patch('pipeline.video.select_frames') as frames, patch.object(p,'_post_json') as post:
            out=self.provider.call_omni(str(self.audio),'video',self.prompt)
        self.assertIn('error',out);frames.assert_not_called();post.assert_not_called()

    def test_direct_opencv_entry_rejects_before_first_frame(self):
        from pipeline import video
        capture=Mock();capture.isOpened.return_value=True
        capture.get.side_effect=[3840,2160,30,60]
        with patch.object(video.cv2,'VideoCapture',return_value=capture):
            with self.assertRaises(video.MediaResourceLimitError):video._open(str(self.audio))
        capture.read.assert_not_called();capture.release.assert_called_once()

    def test_decoder_rejection_is_media_error_without_api_attempt(self):
        with patch.object(adapter.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['ffprobe'])), \
             patch.object(p,'_post_json') as network:
            out=self.provider.call_omni(str(self.audio),'video',self.prompt)
        self.assertEqual(out['_error_kind'],'media_validation_error')
        self.assertEqual(out['_meta']['attempts'],0)
        network.assert_not_called()

    def test_probe_timeout_missing_tool_and_signal_are_not_damaged_media(self):
        for failure in (subprocess.TimeoutExpired(['ffprobe'],15), FileNotFoundError('ffprobe'),
                        subprocess.CalledProcessError(-9,['ffprobe'])):
            with self.subTest(failure=type(failure).__name__), \
                 patch.object(adapter.subprocess,'run',side_effect=failure):
                with self.assertRaises(type(failure)):
                    adapter.media_info(str(self.audio),30,kind='video')


class ImageEvidencePolicyTests(unittest.TestCase):
    def test_two_channels_allow_hypothesis_but_do_not_force_it(self):
        r=fixtures.valid_result(True)
        self.assertEqual(check(r),[])
        r.update(candidates=[],insufficient_evidence=True,
                 insufficient_reason='两通道可读，但存在无法区分的替代解释',translation='只能描述瞬间形态。')
        self.assertEqual(check(r),[])

    def test_one_channel_repeated_or_only_coat_cannot_support_state(self):
        for channel in ('body_posture','coat','position'):
            r=fixtures.valid_result(True)
            r['literal_observations'][1]['channel']=channel
            self.assertTrue(any('两个独立' in e for e in check(r)))
        r=fixtures.valid_result(True)
        r['candidates'][0]['evidence'][1]['signal_confidence']='低'
        self.assertTrue(any('两个独立' in e for e in check(r)))

    def test_three_channels_still_allow_abstention(self):
        r=fixtures.valid_result(True)
        r['literal_observations'].append({'id':'o3','source':'visual','t':0,'channel':'tail',
            'status':'observed','description':'尾部弯曲','reason':None})
        r.update(candidates=[],insufficient_evidence=True,insufficient_reason='多义形态，场景未知',translation='无法确定状态。')
        self.assertEqual(check(r),[])

    def test_no_conflicting_rules_and_no_shared_knowledge_mutation(self):
        from pipeline import prompts
        from datetime import datetime
        for species in ('cat','dog'):
            kb=prompts.load_knowledge(species); original=copy.deepcopy(kb)
            prompt=prompts.omni_prompt(species,'unavailable',kb,{},now=datetime(2026,11,7),modality='image')
            text=prompt.system+prompt.user
            self.assertNotIn(prompts._LEGACY_CHANNEL_RULE,text)
            self.assertNotIn('可单独作为判读依据',text)
            for forced in ('满足即正常输出','这是**充分**的','命中 pattern 即已具备判读基础'):
                self.assertNotIn(forced,text)
            self.assertIn('必要条件，不是充分条件',text)
            self.assertEqual(kb,original)

    def test_audio_video_text_payloads_remain_byte_identical(self):
        import hashlib
        from datetime import datetime
        from pipeline import prompts
        expected={
            'cat:audio':'fdbb931c489b1f9c4355f69a16ebdd3d80e42a18ba022bd267f1111471297401',
            'cat:video':'3ba0eff5357fdd3d57c9dd323a42d9165f44579d7f5669bbf9f5f9ec87241e39',
            'cat:text':'edcfc5e2f127338592f65c8fd7ef6887bed9da07ef6526806eae470e21de9464',
            'dog:audio':'ab70d83e016d0524ccd5662fa5165e0108fe0aa8b0e041777b957ac15ee45fd5',
            'dog:video':'5fe2988dcd6827dcfc012bce910849dabe589ccec80a76ec5b2ff12f7341f590',
            'dog:text':'6ef201740595c3a0053e477862062644a859b281dd91a961609c906a133d36a2'}
        for key,digest in expected.items():
            species,modality=key.split(':')
            prompt=prompts.omni_prompt(species,'unavailable',prompts.load_knowledge(species),{},now=datetime(2026,11,7),modality=modality)
            actual=hashlib.sha256(json.dumps(prompt.messages(),ensure_ascii=False).encode()).hexdigest()
            self.assertEqual(actual,digest,key)
