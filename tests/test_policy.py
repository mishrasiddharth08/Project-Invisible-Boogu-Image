import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from pi_boogu_image.lib import assets, policy
from pi_boogu_image import lora

class PolicyTests(unittest.TestCase):
    def test_tasks_are_locked_to_correct_tab(self):
        for trunk in assets.TRUNKS:
            editing = 'edit' in trunk
            policy.validate_task(trunk, editing, int(editing))
            with self.assertRaises(ValueError):
                policy.validate_task(trunk, not editing, int(not editing))
        for trunk in ('edit', 'edit-turbo'):
            for count in (0, 2):
                with self.assertRaisesRegex(ValueError, 'exactly one'):
                    policy.validate_task(trunk, True, count)
        with self.assertRaises(ValueError):
            policy.validate_task('unknown', False)

    def test_all_trunks(self):
        cases = {'boogu_base_bf16':'base','boogu_turbo_hotfix':'turbo',
                 'boogu_edit_bf16':'edit','boogu_edit_turbo_hotfix_1k_20260708':'edit-turbo'}
        for filename, expected in cases.items():
            self.assertEqual(assets.trunk(filename), expected)

    def test_edit_turbo_keeps_official_dmd_with_manual_steps(self):
        options = policy.settings('edit-turbo', 2048, 2048, 2048, 32, True, 50, 9)
        self.assertEqual((options['num_inference_steps'], options['text_guidance_scale']), (50,1))
        self.assertEqual(options['width'],1024)
        args = policy.arguments('edit-turbo','p','n',options,object())
        self.assertTrue(args['use_dmd_student_inference'])
        self.assertEqual(args['dmd_conditioning_sigma'],0)

    def test_manual_steps_all_models_and_recommended_defaults(self):
        for trunk in assets.TRUNKS:
            self.assertEqual(policy.settings(trunk,1024,1024)['num_inference_steps'], 4 if 'turbo' in trunk else 28)
            for steps in (1, 8, 20, 60):
                self.assertEqual(policy.settings(trunk,1024,1024,steps=steps)['num_inference_steps'], steps)
            with self.assertRaises(ValueError):
                policy.settings(trunk,1024,1024,steps=0)

    def test_reference_modes_fail_closed(self):
        options=policy.settings('base',1024,1024)
        for trunk,reference in [('base',object()),('turbo',object()),('edit',None),('edit-turbo',None)]:
            with self.assertRaises(ValueError):
                policy.arguments(trunk,'p','n',options,reference)

    def test_distill_lock(self):
        for trunk,on in [('base',True),('edit',True),('turbo',False),('edit-turbo',False)]:
            with self.assertRaises(ValueError):
                policy.settings(trunk,1024,1024,distill=on)

    def test_resolution_profiles(self):
        for ram,cap in [(8,1024),(12,1024),(16,1536),(23,1536),(24,2048)]:
            self.assertEqual(policy.profile(ram,'base')['cap'],cap)
            self.assertEqual(policy.profile(ram,'base')['default'],1024)
        options=policy.settings('base',3000,2000,2048,32)
        self.assertEqual(options['width'],2048)
        self.assertEqual(options['height']%16,0)

    def test_truncated_safetensors_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'boogu_base.safetensors'
            head=json.dumps({'weight':{'dtype':'BF16','shape':[4],'data_offsets':[0,8]}}).encode()
            path.write_bytes(struct.pack('<Q',len(head))+head+b'00')
            with self.assertRaises(ValueError): assets.header(path)

    def test_lora_trunks_and_ambiguous_metadata(self):
        with patch.object(lora,'header',return_value={'__metadata__':{'ss_base_model':'Boogu'},
                'transformer.proj.lora_A.weight': {'shape': [2, 4]},
                'transformer.proj.lora_B.weight': {'shape': [4, 2]}}):
            self.assertEqual(lora.classification('landscape.safetensors'),('unknown',False))
            for trunk in assets.TRUNKS:
                self.assertEqual(lora.classification('boogu_'+trunk+'_style_lora.safetensors')[0],trunk)

    def test_foreign_lora(self):
        with patch.object(lora,'header',return_value={'__metadata__':{'ss_base_model':'flux'}}):
            self.assertEqual(lora.classification('flux_style.safetensors'),(None,False))

    def test_distill_rank_verified(self):
        with patch.object(lora,'header',return_value={'a.lora_A.weight':{'shape':[128,512]}}):
            self.assertEqual(lora.classification('boogu_image_turbo_hotfix_lora_rank_128_bf16.safetensors'),('turbo',True))
        with patch.object(lora,'header',return_value={'a.lora_A.weight':{'shape':[64,512]}}):
            with self.assertRaises(ValueError):
                lora.classification('boogu_image_turbo_lora_rank_128.safetensors')

if __name__=='__main__': unittest.main()
