from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from pi_boogu_image import forge
from pi_boogu_image.lib.assets import Asset

class HookTests(unittest.TestCase):
    def test_grid_filter_is_opt_in_and_ignores_stale_model_controls(self):
        asset = Asset(Path('boogu.safetensors'), 'turbo', 'bf16')
        runner = SimpleNamespace(alwayson_scripts=[SimpleNamespace(is_boogu_invisible=True, args_from=0, args_to=4)])
        self.assertTrue(forge.controls(None, runner, ('turbo', 1024, True, True), asset)['reduce_grid'])
        self.assertFalse(forge.controls(None, runner, ('turbo', 1024, True, False), asset)['reduce_grid'])
        self.assertNotIn('reduce_grid', forge.controls(None, runner, ('base', 1024, False, True), asset))

    def test_old_distill_control_cannot_override_checkpoint_mode(self):
        runner = SimpleNamespace(alwayson_scripts=[SimpleNamespace(is_boogu_invisible=True, args_from=0, args_to=3)])
        for trunk in ('base', 'turbo', 'edit', 'edit-turbo'):
            asset = Asset(Path('boogu.safetensors'), trunk, 'bf16')
            result = forge.controls(None, runner, (trunk, 768, 'turbo' not in trunk), asset)
            self.assertEqual(result, {'ceiling': 768, 'distill': 'turbo' in trunk})

    def setUp(self):
        self.calls=[]
        calls=self.calls
        class Runner:
            def run(self,p,*args,**kwargs):
                calls.append(('run',p,args,kwargs))
                return 'stock-script-result'
        self.Runner=Runner
        self.original_run=Runner.run
        def process(p,*args,**kwargs):
            calls.append(('process',p,args,kwargs))
            return 'stock-process-result'
        self.processing=SimpleNamespace(process_images=process)
        self.option=SimpleNamespace(onchange=lambda:calls.append('existing-onchange'))
        self.models=SimpleNamespace(list_models=lambda:None,checkpoints_list={},checkpoint_aliases={})
        self.modules=ModuleType('modules')
        self.modules.scripts=SimpleNamespace(ScriptRunner=Runner)
        self.modules.processing=self.processing
        self.modules.sd_models=self.models
        self.modules.shared=SimpleNamespace(opts=SimpleNamespace(data_labels={'sd_model_checkpoint':self.option}))
        self.modules.script_callbacks=SimpleNamespace(on_script_unloaded=lambda f:None,on_before_ui=lambda f:None)
        self.patches=[patch.dict('sys.modules',{'modules':self.modules}),
                      patch.object(forge,'register_checkpoints'),patch.object(forge.pipeline,'release')]
        for p in self.patches:p.start()
        forge._installed=False
        forge._undo.clear()

    def tearDown(self):
        forge.uninstall()
        for p in reversed(self.patches):p.stop()

    def test_other_checkpoint_forwarding_exact(self):
        with patch.object(forge,'selected',return_value=None):
            self.assertTrue(forge.install())
            p=object()
            self.assertEqual(self.Runner().run(p,0,'value',future=True),'stock-script-result')
            self.assertEqual(self.calls[-1],('run',p,(0,'value'),{'future':True}))
            self.assertEqual(self.processing.process_images(p,7,future=True),'stock-process-result')

    def test_boogu_takeover_does_not_call_stock(self):
        asset=Asset(Path('boogu_edit_turbo.safetensors'),'edit-turbo','bf16')
        with patch.object(forge,'selected',return_value=asset),patch.object(forge,'generate',return_value='boogu'):
            forge.install()
            self.assertEqual(self.Runner().run(object(),0),'boogu')
            self.assertEqual(self.processing.process_images(object()),'boogu')
            self.assertEqual(self.calls,[])

    def test_onchange_chain_preserved_and_uninstalled(self):
        previous=self.option.onchange
        forge.install()
        with patch.object(forge, 'selected', return_value=None):
            self.option.onchange()
        self.assertEqual(self.calls,['existing-onchange'])
        forge.uninstall()
        self.assertIs(self.option.onchange,previous)
        self.assertIs(self.Runner.run,self.original_run)

    def test_drift_leaves_stock_intact(self):
        self.Runner.run=lambda self:None
        changed=self.Runner.run
        self.assertFalse(forge.install())
        self.assertIs(self.Runner.run,changed)

    def test_other_engine_wrapper_preserved_during_uninstall(self):
        forge.install()
        inner=self.Runner.run
        def other(self,p,*args,**kwargs): return inner(self,p,*args,**kwargs)
        self.Runner.run=other
        forge.uninstall()
        self.assertIs(self.Runner.run,other)

if __name__=='__main__':unittest.main()
