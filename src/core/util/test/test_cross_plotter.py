import unittest
import torch
from pathlib import Path
import filecmp
import os

def get_memory_usage():
    import resource
    ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    vram = 0
    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated()
    return ram + vram

MEMORY_GROWTH_THRESHOLD = 1024 * 1024 * 10

class TestCrossPlotter(unittest.TestCase):
    def setUp(self):
        try:
            import matplotlib
        except:
            self.skipTest("matplotlib not installed")
        try:
            mem = get_memory_usage()
        except:
            self.skipTest("Can't check memory usage; OS probably not supported")

    
    def cross_plotter_param_test(self, num : int):
        path = Path(__file__).parent
        result_path = path.joinpath(f'cross_plotter_result{num}.png')
        expected_path = path.joinpath(f'cross_plotter_expected{num}.png')
        
        if result_path.exists():
            self.fail(f"Dangling {result_path}, delete manually to run test")
            return

        from ..plotting import CrossPlotter
        rng = torch.Generator().manual_seed(256)
        cross_plotter = CrossPlotter.basic_config(
            preds = [f'p{i}' for i in range(5)],
            trues = [f't{j}' for j in range(9)],
            axis = (0.0, 1.0, 0.0, 1.0)
        )

        memory_start = None
        batch_size = 30
        for i in range(num):
            if i == num - 1:
                batch_size = 10
            preds = torch.rand(batch_size, 5, generator=rng)
            trues = torch.hstack((preds, torch.rand(batch_size, 4, generator=rng)))
            cross_plotter.update(preds, trues, 'train')
            cross_plotter.update(preds*2, trues, 'val')
            memory = get_memory_usage()
            if memory_start is None:
                memory_start = memory
            elif memory - memory_start > MEMORY_GROWTH_THRESHOLD:
                self.fail(f"Memory leak detected: start_memory = {memory_start}, current = {memory}")
                return
            if i % 1_000 == 0:
                print(f"Memory usage at iteration {i}: {memory}")
        cross_plotter.save(result_path)
        if not expected_path.exists():
            self.fail(f'{expected_path} not found')
            return
        if filecmp.cmp(result_path, expected_path):
            os.remove(result_path)
            return
        else:
            self.fail(f'{result_path} does not match {expected_path}')
            return


    def test_small(self):
        self.cross_plotter_param_test(10)
    
    def test_medium(self):
        self.cross_plotter_param_test(100)

    @unittest.skipIf(int(os.getenv('TEST_LEVEL', 0)) < 1, 'Large test')
    def test_large(self):
        self.cross_plotter_param_test(10_000)




