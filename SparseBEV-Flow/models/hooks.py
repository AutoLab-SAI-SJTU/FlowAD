from mmcv.runner.hooks import HOOKS, Hook
from torch import nn

# __all__ = ['SequentialControlHook']

def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)

@HOOKS.register_module()
class ClipBrakeHook(Hook):
    """ """

    def __init__(self, clip_brake_epoch=23):
        super().__init__()
        self.clip_brake_epoch=clip_brake_epoch

    def set_brake_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.pts_bbox_head.module.clip_brake=flag
        else:
            runner.model.module.pts_bbox_head.clip_brake = flag

    def before_run(self, runner):
        self.set_brake_flag(runner, False)
        if runner.epoch > self.clip_brake_epoch:
            self.set_brake_flag(runner, True)

    def before_train_epoch(self, runner):
        if runner.epoch > self.clip_brake_epoch:
            self.set_brake_flag(runner, True)