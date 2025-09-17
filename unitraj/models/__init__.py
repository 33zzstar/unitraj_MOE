import os,sys 
parentdir = '/home/zzs/zzs/unitraj__MOE/'
sys.path.insert(0,parentdir) 
from unitraj.models.autobot.autobot import AutoBotEgo
from unitraj.models.mtr.MTR import MotionTransformer
from unitraj.models.wayformer.wayformer import Wayformer
from unitraj.models.smart.smart import SMART
from unitraj.models.moe.moe import MOE

__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
    'SMART': SMART,
    'MOE': MOE,
}


def build_model(config):
    model = __all__[config.method.model_name](
        config=config
    )

    return model
