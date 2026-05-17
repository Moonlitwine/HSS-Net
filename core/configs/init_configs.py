from yacs.config import CfgNode as CN

_C = CN()

_C.config = "realsetting.yaml"
_C.exp = "exp"
_C.doc = "doc"
_C.resume = ""
_C.verbose = "info"
_C.seed = 1234
_C.val_folder = "images"
_C.sample_folder = "exp/sample_result"

_C.data = CN()
_C.data.dataset = "RealSetting"
_C.data.dataset_dir = "../datasets/RealSetting"
_C.data.channels = 3
_C.data.num_workers = 8
_C.data.train_filelist = ""
_C.data.val_filelist = ""
_C.data.train_num = 480000

_C.model = CN()
_C.model.ema_rate = 0.999
_C.model.ema = True

_C.data_augment = CN()
_C.data_augment.augment = False
_C.data_augment.flip_prob = 0.0
_C.data_augment.scale_prob = 0.0
_C.data_augment.mixup_prob = 0.0
_C.data_augment.around_padding = False

_C.backbone = CN()
_C.backbone.image_size = 128
_C.backbone.channels = 32
_C.backbone.channels_mult = [1, 2, 4, 8]
_C.backbone.dropout = 0.0

_C.training = CN()
_C.training.batch_size = 6
_C.training.step_size = 240000
_C.training.snapshot_freq = 5000
_C.training.validation_freq = 10000

_C.sampling = CN()
_C.sampling.batch_size = 1
_C.sampling.patch_num = 10
_C.sampling.patch_size = 256
_C.sampling.patch_overlap = 64

_C.optim = CN()
_C.optim.weight_decay = 0.000
_C.optim.optimizer = "Adam"
_C.optim.lr = 0.0004
_C.optim.amsgrad = False
_C.optim.eps = 0.00000001
_C.optim.beta1 = 0.9

_C.scheduler = CN()
_C.scheduler.periods = [80000, 80000, 80000]
_C.scheduler.restart_weights = [1, 0.5, 0.25]
_C.scheduler.eta_mins = [0.0004, 0.0002, 0.0001]

_C.loss = CN()
_C.loss.depth_weight = 0.0
_C.loss.perceptual_weight = 0.1
_C.loss.ssim_weight = 0.0
_C.loss.pixel_weight = 1.0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""

    return _C.clone()

