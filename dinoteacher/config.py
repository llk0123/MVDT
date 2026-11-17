# """
# Additional configs for DINO Teacher, based on config.py in Adaptive Teacher.
# """
#
# from detectron2.config import CfgNode as CN
#
#
# def add_dinoteacher_config(cfg):
#     """
#     Add config for semisupnet.
#     """
#     _C = cfg
#     _C.TEST.VAL_LOSS = True
#
#     _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
#     _C.MODEL.RPN.LOSS = "CrossEntropy"
#     _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"
#
#     _C.SOLVER.IMG_PER_BATCH_LABEL = 1
#     _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
#     _C.SOLVER.FACTOR_LIST = (1,)
#
#     # Override detectron2 defaults; do NOT replace the whole CLIP_GRADIENTS node
#     # (detectron2 build_optimizer requires CLIP_TYPE and NORM_TYPE).
#     _C.SOLVER.CLIP_GRADIENTS.ENABLED = True
#     _C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
#     _C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 5.0
#     _C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
#
#     _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
#     _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
#     _C.DATASETS.CROSS_DATASET = True
#     _C.TEST.EVALUATOR = "COCOeval"
#
#     _C.SEMISUPNET = CN()
#
#     # Output dimension of the MLP projector after `res5` block
#     _C.SEMISUPNET.MLP_DIM = 128
#
#     # Semi-supervised training
#     _C.SEMISUPNET.Trainer = "ateacher"
#     _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
#     _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
#     _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
#     _C.SEMISUPNET.BURN_UP_STEP = 12000
#     _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
#     _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
#     _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
#     _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
#     _C.SEMISUPNET.DIS_TYPE = "res4"
#     _C.SEMISUPNET.DIS_LOSS_WEIGHT = 0.1
#
#     # dataloader
#     # supervision level
#     _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
#     _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
#     _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"
#
#     _C.EMAMODEL = CN()
#     _C.EMAMODEL.SUP_CONSIST = True
#
#
#     # DINO Teacher alignment
#     _C.SEMISUPNET.USE_FEATURE_ALIGN = False
#     _C.SEMISUPNET.FEATURE_ALIGN_LAYER = 'res4'
#     _C.SEMISUPNET.ALIGN_MODEL = "dinov2_vitb14"
#     _C.SEMISUPNET.DINO_PATCH_SIZE = 14
#     _C.SEMISUPNET.ALIGN_HEAD_TYPE = "attention"  # attention, MLP, MLP3, linear
#     _C.SEMISUPNET.ALIGN_HEAD_PROJ_DIM = 1024
#     _C.SEMISUPNET.ALIGN_PROJ_GELU = False
#     _C.SEMISUPNET.ALIGN_HEAD_NORMALIZE = True
#     _C.SEMISUPNET.ALIGN_EASY_ONLY = True
#     _C.SEMISUPNET.FEATURE_ALIGN_TARGET_START = 5000
#     _C.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT = 1.0
#     _C.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT_TARGET = 1.0
#
#     # DINO Teacher Labels
#     _C.SEMISUPNET.LABELER_TARGET_PSEUDOGT = None
#     _C.SEMISUPNET.LABELER_PSEUDOGT_SWAP = None
#     _C.SEMISUPNET.LABELER_PSEUDOGT_SWAP_ITER = 100000
#
#     # EDL-aware pseudo-label filtering (lightweight: score-derived Beta uncertainty)
#     _C.SEMISUPNET.EDL_MODE = "lightweight"  # lightweight | full
#     _C.SEMISUPNET.USE_EDL_PSEUDO_FILTER = False
#     _C.SEMISUPNET.EDL_UNCERTAINTY_THRESHOLD = 0.02
#     _C.SEMISUPNET.EDL_EVIDENCE_STRENGTH = 10.0
#     _C.SEMISUPNET.EDL_REGU_WEIGHT = 1.0
#     _C.SEMISUPNET.EDL_LAMBDA_MIN = 1.0
#     _C.SEMISUPNET.EDL_LAMBDA_MAX = 100.0
#     _C.SEMISUPNET.EDL_LAMBDA_INIT = 10.0
#     _C.SEMISUPNET.EDL_FILTER_AT_GEN = False
#
#     # DINO ViT Backbone
#     _C.SEMISUPNET.DINO_BBONE_MODEL = "dinov2_vitl14"
#     _C.SEMISUPNET.DINO_BBONE_LR_SCALE = 0.0     # 0.0 = frozen backbone
#

"""
Additional configs for DINO Teacher, based on config.py in Adaptive Teacher.
"""

from detectron2.config import CfgNode as CN


def add_dinoteacher_config(cfg):
    """
    Add config for semisupnet.
    """
    _C = cfg
    _C.TEST.VAL_LOSS = True

    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

    _C.SOLVER.IMG_PER_BATCH_LABEL = 1
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 1
    _C.SOLVER.FACTOR_LIST = (1,)

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.CROSS_DATASET = True
    _C.TEST.EVALUATOR = "COCOeval"

    _C.SEMISUPNET = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.SEMISUPNET.MLP_DIM = 128

    # Semi-supervised training
    _C.SEMISUPNET.Trainer = "ateacher"
    _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
    _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
    _C.SEMISUPNET.BURN_UP_STEP = 12000
    _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
    _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
    _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
    _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
    _C.SEMISUPNET.DIS_TYPE = "res4"
    _C.SEMISUPNET.DIS_LOSS_WEIGHT = 0.1

    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True


    # DINO Teacher alignment
    _C.SEMISUPNET.USE_FEATURE_ALIGN = False
    _C.SEMISUPNET.FEATURE_ALIGN_LAYER = 'res4'
    _C.SEMISUPNET.ALIGN_MODEL = "dinov2_vitb14"
    _C.SEMISUPNET.DINO_PATCH_SIZE = 14
    _C.SEMISUPNET.ALIGN_HEAD_TYPE = "attention"  # attention, MLP, MLP3, linear
    _C.SEMISUPNET.ALIGN_HEAD_PROJ_DIM = 1024
    _C.SEMISUPNET.ALIGN_PROJ_GELU = False
    _C.SEMISUPNET.ALIGN_HEAD_NORMALIZE = True
    _C.SEMISUPNET.ALIGN_EASY_ONLY = True
    _C.SEMISUPNET.FEATURE_ALIGN_TARGET_START = 5000
    _C.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT = 1.0
    _C.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT_TARGET = 1.0

    # DINO Teacher Labels
    _C.SEMISUPNET.LABELER_TARGET_PSEUDOGT = None
    _C.SEMISUPNET.LABELER_PSEUDOGT_SWAP = None
    _C.SEMISUPNET.LABELER_PSEUDOGT_SWAP_ITER = 100000

    # DINO ViT Backbone
    _C.SEMISUPNET.DINO_BBONE_MODEL = "dinov2_vitl14"
    _C.SEMISUPNET.DINO_BBONE_LR_SCALE = 0.0     # 0.0 = frozen backbone