"""Reference defaults. Override filesystem roots through the CLI, not numerical settings."""
from pathlib import Path

DATA = Path("datasets")
RET_COMMIT = "2dd9ae05f34a34e2df3b955f743215ac3cda0f82"
HVI_COMMIT = "eb43d7d91e9a336c66856824ff9e4603ae41f408"
FIT_TRAIN_COUNT=None
CV_FOLDS=3
LAMBDA_GRID=[0.0,0.05,0.2,0.5,1.0,2.0,5.0]
REGULARIZATION_CENTER=0.5
SPLIT_SEED=20261123
CFG={
 'LOLv1':dict(root=DATA/'LOLv1',train_low='Train/input',train_gt='Train/target',test_low='Test/input',test_gt='Test/target',n_test=15,ret='LOL_v1.pth',hvi='LOLv1/w_perc.pth',gate='gated',hvi_alpha=None,ref_ret=25.16,ref_hvi=23.8091),
 'LOLv2_Real':dict(root=DATA/'LOLv2',train_low='Real_captured/Train/Low',train_gt='Real_captured/Train/Normal',test_low='Real_captured/Test/Low',test_gt='Real_captured/Test/Normal',n_test=100,ret='LOL_v2_real.pth',hvi='LOLv2_real/best_PSNR.pth',gate='gated2',hvi_alpha=.8,ref_ret=22.80,ref_hvi=23.9040),
 'LOLv2_Synthetic':dict(root=DATA/'LOLv2',train_low='Synthetic/Train/Low',train_gt='Synthetic/Train/Normal',test_low='Synthetic/Test/Low',test_gt='Synthetic/Test/Normal',n_test=100,ret='LOL_v2_synthetic.pth',hvi='LOLv2_syn/wo_perc.pth',gate='none',hvi_alpha=None,ref_ret=25.67,ref_hvi=25.7048),
}
RET_EXPECTED_HASH={
 'LOL_v1.pth':'9b4810dc51322fab70be8658ee0ad1df64a54aa814497b7849767baa2ff32cd1',
 'LOL_v2_real.pth':'539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b',
 'LOL_v2_synthetic.pth':'2f793853c6b1ba0b4a3ff12b7b10f0f2c851eaefda1006437eb3f35229838c4a',
}

HVI_REFERENCE_HASH = {'LOLv1': '9206385b514014b60bf9396151a011cabf68abb380c2ad87ffde3d53e8227926', 'LOLv2_Real': '44a7f92cf02a1da322c1a34b3aa940341dc93fd53840b7f19b49f461200b00a6', 'LOLv2_Synthetic': '519cc5fc55491cf36b47b8295c0204dde29ba887feeed06a77e716072469e55e'}
METHODS = ["ret", "hvi", "rgb_fit", "ycc_fit", "ycc_regularized", "ycc_joint", "ycc_joint_regularized"]
