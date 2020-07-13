import os


HPARAMS_REGISTRY = {}
DEFAULT_OUT_DIR = os.path.expandvars('$HOME/dist-aug')


class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


good_baseline_sm = Hyperparams()
good_baseline_sm.float16 = True
good_baseline_sm.fp16_mean_var = True
good_baseline_sm.fp16_allreduce = True
good_baseline_sm.no_vocab_rounding = False
good_baseline_sm.skip_initial_evals = True
good_baseline_sm.n_ctx = 2048
good_baseline_sm.n_layer = 32
good_baseline_sm.n_head = 4
good_baseline_sm.n_batch = 16
good_baseline_sm.n_embd = 256
good_baseline_sm.activation = 'quick_gelu'
good_baseline_sm.optimizer = 'bs_adam'
good_baseline_sm.blocksparse_op = True
good_baseline_sm.recompute = True
good_baseline_sm.resid_pdrop = 0.05
good_baseline_sm.warmup_iters = 7500
good_baseline_sm.embd_pdrop = 0.05
good_baseline_sm.lr = 0.0007
good_baseline_sm.total_epochs = 120
good_baseline_sm.pos_embd_std = 0.007
good_baseline_sm.w_embd_std = 0.013
good_baseline_sm.fp16_loss_scale = 2.0**16
good_baseline_sm.merge_layer_allreduce = 1
good_baseline_sm.max_grad_norm = 1.0
good_baseline_sm.blocksize = 64
good_baseline_sm.attention_layers = 'a'
good_baseline_sm.mlp_w1 = 0.125
good_baseline_sm.qk_w = 0.125
good_baseline_sm.v_w = 0.125
good_baseline_sm.post_w = 0.125
good_baseline_sm.mlp_w2 = 0.5
good_baseline_sm.mlp_multiple = 4.0
good_baseline_sm.qk_ratio = 1.0
HPARAMS_REGISTRY['good_base_sm'] = good_baseline_sm

good_baseline_med = Hyperparams()
good_baseline_med.n_layer = 64
good_baseline_med.lr = 0.0005
good_baseline_med.n_batch = 4
HPARAMS_REGISTRY['good_base_med'] = good_baseline_med

good_baseline_large = Hyperparams()
good_baseline_large.n_layer = 64
good_baseline_large.n_head = 16
good_baseline_large.n_embd = 512
good_baseline_large.n_batch = 1
HPARAMS_REGISTRY['good_base_lg'] = good_baseline_large

sample_during_eval_8gpu = Hyperparams()
sample_during_eval_8gpu.sample_during_eval = True
sample_during_eval_8gpu.samples_to_generate = 1
sample_during_eval_8gpu.sample_batch = 1
sample_during_eval_8gpu.sample_grid_dim = 4
HPARAMS_REGISTRY['sample-during-eval-8gpu'] = sample_during_eval_8gpu

c10 = Hyperparams()
c10.n_ctx = 3072
c10.dataset = 'cifar10'
c10.mlp_multiple = 2.0
c10.qk_ratio = 0.5
c10.n_embd = 256
HPARAMS_REGISTRY['cifar10'] = c10

c10_dense = Hyperparams()
c10_dense.update(good_baseline_sm)
c10_dense.update(sample_during_eval_8gpu)
c10_dense.update(c10)
c10_dense.lr = 0.00035
c10_dense.dynamic_loss_scaling = True
c10_dense.warmup_iters = 15000
c10_dense.max_grad_norm = 1.0
c10_dense.resid_pdrop = 0.25
c10_dense.embd_pdrop = 0.0
c10_dense.n_batch = 2
c10_dense.n_layer = 128
c10_dense.merge_layer_allreduce = 4
c10_dense.n_head = 2
c10_dense.total_epochs = 140
c10_dense.qk_w = 0.125
c10_dense.mlp_w1 = 0.125
c10_dense.mlp_w2 = 0.125
c10_dense.post_w = 0.125
c10_dense.logits_w = 0.0
c10_dense.pos_embd_std = 0.01
c10_dense.w_embd_std = 0.01
c10_dense.blocksize = 32
c10_dense.l2_loss = 0.01
HPARAMS_REGISTRY['c10-dense'] = c10_dense

c10_sparse = Hyperparams()
c10_sparse.update(c10_dense)
c10_sparse.blocksize = 32
c10_sparse.local_attn_ctx = 96
c10_sparse.attention_layers = 'bT,b,b,b'
c10_sparse.test_size = 2000
c10_sparse.datapoints = 48000
HPARAMS_REGISTRY['c10-gemnet'] = c10_sparse

c10_58m = Hyperparams()
c10_58m.update(c10_sparse)
HPARAMS_REGISTRY['c10-58m'] = c10_58m

c10_58m_rot = Hyperparams()
c10_58m_rot.update(c10_58m)
c10_58m_rot.use_rotation = True
c10_58m_rot.total_epochs = 10000
c10_58m_rot.resid_pdrop = 0.01
HPARAMS_REGISTRY['c10-58m-rot'] = c10_58m_rot

c10_58m_rot_tr = Hyperparams()
c10_58m_rot_tr.update(c10_58m)
c10_58m_rot_tr.use_rotation = True
c10_58m_rot_tr.use_transposition = True
c10_58m_rot_tr.total_epochs = 10000
c10_58m_rot_tr.resid_pdrop = 0.01
HPARAMS_REGISTRY['c10-58m-rot-tr'] = c10_58m_rot_tr

c10_15m_dense = Hyperparams()
c10_15m_dense.update(c10_dense)
c10_15m_dense.n_layer = 32
c10_15m_dense.n_batch = 16
c10_15m_dense.resid_pdrop = 0.005
c10_15m_dense.total_epochs = 10000
c10_15m_dense.test_size = 2000
c10_15m_dense.datapoints = 48000
HPARAMS_REGISTRY['c10_15m_dense'] = c10_15m_dense

c10_15m = Hyperparams()
c10_15m.update(c10_sparse)
c10_15m.n_layer = 32
c10_15m.n_batch = 16
c10_15m.resid_pdrop = 0.005
c10_15m.total_epochs = 10000
HPARAMS_REGISTRY['c10_15m'] = c10_15m

c10_15m_rot = Hyperparams()
c10_15m_rot.update(c10_15m)
c10_15m_rot.use_rotation = True
HPARAMS_REGISTRY['c10_15m_rot'] = c10_15m_rot

c10_15m_rot_tr = Hyperparams()
c10_15m_rot_tr.update(c10_15m)
c10_15m_rot_tr.use_rotation = True
c10_15m_rot_tr.use_transposition = True
HPARAMS_REGISTRY['c10_15m_rot_tr'] = c10_15m_rot_tr

c10_15m_tr = Hyperparams()
c10_15m_tr.update(c10_15m)
c10_15m_tr.use_transposition = True
HPARAMS_REGISTRY['c10_15m_tr'] = c10_15m_tr

c10_15m_rev = Hyperparams()
c10_15m_rev.update(c10_15m)
c10_15m_rev.use_reverse = True
HPARAMS_REGISTRY['c10_15m_rev'] = c10_15m_rev

c10_15m_c = Hyperparams()
c10_15m_c.update(c10_15m)
c10_15m_c.use_color = True
HPARAMS_REGISTRY['c10_15m_c'] = c10_15m_c

c10_15m_js = Hyperparams()
c10_15m_js.update(c10_15m)
c10_15m_js.use_jigsaw = True
c10_15m_js.jigsaw_grid_size = 2
HPARAMS_REGISTRY['c10_15m_js'] = c10_15m_js

c10_15m_lr = Hyperparams()
c10_15m_lr.update(c10_15m)
c10_15m_lr.aug = True
HPARAMS_REGISTRY['c10_15m_lr'] = c10_15m_lr

c10_15m_ra_n2_m3 = Hyperparams()
c10_15m_ra_n2_m3.update(c10_15m)
c10_15m_ra_n2_m3.rand_augment = True
c10_15m_ra_n2_m3.rand_augment_conditioning = True
c10_15m_ra_n2_m3.rand_augment_n = 2
c10_15m_ra_n2_m3.rand_augment_m = 3
HPARAMS_REGISTRY['c10_15m_ra_n2_m3'] = c10_15m_ra_n2_m3

c10_15m_ra_n1_m2 = Hyperparams()
c10_15m_ra_n1_m2.update(c10_15m)
c10_15m_ra_n1_m2.rand_augment = True
c10_15m_ra_n1_m2.rand_augment_conditioning = True
c10_15m_ra_n1_m2.rand_augment_n = 1
c10_15m_ra_n1_m2.rand_augment_m = 2
HPARAMS_REGISTRY['c10_15m_ra_n1_m2'] = c10_15m_ra_n1_m2

c10_15m_i32_nocond = Hyperparams()
c10_15m_i32_nocond.update(c10_15m)
c10_15m_i32_nocond.dataset = 'imagenet32cifar'
c10_15m_i32_nocond.use_imagenet_fraction = 1.0
c10_15m_i32_nocond.eval_after_n_examples = 48000
c10_15m_i32_nocond.use_dataset_conditioning = True
c10_15m_i32_nocond.use_unconditional_augmentation = True
HPARAMS_REGISTRY['c10_15m_i32_nocond'] = c10_15m_i32_nocond

c10_15m_i32_cond = Hyperparams()
c10_15m_i32_cond.update(c10_15m)
c10_15m_i32_cond.dataset = 'imagenet32cifar'
c10_15m_i32_cond.use_imagenet_fraction = 1.0
c10_15m_i32_cond.eval_after_n_examples = 48000
c10_15m_i32_cond.use_dataset_conditioning = True
HPARAMS_REGISTRY['c10_15m_i32_cond'] = c10_15m_i32_cond

c10_15m_ss_i32_nocond = Hyperparams()
c10_15m_ss_i32_nocond.update(c10_15m)
c10_15m_ss_i32_nocond.auxiliary_dataset = 'imagenet32'
c10_15m_ss_i32_nocond.auxiliary_dataset_fraction = 0.5
c10_15m_ss_i32_nocond.use_dataset_conditioning = True
c10_15m_ss_i32_nocond.use_unconditional_augmentation = True
HPARAMS_REGISTRY['c10_15m_ss_i32_nocond'] = c10_15m_ss_i32_nocond

c10_15m_ss_i32_cond = Hyperparams()
c10_15m_ss_i32_cond.update(c10_15m)
c10_15m_ss_i32_cond.auxiliary_dataset = 'imagenet32'
c10_15m_ss_i32_cond.auxiliary_dataset_fraction = 0.5
c10_15m_ss_i32_cond.use_dataset_conditioning = True
HPARAMS_REGISTRY['c10_15m_ss_i32_cond'] = c10_15m_ss_i32_cond

c10_15m_dense_rd = Hyperparams()
c10_15m_dense_rd.update(c10_15m_dense)
c10_15m_dense_rd.use_randomly_determined_order = True
c10_15m_dense_rd.randomly_determined_order_num_perms = 3
c10_15m_dense_rd.randomly_determined_order_seed = 42
HPARAMS_REGISTRY['c10_15m_dense_rd'] = c10_15m_dense_rd

c10_15m_rd = Hyperparams()
c10_15m_rd.update(c10_15m)
c10_15m_rd.use_randomly_determined_order = True
c10_15m_rd.randomly_determined_order_num_perms = 3
c10_15m_rd.randomly_determined_order_seed = 42
HPARAMS_REGISTRY['c10_15m_rd'] = c10_15m_rd

c10_15m_rd_s314 = Hyperparams()
c10_15m_rd_s314.update(c10_15m)
c10_15m_rd_s314.use_randomly_determined_order = True
c10_15m_rd_s314.randomly_determined_order_num_perms = 3
c10_15m_rd_s314.randomly_determined_order_seed = 314
HPARAMS_REGISTRY['c10_15m_rd_s314'] = c10_15m_rd_s314

c10_15m_rd_s2718 = Hyperparams()
c10_15m_rd_s2718.update(c10_15m)
c10_15m_rd_s2718.use_randomly_determined_order = True
c10_15m_rd_s2718.randomly_determined_order_num_perms = 3
c10_15m_rd_s2718.randomly_determined_order_seed = 2718
HPARAMS_REGISTRY['c10_15m_rd_s2718'] = c10_15m_rd_s2718

c10_15m_rd_s1618 = Hyperparams()
c10_15m_rd_s1618.update(c10_15m)
c10_15m_rd_s1618.use_randomly_determined_order = True
c10_15m_rd_s1618.randomly_determined_order_num_perms = 3
c10_15m_rd_s1618.randomly_determined_order_seed = 1618
HPARAMS_REGISTRY['c10_15m_rd_s1618'] = c10_15m_rd_s1618

imagenet64_8gpu = Hyperparams()
imagenet64_8gpu.update(c10_sparse)
imagenet64_8gpu.n_batch = 16
imagenet64_8gpu.n_embd = 512
imagenet64_8gpu.n_layer = 28
imagenet64_8gpu.n_head = 4
imagenet64_8gpu.dataset = 'imagenet64'
imagenet64_8gpu.blocksize = 64
imagenet64_8gpu.local_attn_ctx = 128
imagenet64_8gpu.lr = 0.00025
imagenet64_8gpu.n_ctx = 8192
imagenet64_8gpu.resid_pdrop = 0.01
imagenet64_8gpu.embd_pdrop = 0.01
imagenet64_8gpu.total_epochs = 50
imagenet64_8gpu.mlp_w1 = 0.125
imagenet64_8gpu.qk_w = 0.125
imagenet64_8gpu.v_w = 0.125
imagenet64_8gpu.post_w = 0.125
imagenet64_8gpu.mlp_w2 = 0.5
imagenet64_8gpu.mlp_multiple = 4.0
imagenet64_8gpu.qk_ratio = 1.0
HPARAMS_REGISTRY['imagenet64-8gpu'] = imagenet64_8gpu

c10_150m_baseline = Hyperparams()
c10_150m_baseline.update(imagenet64_8gpu)
c10_150m_baseline.blocksize = 32
c10_150m_baseline.local_attn_ctx = 96
c10_150m_baseline.n_batch = 2
c10_150m_baseline.lr = 0.00015
c10_150m_baseline.merge_layer_allreduce = 4
c10_150m_baseline.n_layer = 48
c10_150m_baseline.resid_pdrop = 0.005
c10_150m_baseline.pos_embd_std = 0.01
c10_150m_baseline.w_embd_std = 0.01
c10_150m_baseline.dynamic_loss_scaling = True
c10_150m_baseline.embd_pdrop = 0.0
c10_150m_baseline.mlp_w2 = 0.125
c10_150m_baseline.n_ctx = 3072
c10_150m_baseline.n_head = 16
c10_150m_baseline.attention_layers = 'b,bT,b,b'
c10_150m_baseline.dataset = 'cifar10'
c10_150m_baseline.total_epochs = 10000
c10_150m_baseline.test_size = 2000
c10_150m_baseline.datapoints = 48000
HPARAMS_REGISTRY['c10_150m_baseline'] = c10_150m_baseline

c10_150m_pgd1 = Hyperparams()
c10_150m_pgd1.update(c10_150m_baseline)
c10_150m_pgd1.use_linf_pgd = True
c10_150m_pgd1.linf_pgd_epsilon = 1.0
c10_150m_pgd1.linf_pgd_n = 1
c10_150m_pgd1.linf_pgd_a = 1.0
HPARAMS_REGISTRY['c10_150m_pgd1'] = c10_150m_pgd1

c10_150m_pgd3 = Hyperparams()
c10_150m_pgd3.update(c10_150m_baseline)
c10_150m_pgd3.use_linf_pgd = True
c10_150m_pgd3.linf_pgd_epsilon = 2.0
c10_150m_pgd3.linf_pgd_n = 3
c10_150m_pgd3.linf_pgd_a = 1.0
HPARAMS_REGISTRY['c10_150m_pgd3'] = c10_150m_pgd3

c10_150m_pgd4 = Hyperparams()
c10_150m_pgd4.update(c10_150m_baseline)
c10_150m_pgd4.use_linf_pgd = True
c10_150m_pgd4.linf_pgd_epsilon = 3.0
c10_150m_pgd4.linf_pgd_n = 4
c10_150m_pgd4.linf_pgd_a = 1.0
HPARAMS_REGISTRY['c10_150m_pgd4'] = c10_150m_pgd4

c10_150m_pgd5 = Hyperparams()
c10_150m_pgd5.update(c10_150m_baseline)
c10_150m_pgd5.use_linf_pgd = True
c10_150m_pgd5.linf_pgd_epsilon = 4.0
c10_150m_pgd5.linf_pgd_n = 5
c10_150m_pgd5.linf_pgd_a = 1.0
HPARAMS_REGISTRY['c10_150m_pgd5'] = c10_150m_pgd5

c10_150m_rot = Hyperparams()
c10_150m_rot.update(c10_150m_baseline)
c10_150m_rot.use_rotation = True
HPARAMS_REGISTRY['c10_150m_rot'] = c10_150m_rot

c10_150m_tr = Hyperparams()
c10_150m_tr.update(c10_150m_baseline)
c10_150m_tr.use_transposition = True
HPARAMS_REGISTRY['c10_150m_tr'] = c10_150m_tr

c10_150m_js = Hyperparams()
c10_150m_js.update(c10_150m_baseline)
c10_150m_js.use_jigsaw = True
c10_150m_js.jigsaw_grid_size = 2
HPARAMS_REGISTRY['c10_150m_js'] = c10_150m_js

c10_150m_color = Hyperparams()
c10_150m_color.update(c10_150m_baseline)
c10_150m_color.use_color = True
HPARAMS_REGISTRY['c10_150m_color'] = c10_150m_color

c10_150m_tr = Hyperparams()
c10_150m_tr.update(c10_150m_baseline)
c10_150m_tr.use_transposition = True
HPARAMS_REGISTRY['c10_150m_tr'] = c10_150m_tr

c10_150m_rot_tr = Hyperparams()
c10_150m_rot_tr.update(c10_150m_baseline)
c10_150m_rot_tr.use_rotation = True
c10_150m_rot_tr.use_transposition = True
HPARAMS_REGISTRY['c10_150m_rot_tr'] = c10_150m_rot_tr

c10_150m_rot_js = Hyperparams()
c10_150m_rot_js.update(c10_150m_baseline)
c10_150m_rot_js.use_rotation = True
c10_150m_rot_js.use_jigsaw = True
c10_150m_rot_js.jigsaw_grid_size = 2
HPARAMS_REGISTRY['c10_150m_rot_js'] = c10_150m_rot_js

c10_150m_rot_js_tr = Hyperparams()
c10_150m_rot_js_tr.update(c10_150m_baseline)
c10_150m_rot_js_tr.use_rotation = True
c10_150m_rot_js_tr.use_jigsaw = True
c10_150m_rot_js_tr.jigsaw_grid_size = 2
c10_150m_rot_js_tr.use_transposition = True
HPARAMS_REGISTRY['c10_150m_rot_js_tr'] = c10_150m_rot_js_tr

c10_150m_rot_js_tr_c = Hyperparams()
c10_150m_rot_js_tr_c.update(c10_150m_baseline)
c10_150m_rot_js_tr_c.use_rotation = True
c10_150m_rot_js_tr_c.use_jigsaw = True
c10_150m_rot_js_tr_c.jigsaw_grid_size = 2
c10_150m_rot_js_tr_c.use_transposition = True
c10_150m_rot_js_tr_c.use_color = True
HPARAMS_REGISTRY['c10_150m_rot_js_tr_c'] = c10_150m_rot_js_tr_c

c10_150m_imagenet = Hyperparams()
c10_150m_imagenet.update(c10_150m_baseline)
c10_150m_imagenet.dataset = 'imagenet32cifar'
c10_150m_imagenet.use_imagenet_fraction = 1.0
c10_150m_imagenet.eval_after_n_examples = 48000
c10_150m_imagenet.use_dataset_conditioning = True
HPARAMS_REGISTRY['c10_150m_imagenet'] = c10_150m_imagenet

c10_150m_aug = Hyperparams()
c10_150m_aug.update(c10_150m_baseline)
c10_150m_aug.aug = True
c10_150m_aug.resid_pdrop = 0.40
HPARAMS_REGISTRY['c10_150m_aug'] = c10_150m_aug

c10_150m_randaugment_dataaug = Hyperparams()
c10_150m_randaugment_dataaug.update(c10_150m_baseline)
c10_150m_randaugment_dataaug.rand_augment = True
c10_150m_randaugment_dataaug.rand_augment_n = 2
c10_150m_randaugment_dataaug.rand_augment_m = 3
HPARAMS_REGISTRY['c10_150m_randaugment_dataaug'] = c10_150m_randaugment_dataaug

c10_150m_randaugment_distaug = Hyperparams()
c10_150m_randaugment_distaug.update(c10_150m_baseline)
c10_150m_randaugment_distaug.rand_augment = True
c10_150m_randaugment_distaug.rand_augment_conditioning = True
c10_150m_randaugment_distaug.rand_augment_n = 2
c10_150m_randaugment_distaug.rand_augment_m = 3
HPARAMS_REGISTRY['c10_150m_randaugment_distaug'] = c10_150m_randaugment_distaug

c10_150m_rot = Hyperparams()
c10_150m_rot.update(c10_150m_baseline)
c10_150m_rot.use_rotation = True
HPARAMS_REGISTRY['c10-150m-rot'] = c10_150m_rot

c10_150m_rot_c_tr = Hyperparams()
c10_150m_rot_c_tr.update(c10_150m_baseline)
c10_150m_rot_c_tr.use_rotation = True
c10_150m_rot_c_tr.use_color = True
c10_150m_rot_c_tr.use_transposition = True
HPARAMS_REGISTRY['c10-150m-rot-c-tr'] = c10_150m_rot_c_tr

c10_150m_rot_c_tr_js = Hyperparams()
c10_150m_rot_c_tr_js.update(c10_150m_baseline)
c10_150m_rot_c_tr_js.use_rotation = True
c10_150m_rot_c_tr_js.use_color = True
c10_150m_rot_c_tr_js.use_transposition = True
c10_150m_rot_c_tr_js.use_jigsaw = True
c10_150m_rot_c_tr_js.jigsaw_grid_size = 2
HPARAMS_REGISTRY['c10-150m-rot-c-tr-js'] = c10_150m_rot_c_tr_js

c10_150m_rot_tr_js = Hyperparams()
c10_150m_rot_tr_js.update(c10_150m_baseline)
c10_150m_rot_tr_js.use_rotation = True
c10_150m_rot_tr_js.use_transposition = True
c10_150m_rot_tr_js.use_jigsaw = True
c10_150m_rot_tr_js.jigsaw_grid_size = 2
HPARAMS_REGISTRY['c10-150m-rot-tr-js'] = c10_150m_rot_tr_js

c10_150m_rot_c = Hyperparams()
c10_150m_rot_c.update(c10_150m_baseline)
c10_150m_rot_c.use_rotation = True
c10_150m_rot_c.use_color = True
HPARAMS_REGISTRY['c10-150m-rot-c'] = c10_150m_rot_c

c10_150m_rot_tr = Hyperparams()
c10_150m_rot_tr.update(c10_150m_baseline)
c10_150m_rot_tr.use_rotation = True
c10_150m_rot_tr.use_transposition = True
HPARAMS_REGISTRY['c10-150m-rot-tr'] = c10_150m_rot_tr

c10_150m_rot_tr_ra_n2_m3 = Hyperparams()
c10_150m_rot_tr_ra_n2_m3.update(c10_150m_baseline)
c10_150m_rot_tr_ra_n2_m3.use_rotation = True
c10_150m_rot_tr_ra_n2_m3.use_transposition = True
c10_150m_rot_tr_ra_n2_m3.rand_augment = True
c10_150m_rot_tr_ra_n2_m3.rand_augment_n = 2
c10_150m_rot_tr_ra_n2_m3.rand_augment_m = 3
c10_150m_rot_tr_ra_n2_m3.rand_augment_conditioning = True
c10_150m_rot_tr_ra_n2_m3.rand_augment_rate = 0.5
HPARAMS_REGISTRY['c10-150m-rot-tr-ra-n2-m3'] = c10_150m_rot_tr_ra_n2_m3

c10_150m_rot_tr_ra_n1_m2 = Hyperparams()
c10_150m_rot_tr_ra_n1_m2.update(c10_150m_baseline)
c10_150m_rot_tr_ra_n1_m2.use_rotation = True
c10_150m_rot_tr_ra_n1_m2.use_transposition = True
c10_150m_rot_tr_ra_n1_m2.rand_augment = True
c10_150m_rot_tr_ra_n1_m2.rand_augment_n = 1
c10_150m_rot_tr_ra_n1_m2.rand_augment_m = 2
c10_150m_rot_tr_ra_n1_m2.rand_augment_conditioning = True
c10_150m_rot_tr_ra_n1_m2.rand_augment_rate = 0.5
HPARAMS_REGISTRY['c10-150m-rot-tr-ra-n1-m2'] = c10_150m_rot_tr_ra_n1_m2

c10_150m_rot_c_tr_js_ra_n1_m2 = Hyperparams()
c10_150m_rot_c_tr_js_ra_n1_m2.update(c10_150m_baseline)
c10_150m_rot_c_tr_js_ra_n1_m2.use_rotation = True
c10_150m_rot_c_tr_js_ra_n1_m2.use_color = True
c10_150m_rot_c_tr_js_ra_n1_m2.use_transposition = True
c10_150m_rot_c_tr_js_ra_n1_m2.use_jigsaw = True
c10_150m_rot_c_tr_js_ra_n1_m2.jigsaw_grid_size = 2
c10_150m_rot_c_tr_js_ra_n1_m2.rand_augment = True
c10_150m_rot_c_tr_js_ra_n1_m2.rand_augment_n = 1
c10_150m_rot_c_tr_js_ra_n1_m2.rand_augment_m = 2
c10_150m_rot_c_tr_js_ra_n1_m2.rand_augment_conditioning = True
c10_150m_rot_c_tr_js_ra_n1_m2.rand_augment_rate = 0.5
HPARAMS_REGISTRY['c10-150m-rot-c-tr-js-ra-n1-m2'] = c10_150m_rot_c_tr_js_ra_n1_m2

c10_150m_c_tr = Hyperparams()
c10_150m_c_tr.update(c10_150m_baseline)
c10_150m_c_tr.use_color = True
c10_150m_c_tr.use_transposition = True
HPARAMS_REGISTRY['c10-150m-c-tr'] = c10_150m_c_tr

c10_10m_baseline = Hyperparams()
c10_10m_baseline.update(c10_150m_baseline)
c10_10m_baseline.n_embd = 128
c10_10m_baseline.n_batch = 16
HPARAMS_REGISTRY['c10_10m_baseline'] = c10_10m_baseline

c10_10m_rot = Hyperparams()
c10_10m_rot.update(c10_10m_baseline)
c10_10m_rot.use_rotation = True
HPARAMS_REGISTRY['c10_10m_rot'] = c10_10m_rot

c10_2m_baseline = Hyperparams()
c10_2m_baseline.update(c10_150m_baseline)
c10_2m_baseline.n_embd = 64
c10_2m_baseline.n_batch = 16
c10_2m_baseline.n_head = 8
HPARAMS_REGISTRY['c10_2m_baseline'] = c10_2m_baseline

c10_2m_rot = Hyperparams()
c10_2m_rot.update(c10_2m_baseline)
c10_2m_rot.use_rotation = True
HPARAMS_REGISTRY['c10_2m_rot'] = c10_2m_rot

i64_150m_32gpu = Hyperparams()
i64_150m_32gpu.update(imagenet64_8gpu)
i64_150m_32gpu.n_batch = 4
i64_150m_32gpu.lr = 0.00015
i64_150m_32gpu.l2_loss = 0.001
i64_150m_32gpu.total_epochs = 10000
i64_150m_32gpu.merge_layer_allreduce = 4
i64_150m_32gpu.n_layer = 48
i64_150m_32gpu.resid_pdrop = 0.005
i64_150m_32gpu.blocksize = 32
i64_150m_32gpu.pos_embd_std = 0.01
i64_150m_32gpu.w_embd_std = 0.01
i64_150m_32gpu.dropout_broadcast_dims = None
i64_150m_32gpu.dynamic_loss_scaling = True
i64_150m_32gpu.embd_pdrop = 0.0
i64_150m_32gpu.mlp_w2 = 0.125
i64_150m_32gpu.n_ctx = 12288
i64_150m_32gpu.n_head = 16
i64_150m_32gpu.attention_layers = 'b,bT,b,b'
HPARAMS_REGISTRY['i64_150m_32gpu'] = i64_150m_32gpu

i64_150m_32gpu_rot = Hyperparams()
i64_150m_32gpu_rot.update(i64_150m_32gpu)
i64_150m_32gpu_rot.use_rotation = True
HPARAMS_REGISTRY['i64_150m_32gpu_rot_32gpu'] = i64_150m_32gpu_rot

i64_150m_32gpu_rot_tr = Hyperparams()
i64_150m_32gpu_rot_tr.update(i64_150m_32gpu)
i64_150m_32gpu_rot_tr.use_rotation = True
i64_150m_32gpu_rot_tr.use_transposition = True
HPARAMS_REGISTRY['i64_150m_32gpu_rot_tr_32gpu'] = i64_150m_32gpu_rot_tr

i64_300m_64gpu = Hyperparams()
i64_300m_64gpu.update(i64_150m_32gpu)
i64_300m_64gpu.n_layer = 96
i64_300m_64gpu.n_batch = 2
HPARAMS_REGISTRY['i64_300m_64gpu'] = i64_300m_64gpu

i64_300m_64gpu_rot = Hyperparams()
i64_300m_64gpu_rot.update(i64_300m_64gpu)
i64_300m_64gpu_rot.use_rotation = True
HPARAMS_REGISTRY['i64_300m_64gpu_rot'] = i64_300m_64gpu_rot

i64_300m_64gpu_rot_tr = Hyperparams()
i64_300m_64gpu_rot_tr.update(i64_300m_64gpu)
i64_300m_64gpu_rot_tr.use_rotation = True
i64_300m_64gpu_rot_tr.use_transposition = True
HPARAMS_REGISTRY['i64_300m_64gpu_rot_tr'] = i64_300m_64gpu_rot_tr

i64_300m_64gpu_rot_c_tr = Hyperparams()
i64_300m_64gpu_rot_c_tr.update(i64_300m_64gpu)
i64_300m_64gpu_rot_c_tr.use_rotation = True
i64_300m_64gpu_rot_c_tr.use_color = True
i64_300m_64gpu_rot_c_tr.use_transposition = True
HPARAMS_REGISTRY['i64_300m_64gpu_rot_c_tr'] = i64_300m_64gpu_rot_c_tr


def parse_args_and_update_hparams(H, parser, s=None):
    args = parser.parse_args(s)
    valid_args = set(args.__dict__.keys())
    hparam_sets = [x for x in args.hparam_sets.split(',') if x]
    for hp_set in hparam_sets:
        hps = HPARAMS_REGISTRY[hp_set]
        for k in hps:
            if k not in valid_args:
                raise ValueError(f"{k} not in default args")
        parser.set_defaults(**hps)
    H.update(parser.parse_args().__dict__)
    # H is updated in place, so return nothing.


def add_arguments(parser):
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument('--desc', type=str, default='test')
    parser.add_argument('--print_params', action="store_true")
    parser.add_argument('--hparam_sets', '--hps', type=str, default='')

    # dataset params
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--auxiliary_dataset', type=str, default=None)
    parser.add_argument('--auxiliary_dataset_fraction', type=float, default=0.5)
    parser.add_argument('--auxiliary_dataset_subset_size', type=int, default=None)
    parser.add_argument('--auxiliary_dataset_seed', type=int, default=42)

    # Training params
    parser.add_argument('--n_batch', type=int, default=128)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # Transformer architectural parameters
    parser.add_argument('--n_embd', type=int, default=512)
    parser.add_argument('--n_ctx', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dropout_broadcast_dims', type=str, default=None)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--mlp_multiple', type=float, default=4.0)
    parser.add_argument('--qk_ratio', type=float, default=1.0)
    parser.add_argument('--attention_layers', type=str, default='a')
    parser.add_argument('--local_attn_ctx', type=int, default=64)
    parser.add_argument('--pos_embd_std', type=float, default=0.007)
    parser.add_argument('--w_embd_std', type=float, default=0.013)
    parser.add_argument('--mlp_w1', type=float, default=0.125)
    parser.add_argument('--mlp_w2', type=float, default=0.125)
    parser.add_argument('--qk_w', type=float, default=0.125)
    parser.add_argument('--v_w', type=float, default=0.125)
    parser.add_argument('--post_w', type=float, default=0.125)
    parser.add_argument('--logits_w', type=float, default=0.125)
    parser.add_argument('--preconv_w', type=float, default=0.125)

    # rand augment params
    # https://arxiv.org/pdf/1909.13719.pdf
    parser.add_argument('--rand_augment', action="store_true")
    parser.add_argument('--rand_augment_conditioning', action="store_true")
    parser.add_argument('--rand_augment_rate', type=float, default=0.95)
    parser.add_argument('--rand_augment_n', type=int, default=1)  # Number of sequential perturbations -- range [1, 3]
    parser.add_argument('--rand_augment_m', type=int, default=2)  # Magnitude of pertubations -- range [2, 30]

    # Distr Aug Params
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--permute_embeddings', dest='permute_embeddings', action="store_true")
    parser.add_argument('--no_permute_embeddings', dest='permute_embeddings', action="store_false")
    parser.set_defaults(permute_embeddings=True)
    parser.add_argument('--use_imagenet_fraction', type=float, default=1.0)
    parser.add_argument('--unaugmented_data_rate', type=float, default=None)
    parser.add_argument('--use_rotation', action="store_true")
    parser.add_argument('--use_dataset_conditioning', action="store_true")
    parser.add_argument('--no_dataset_conditioning', action="store_false", dest="use_dataset_conditioning")
    parser.add_argument('--use_color', action="store_true")
    parser.add_argument('--use_transposition', action="store_true")
    parser.add_argument('--use_randomly_determined_order', action="store_true")
    parser.add_argument('--randomly_determined_order_num_perms', type=int, default=3)
    parser.add_argument('--randomly_determined_order_seed', type=int, default=42)
    parser.add_argument('--randomly_determined_order_use_lookahead', action="store_true")
    parser.add_argument('--use_reverse', action="store_true")
    parser.add_argument('--use_linf_pgd', action="store_true")
    parser.add_argument('--use_jigsaw', action="store_true")
    parser.add_argument('--jigsaw_grid_size', type=int, default=2)
    parser.add_argument('--use_unconditional_augmentation', action='store_true')

    parser.add_argument('--datapoints', type=int, default=None)
    parser.add_argument('--test_size', type=int, default=None)
    # Training params
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--aug_seed', type=int, default=314)
    parser.add_argument('--optimizer', type=str, default='bs_adam')
    parser.add_argument('--activation', type=str, default='quick_gelu')
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--l2_loss', type=float, default=0.0)
    parser.add_argument('--recompute', action="store_true", dest="recompute")
    parser.add_argument('--no_recompute', action="store_false", dest="recompute")
    parser.add_argument('--float16', action="store_true")
    parser.add_argument('--no_float16', action="store_false", dest='float16')
    parser.add_argument('--blocksparse_op', action="store_true")
    parser.add_argument('--no_blocksparse_op', action="store_false", dest="blocksparse_op")
    parser.add_argument('--blocksize', type=int, default=64)
    parser.add_argument('--fp16_allreduce', action="store_true")
    parser.add_argument('--no_fp16_allreduce', action="store_false", dest='fp16_allreduce')
    parser.add_argument('--merge_layer_allreduce', default=0, type=int)
    parser.add_argument('--fp32_gains_biases', action="store_true")
    parser.add_argument('--fp16_loss_scale', type=float, default=2.0**16)
    parser.add_argument('--min_loss_scale', type=float, default=2.0**10)
    parser.add_argument('--fp16_loss_freq', type=int, default=1000)
    parser.add_argument('--fp16_mean_var', action='store_true')
    parser.add_argument('--no_fp16_mean_var', action='store_false',
                        dest='fp16_mean_var')
    parser.add_argument('--dynamic_loss_scaling', action='store_true')
    parser.add_argument('--no_dynamic_loss_scaling', action='store_false',
                        dest='dynamic_loss_scaling')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lr_offset', type=int, default=0)
    parser.add_argument('--decay_lr_linearly', action="store_true")
    parser.add_argument('--no_vocab_rounding', action="store_true")
    parser.add_argument('--disable_ema_vars', action="store_true")
    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--exit_after_n_epochs', type=int, default=None)
    parser.add_argument('--warmup_iters', type=int, default=5000)
    parser.add_argument('--weights_beta', type=float, default=0.999)
    parser.add_argument('--iters_per_log', type=int, default=500)
    parser.add_argument('--aug_eval', type=str, default=None)
    parser.add_argument('--aug_eval_n_examples', type=int, default=None)
    parser.add_argument('--eval_after_n_examples', type=int, default=None)
    parser.add_argument('--epochs_per_save', type=int, default=1)
    parser.add_argument('--epochs_per_backup', type=int, default=1)
    parser.add_argument('--epochs_per_eval', type=int, default=1)

    # eval stuff
    parser.add_argument('--skip_initial_evals', action="store_true")
    parser.add_argument('--eval_and_exit', action="store_true")
    parser.add_argument('--no_skip_initial_evals', action="store_false",
                        dest='skip_initial_evals')
    parser.add_argument('--eval_test', action="store_true")
    parser.add_argument('--eval_start_idx', type=int, default=0)
    parser.add_argument('--eval_n_examples', type=int, default=100000)

    # Generating unconditional samples
    parser.add_argument('--sample_batch', type=int, default=4)
    parser.add_argument('--samples_to_generate', type=int, default=4)
    parser.add_argument('--sample_grid_dim', type=int, default=4)
    parser.add_argument('--sample_and_exit', action="store_true")
    parser.add_argument('--sample_during_eval', action="store_true")
    parser.add_argument('--sample_f16', action="store_true")
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--no_sample_during_eval', action="store_false", dest='sample_during_eval')

    # Restoring jobs
    parser.add_argument('--restore_path', type=str, default='')
    return parser
