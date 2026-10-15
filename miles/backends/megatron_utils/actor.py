import json
import logging
import random
import socket
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig

from miles.ray.train_actor import TrainRayActor
from miles.utils import tracking_utils, train_dump_utils
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group, init_process_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.processing_utils import load_tokenizer
from miles.utils.ray_utils import Box
from miles.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from miles.utils.replay_base import all_replay_managers
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from ..training_utils.cp_utils import slice_with_cp
from ..training_utils.data import DataIterator, get_data_iterator, get_rollout_data, sync_actor_critic_data
from ..training_utils.log_utils import log_cpu_memory, log_perf_data, log_rollout_data
from ..training_utils.loss import compute_advantages_and_returns, get_log_probs_and_entropy, get_values
from ..training_utils.parallel import get_parallel_state
from .checkpoint import load_checkpoint
from .head_sparse import apply_head_mask, restore_frozen_values
from .initialize import init, is_megatron_main_rank
from .lora_utils import (
    apply_lora_head_nonzero_value_freeze,
    apply_lora_head_random_freeze,
    apply_lora_head_value_freeze_from_manifest,
    apply_full_weight_neuron_value_freeze_from_manifest,
    apply_lora_mlp_nonzero_value_freeze,
    apply_lora_o_proj_value_freeze,
    collect_lora_qkv_head_activity,
    collect_lora_head_sparsity_stats,
    enforce_lora_head_freeze_mask,
    enforce_lora_value_freeze_mask,
    is_lora_enabled,
)
from .model import forward_only, initialize_model_and_optimizer, save, train
from .parallel import verify_megatron_parallel_state
from .replay_utils import get_register_replay_list_func
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed.broadcast import UpdateWeightFromDistributed
from .update_weight.update_weight_from_distributed.p2p import UpdateWeightP2P
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
    ) -> int | None:
        monkey_patch_torch_dist()

        super().init(args, role, with_ref)

        init(args)

        if args.dumper_enable:
            from sglang.srt.debug_utils.dumper import dumper

            dumper.apply_source_patches()

        self._is_main_rank = is_megatron_main_rank()
        self._lora_zero_mask_applied = False
        self._lora_head_nonzero_value_freeze_applied = False
        self._full_weight_head_value_freeze_applied = False
        self._full_weight_neuron_value_freeze_applied = False
        self._lora_head_activation_seen: set[tuple[int, str, int]] = set()
        self._lora_head_activation_count = 0

        if self._is_main_rank:
            init_tracking(args, primary=False)

        unsupported = {"train_actor", "train_log_probs"} & set(args.profile_target)
        if unsupported and args.use_pytorch_profiler:
            raise NotImplementedError(
                f"--profile-target {' '.join(sorted(unsupported))} is not supported for Megatron backend"
            )
        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = load_tokenizer(
                    self.args.hf_checkpoint, chat_template_path=self.args.chat_template_path, trust_remote_code=True
                )
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {
            "dp_size": get_parallel_state().intra_dp.size,
        }
        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                # --train-memory-margin-bytes can tune this
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        if self.args.debug_rollout_only:
            return 0

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters
        else:
            for m in all_replay_managers:
                m.enabled = getattr(self.args, f"use_{m.name}_replay")
                m.enable_check_replay_result = m.enabled and self.args.ci_test

        self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
            args, role
        )

        parallel_state = get_parallel_state()
        if parallel_state.cp.size > 1:
            from miles_plugins.models.cp_utils import detect_and_setup_hybrid_cp

            for model_chunk in self.model:
                detect_and_setup_hybrid_cp(
                    model_chunk, parallel_state.cp.group, parallel_state.cp.rank, parallel_state.cp.size
                )

        verify_megatron_parallel_state(self.model)

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return

        random_freeze_zero_ratio = getattr(self.args, "lora_head_random_freeze_zero_ratio_at", -1.0)
        if random_freeze_zero_ratio >= 0.0:
            parallel_state = get_parallel_state()
            random_freeze_metrics = apply_lora_head_random_freeze(
                self.model,
                self.args.kv_channels,
                random_freeze_zero_ratio,
                getattr(self.args, "lora_head_random_freeze_seed", 1234),
                shard_id=f"tp{parallel_state.tp.rank}:pp{parallel_state.pp.rank}",
            )
            self._lora_zero_mask_applied = random_freeze_metrics is not None
            if self._is_main_rank and random_freeze_metrics is not None:
                log_dict = {f"lora_head_random_freeze/{key}": value for key, value in random_freeze_metrics.items()}
                log_dict["lora_head_random_freeze/applied"] = 1.0
                log_dict["rollout/step"] = 0.0
                tracking_utils.log(self.args, log_dict, step_key="rollout/step")
                logger.info(f"lora_head_random_freeze: {log_dict}")

        head_freeze_mask_path = getattr(self.args, "lora_head_freeze_mask_path", None)
        if head_freeze_mask_path is not None:
            parallel_state = get_parallel_state()
            if parallel_state.pp.size != 1:
                raise NotImplementedError(
                    "--lora-head-freeze-mask-path currently requires pipeline-model-parallel-size=1"
                )
            manifest_freeze_metrics = apply_lora_head_value_freeze_from_manifest(
                self.model,
                head_freeze_mask_path,
                self.args.kv_channels,
                self.args.num_attention_heads,
                self.args.num_query_groups,
                tp_rank=parallel_state.tp.rank,
                tp_size=parallel_state.tp.size,
                num_layers=self.args.num_layers,
            )
            self._lora_head_nonzero_value_freeze_applied = manifest_freeze_metrics is not None
            if self._is_main_rank and manifest_freeze_metrics is not None:
                log_dict = {
                    f"lora_head_manifest_freeze/{key}": value for key, value in manifest_freeze_metrics.items()
                }
                log_dict["lora_head_manifest_freeze/applied"] = 1.0
                log_dict["rollout/step"] = 0.0
                tracking_utils.log(self.args, log_dict, step_key="rollout/step")
                logger.info(f"lora_head_manifest_freeze: {log_dict}")

        full_weight_head_freeze_mask_path = getattr(self.args, "full_weight_head_freeze_mask_path", None)
        if full_weight_head_freeze_mask_path is not None:
            parallel_state = get_parallel_state()
            if parallel_state.pp.size != 1:
                raise NotImplementedError(
                    "--full-weight-head-freeze-mask-path currently requires pipeline-model-parallel-size=1"
                )
            full_weight_freeze_metrics = apply_head_mask(
                self.model,
                full_weight_head_freeze_mask_path,
                self.args.kv_channels,
                self.args.num_attention_heads,
                self.args.num_query_groups,
                include_output_projection=getattr(self.args, "full_weight_head_freeze_include_o_proj", True),
                tp_rank=parallel_state.tp.rank,
                tp_size=parallel_state.tp.size,
                num_layers=self.args.num_layers,
            )
            self._full_weight_head_value_freeze_applied = full_weight_freeze_metrics is not None
            if self._is_main_rank and full_weight_freeze_metrics is not None:
                log_dict = {
                    f"full_weight_head_freeze/{key}": value for key, value in full_weight_freeze_metrics.items()
                }
                log_dict["full_weight_head_freeze/applied"] = 1.0
                log_dict["rollout/step"] = 0.0
                tracking_utils.log(self.args, log_dict, step_key="rollout/step")
                logger.info(f"full_weight_head_freeze: {log_dict}")

        full_weight_neuron_freeze_mask_path = getattr(self.args, "full_weight_neuron_freeze_mask_path", None)
        if full_weight_neuron_freeze_mask_path is not None:
            parallel_state = get_parallel_state()
            if parallel_state.pp.size != 1:
                raise NotImplementedError(
                    "--full-weight-neuron-freeze-mask-path currently requires pipeline-model-parallel-size=1"
                )
            neuron_freeze_metrics = apply_full_weight_neuron_value_freeze_from_manifest(
                self.model,
                full_weight_neuron_freeze_mask_path,
                self.args.ffn_hidden_size,
                tp_rank=parallel_state.tp.rank,
                tp_size=parallel_state.tp.size,
                num_layers=self.args.num_layers,
            )
            self._full_weight_neuron_value_freeze_applied = neuron_freeze_metrics is not None
            if self._is_main_rank and neuron_freeze_metrics is not None:
                log_dict = {f"full_weight_neuron_freeze/{key}": value for key, value in neuron_freeze_metrics.items()}
                log_dict["full_weight_neuron_freeze/applied"] = 1.0
                log_dict["rollout/step"] = 0.0
                tracking_utils.log(self.args, log_dict, step_key="rollout/step")
                logger.info(f"full_weight_neuron_freeze: {log_dict}")

        if getattr(self.args, "lora_head_freeze_nonzero_on_load", False):
            if getattr(self.args, "lora_adapter_path", None) is None:
                if self._is_main_rank:
                    logger.warning(
                        "--lora-head-freeze-nonzero-on-load was set without --lora-adapter-path; "
                        "skipping non-zero LoRA head freeze."
                    )
            else:
                nonzero_freeze_metrics = apply_lora_head_nonzero_value_freeze(
                    self.model,
                    self.args.kv_channels,
                    getattr(self.args, "lora_head_freeze_nonzero_threshold", 1e-6),
                    getattr(self.args, "lora_head_freeze_nonzero_include_o_proj", False),
                    getattr(self.args, "lora_head_freeze_nonzero_reference_adapter_path", None),
                )
                self._lora_head_nonzero_value_freeze_applied = nonzero_freeze_metrics is not None
                if self._is_main_rank and nonzero_freeze_metrics is not None:
                    log_dict = {
                        f"lora_head_nonzero_freeze/{key}": value for key, value in nonzero_freeze_metrics.items()
                    }
                    log_dict["rollout/step"] = 0.0
                    tracking_utils.log(self.args, log_dict, step_key="rollout/step")
                    logger.info(f"lora_head_nonzero_freeze_on_load: {log_dict}")

        if getattr(self.args, "lora_o_proj_freeze_on_load", False):
            if getattr(self.args, "lora_adapter_path", None) is None:
                if self._is_main_rank:
                    logger.warning(
                        "--lora-o-proj-freeze-on-load was set without --lora-adapter-path; "
                        "skipping O projection LoRA freeze."
                    )
            else:
                o_proj_freeze_metrics = apply_lora_o_proj_value_freeze(self.model)
                self._lora_head_nonzero_value_freeze_applied = (
                    self._lora_head_nonzero_value_freeze_applied or o_proj_freeze_metrics is not None
                )
                if self._is_main_rank and o_proj_freeze_metrics is not None:
                    log_dict = {f"lora_o_proj_freeze/{key}": value for key, value in o_proj_freeze_metrics.items()}
                    log_dict["rollout/step"] = 0.0
                    tracking_utils.log(self.args, log_dict, step_key="rollout/step")
                    logger.info(f"lora_o_proj_freeze_on_load: {log_dict}")

        if getattr(self.args, "lora_mlp_freeze_on_load", False) or getattr(
            self.args, "lora_mlp_freeze_nonzero_on_load", False
        ):
            if getattr(self.args, "lora_adapter_path", None) is None:
                if self._is_main_rank:
                    logger.warning(
                        "--lora-mlp-freeze-on-load/--lora-mlp-freeze-nonzero-on-load was set without "
                        "--lora-adapter-path; skipping MLP LoRA freeze."
                    )
            else:
                mlp_freeze_metrics = apply_lora_mlp_nonzero_value_freeze(
                    self.model,
                    getattr(self.args, "lora_mlp_freeze_nonzero_threshold", 1e-6),
                    getattr(self.args, "lora_mlp_freeze_on_load", False),
                )
                self._lora_head_nonzero_value_freeze_applied = (
                    self._lora_head_nonzero_value_freeze_applied or mlp_freeze_metrics is not None
                )
                if self._is_main_rank and mlp_freeze_metrics is not None:
                    log_prefix = (
                        "lora_mlp_freeze"
                        if getattr(self.args, "lora_mlp_freeze_on_load", False)
                        else "lora_mlp_nonzero_freeze"
                    )
                    log_dict = {f"{log_prefix}/{key}": value for key, value in mlp_freeze_metrics.items()}
                    log_dict["rollout/step"] = 0.0
                    tracking_utils.log(self.args, log_dict, step_key="rollout/step")
                    logger.info(f"lora_mlp_nonzero_freeze_on_load: {log_dict}")

        start_rollout_id = loaded_rollout_id + 1

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size

        if self.args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        else:
            if self.args.update_weight_transfer_mode == "broadcast":
                update_weight_cls = UpdateWeightFromDistributed
            else:
                update_weight_cls = UpdateWeightP2P
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            is_lora=is_lora_enabled(args),
        )

        # empty cache after initialization
        clear_memory()

        self._switch_model("actor")
        if self.args.offload_train:
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from miles.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        destroy_process_groups()

        tag = "default" if is_lora_enabled(self.args) else None
        torch_memory_saver.pause(tag=tag)

        print_memory("after offload model")

        if self._is_main_rank and hasattr(self, "_last_rollout_id"):
            log_cpu_memory(self._last_rollout_id, self.args, "after_offload_train")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        tag = "default" if is_lora_enabled(self.args) else None
        torch_memory_saver.resume(tag=tag)

        clear_memory()
        reload_process_groups()
        print_memory("after wake_up model")

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def _set_replay_stage(self, stage: str) -> None:
        for m in all_replay_managers:
            m.stage = stage

    def _fill_replay_data(
        self,
        data_iterator,
        num_microbatches,
        rollout_data,
        data_key: str,
        replay_list: list,
        register_replay_list_func,
        if_sp_region=True,
    ):
        if data_key not in rollout_data:
            raise ValueError(f"{data_key} is required in rollout_data for replay.")

        for iterator in data_iterator:
            iterator.reset()

        parallel_state = get_parallel_state()
        tp_rank = parallel_state.tp.rank
        tp_size = parallel_state.tp.size
        qkv_format = self.args.qkv_format

        def pad_func(data, pad):
            _, num_layers, topk = data.shape
            pad_tensor = torch.full(
                (pad, num_layers, topk),
                fill_value=-1,
                device=data.device,
                dtype=data.dtype,
            )
            return torch.cat([data, pad_tensor], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next([data_key, "tokens", "max_seq_lens"])
            replay_data = batch[data_key]
            tokens = batch["tokens"]
            assert len(replay_data) == len(tokens)
            for a, b in zip(replay_data, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            replay_data = [pad_func(r, 1) for r in replay_data]
            # TODO: maybe extract a common process function for here and get_batch?

            if qkv_format == "bshd":
                max_seqlen = batch["max_seq_lens"][0]
                replay_data = [slice_with_cp(r, pad_func, qkv_format, max_seqlen) for r in replay_data]
                replay_data = torch.stack(replay_data, dim=0)
                batch_size, seqlen, num_layers, topk = replay_data.shape
                replay_data = replay_data.reshape(batch_size * seqlen, num_layers, topk)
            else:
                replay_data = [slice_with_cp(r, pad_func, qkv_format) for r in replay_data]
                replay_data = torch.cat(replay_data, dim=0)
                pad_size = parallel_state.tp.size * self.args.data_pad_size_multiplier
                pad = (pad_size - replay_data.size(0) % pad_size) % pad_size
                if pad != 0:
                    replay_data = pad_func(replay_data, pad)

            if self.args.sequence_parallel and if_sp_region:
                seqlen = replay_data.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                replay_data = replay_data[start:end]

            register_replay_list_func(replay_list, replay_data, self.model)

        del rollout_data[data_key]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        self._last_rollout_id = rollout_id
        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = get_rollout_data(self.args, rollout_data_ref)
            if self.args.debug_rollout_only:
                log_rollout_data(rollout_id, self.args, rollout_data)
                return

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
        )

    def _use_rollout_replay(self, m) -> bool:
        return getattr(self.args, f"use_rollout_{m.name}_replay")

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        for m in all_replay_managers:
            if self._use_rollout_replay(m):
                self._fill_replay_data(
                    data_iterator,
                    num_microbatches,
                    rollout_data,
                    data_key=m.data_key,
                    replay_list=m.replays,
                    register_replay_list_func=get_register_replay_list_func(m),
                    if_sp_region=m.if_sp_region,
                )

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    self._set_replay_stage("fallthrough")
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )
                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    for m in all_replay_managers:
                        if m.enabled:
                            if self._use_rollout_replay(m):
                                m.stage = "replay_forward"
                            else:
                                m.stage = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    for m in all_replay_managers:
                        if self._use_rollout_replay(m):
                            m.clear_all_forward()

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            lora_enabled = is_lora_enabled(self.args)

            # Gradient-importance criterion: main_grad is zeroed once
            # optimizer.step runs, so snapshot the fused-qkv gradient just BEFORE
            # the step by wrapping optimizer.step once. Gated off by default.
            head_grad_importance_output = getattr(self.args, "head_grad_importance_output", None)
            mlp_neuron_grad_importance_output = getattr(self.args, "mlp_neuron_grad_importance_output", None)
            if (head_grad_importance_output or mlp_neuron_grad_importance_output) and not getattr(
                self, "_head_grad_step_patched", False
            ):
                if mlp_neuron_grad_importance_output:
                    from miles.utils.mlp_neuron_grad_importance import assert_single_dp_for_grad_norms

                    assert_single_dp_for_grad_norms(get_parallel_state().intra_dp.size)
                    if not getattr(self.args, "ffn_hidden_size", None):
                        raise ValueError(
                            "--mlp-neuron-grad-importance-output needs --ffn-hidden-size to map the fused "
                            "linear_fc1 rows onto neurons"
                        )
                self._head_grad_snapshot = {}
                self._mlp_neuron_grad_local_parts = []
                _orig_optimizer_step = self.optimizer.step

                def _head_grad_snapshot_step(*step_args, **step_kwargs):
                    if head_grad_importance_output:
                        from miles.utils.head_grad_importance import snapshot_qkv_head_gradients

                        self._head_grad_snapshot = snapshot_qkv_head_gradients(self.model, self.optimizer)
                    if mlp_neuron_grad_importance_output:
                        # Reduce to per-neuron norms in place: a full fc1/fc2 gradient
                        # snapshot would be several GiB per rank.
                        from miles.utils.mlp_neuron_grad_importance import (
                            compute_local_mlp_neuron_grad_norms,
                        )

                        _ps = get_parallel_state()
                        self._mlp_neuron_grad_local_parts = compute_local_mlp_neuron_grad_norms(
                            self.model,
                            self.args.ffn_hidden_size,
                            tp_rank=_ps.tp.rank,
                            tp_size=_ps.tp.size,
                        )
                    return _orig_optimizer_step(*step_args, **step_kwargs)

                self.optimizer.step = _head_grad_snapshot_step
                self._head_grad_step_patched = True

            # Train
            self._set_replay_stage("replay_backward")
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            # Dump the per-group L2 norm of the gradient snapshot captured just
            # before the step (see the optimizer.step wrapper above). Gated off by
            # default so it has zero effect on existing runs.
            if head_grad_importance_output:
                from miles.utils.head_grad_importance import collect_qkv_head_grad_norms

                ps = get_parallel_state()
                grad_events = collect_qkv_head_grad_norms(
                    self.model,
                    self.args.kv_channels,
                    self.args.num_attention_heads,
                    self.args.num_query_groups,
                    tp_rank=ps.tp.rank,
                    tp_size=ps.tp.size,
                    grad_snapshot=getattr(self, "_head_grad_snapshot", None),
                )
                if self._is_main_rank and grad_events:
                    grad_events.sort(key=lambda e: float(e["grad_norm"]), reverse=True)
                    out_path = Path(head_grad_importance_output)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    step_val = int(compute_rollout_step(self.args, rollout_id))
                    with out_path.open("a") as f:
                        for rank_idx, event in enumerate(grad_events):
                            record = dict(event)
                            record["importance_rank"] = int(rank_idx)
                            record["rollout_id"] = int(rollout_id)
                            record["rollout_step"] = step_val
                            record["score_method"] = "optimizer_owned_global_gradient_v2"
                            f.write(json.dumps(record, sort_keys=True) + "\n")
                    logger.info(
                        f"head_grad_importance {rollout_id}: wrote {len(grad_events)} groups "
                        f"(top grad_norm={grad_events[0]['grad_norm']:.4g}, "
                        f"min={grad_events[-1]['grad_norm']:.4g})"
                    )

            # Same criterion at MLP-neuron granularity: one group per FFN
            # intermediate channel (gate row + up row + down column). Gated off by
            # default so it has zero effect on existing runs.
            if mlp_neuron_grad_importance_output:
                from miles.utils.mlp_neuron_grad_importance import merge_mlp_neuron_grad_norms

                neuron_events = merge_mlp_neuron_grad_norms(getattr(self, "_mlp_neuron_grad_local_parts", []))
                if self._is_main_rank and neuron_events:
                    neuron_events.sort(key=lambda e: float(e["grad_norm"]), reverse=True)
                    out_path = Path(mlp_neuron_grad_importance_output)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    step_val = int(compute_rollout_step(self.args, rollout_id))
                    with out_path.open("a") as f:
                        for rank_idx, event in enumerate(neuron_events):
                            record = {
                                "layer": int(event["layer"]),
                                "neuron": int(event["neuron"]),
                                "grad_norm": float(f"{event['grad_norm']:.6g}"),
                                "gate_norm": float(f"{event['gate_norm']:.6g}"),
                                "up_norm": float(f"{event['up_norm']:.6g}"),
                                "down_norm": float(f"{event['down_norm']:.6g}"),
                                "importance_rank": int(rank_idx),
                                "rollout_id": int(rollout_id),
                                "rollout_step": step_val,
                            }
                            f.write(json.dumps(record, sort_keys=True) + "\n")
                    logger.info(
                        f"mlp_neuron_grad_importance {rollout_id}: wrote {len(neuron_events)} neurons "
                        f"(top grad_norm={neuron_events[0]['grad_norm']:.4g}, "
                        f"min={neuron_events[-1]['grad_norm']:.4g})"
                    )

            if lora_enabled and self._lora_head_nonzero_value_freeze_applied:
                enforce_lora_value_freeze_mask(self.model)

            if self._full_weight_head_value_freeze_applied:
                restore_frozen_values(self.model)

            if self._full_weight_neuron_value_freeze_applied:
                enforce_lora_value_freeze_mask(self.model)

            if lora_enabled and self._lora_zero_mask_applied:
                freeze_mask_stats = enforce_lora_head_freeze_mask(self.model)
                if self._is_main_rank:
                    log_dict = {f"lora_head_zero_mask/{key}": value for key, value in freeze_mask_stats.items()}
                    log_dict["lora_head_zero_mask/applied"] = 1.0
                    log_dict["lora_head_zero_mask/rollout_id"] = float(rollout_id)
                    log_dict["rollout/step"] = compute_rollout_step(self.args, rollout_id)
                    tracking_utils.log(self.args, log_dict, step_key="rollout/step")
                    logger.info(f"lora_head_zero_mask {rollout_id}: {log_dict}")

            if lora_enabled and self._lora_head_nonzero_value_freeze_applied:
                value_freeze_stats = enforce_lora_value_freeze_mask(self.model)
                if self._is_main_rank:
                    log_dict = {f"lora_value_freeze/{key}": value for key, value in value_freeze_stats.items()}
                    if self._lora_head_nonzero_value_freeze_applied:
                        log_dict.update(
                            {f"lora_head_nonzero_freeze/{key}": value for key, value in value_freeze_stats.items()}
                        )
                    log_dict["lora_value_freeze/applied"] = 1.0
                    log_dict["lora_value_freeze/rollout_id"] = float(rollout_id)
                    if self._lora_head_nonzero_value_freeze_applied:
                        log_dict["lora_head_nonzero_freeze/applied"] = 1.0
                        log_dict["lora_head_nonzero_freeze/rollout_id"] = float(rollout_id)
                    log_dict["rollout/step"] = compute_rollout_step(self.args, rollout_id)
                    tracking_utils.log(self.args, log_dict, step_key="rollout/step")
                    logger.info(f"lora_value_freeze {rollout_id}: {log_dict}")

            activation_order_output = getattr(self.args, "lora_head_activation_order_output", None)
            if lora_enabled and activation_order_output:
                parallel_state = get_parallel_state()
                active_heads = collect_lora_qkv_head_activity(
                    self.model,
                    self.args.kv_channels,
                    getattr(self.args, "lora_head_activation_order_threshold", 1e-6),
                    self.args.num_attention_heads,
                    self.args.num_query_groups,
                    tp_rank=parallel_state.tp.rank,
                    tp_size=parallel_state.tp.size,
                )
                new_events = []
                for event in active_heads:
                    key = (int(event["layer"]), str(event["projection"]), int(event["head"]))
                    if key in self._lora_head_activation_seen:
                        continue
                    self._lora_head_activation_seen.add(key)
                    event = dict(event)
                    event["activation_index"] = self._lora_head_activation_count
                    event["rollout_id"] = int(rollout_id)
                    event["rollout_step"] = int(compute_rollout_step(self.args, rollout_id))
                    self._lora_head_activation_count += 1
                    new_events.append(event)
                if self._is_main_rank and new_events:
                    output_path = Path(activation_order_output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with output_path.open("a") as f:
                        for event in new_events:
                            f.write(json.dumps(event, sort_keys=True) + "\n")
                    log_dict = {
                        "lora_head_activation_order/new_heads": float(len(new_events)),
                        "lora_head_activation_order/total_heads": float(len(self._lora_head_activation_seen)),
                        "lora_head_activation_order/rollout_id": float(rollout_id),
                        "rollout/step": compute_rollout_step(self.args, rollout_id),
                    }
                    tracking_utils.log(self.args, log_dict, step_key="rollout/step")
                    logger.info(f"lora_head_activation_order {rollout_id}: {log_dict}")

            if lora_enabled:
                head_sparsity_metrics = collect_lora_head_sparsity_stats(self.model, self.args.kv_channels)
                if self._is_main_rank and head_sparsity_metrics is not None:
                    log_dict = {f"lora_head_sparsity/{key}": value for key, value in head_sparsity_metrics.items()}
                    log_dict["lora_head_sparsity/rollout_id"] = float(rollout_id)
                    log_dict["rollout/step"] = compute_rollout_step(self.args, rollout_id)
                    tracking_utils.log(self.args, log_dict, step_key="rollout/step")
                    logger.info(f"lora_head_sparsity {rollout_id}: {log_dict}")

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        for m in all_replay_managers:
            if m.enabled:
                m.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args)

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from miles.backends.megatron_utils.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            destroy_process_groups()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote()
        )

        if self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        if self.args.debug_skip_weight_update:
            if dist.get_rank() == 0:
                logger.warning("Skipping actor-to-rollout weight update because " "--debug-skip-weight-update is set.")
            if self.args.offload_train:
                destroy_process_groups()
            return

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0 and not is_lora_enabled(self.args):
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        self._actor_critic_groups = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )
