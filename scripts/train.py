# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Entry point for Dora to launch solvers for running training loops.
See more info on how to use Dora: https://github.com/facebookresearch/dora
"""

import logging
import multiprocessing
import os
from pathlib import Path
import sys
import typing as tp

from dora import git_save, hydra_main, XP
import flashy
import hydra
import omegaconf

from audiocraft.environment import AudioCraftEnvironment
from audiocraft.utils.cluster import get_slurm_parameters

logger = logging.getLogger(__name__)

class SonicSealTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def resolve_config_dset_paths(self):
        """Enable Dora to load manifest from git clone repository."""
        # Manifest files for the different splits
        for key, value in self.cfg.datasource.items():
            if isinstance(value, str):
                self.cfg.datasource[key] = git_save.to_absolute_path(value)

    def get_solver(self):
        from audiocraft import solvers
        # Convert batch size to batch size for each GPU
        assert self.cfg.dataset.batch_size % flashy.distrib.world_size() == 0
        self.cfg.dataset.batch_size //= flashy.distrib.world_size()
        for split in ['train', 'valid', 'evaluate', 'generate']:
            if hasattr(self.cfg.dataset, split) and hasattr(self.cfg.dataset[split], 'batch_size'):
                assert self.cfg.dataset[split].batch_size % flashy.distrib.world_size() == 0
                self.cfg.dataset[split].batch_size //= flashy.distrib.world_size()
        self.resolve_config_dset_paths()
        solver = solvers.get_solver(self.cfg)
        return solver

    @staticmethod
    def get_solver_from_xp(xp: XP, override_cfg: tp.Optional[tp.Union[dict, omegaconf.DictConfig]] = None,
                           restore: bool = True, load_best: bool = True,
                           ignore_state_keys: tp.List[str] = [], disable_fsdp: bool = True):
        """Given a XP, return the Solver object."""
        logger.info(f"Loading solver from XP {xp.sig}. Overrides used: {xp.argv}")
        cfg = xp.cfg
        if override_cfg is not None:
            cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.DictConfig(override_cfg))
        if disable_fsdp and cfg.fsdp.use:
            cfg.fsdp.use = False
            ignore_state_keys += ['model', 'ema', 'best_state']

        try:
            with xp.enter():
                trainer = SonicSealTrainer(cfg)
                solver = trainer.get_solver()
                if restore:
                    solver.restore(load_best=load_best, ignore_state_keys=ignore_state_keys)
            return solver
        finally:
            hydra.core.global_hydra.GlobalHydra.instance().clear()

    @staticmethod
    def get_solver_from_sig(sig: str, *args, **kwargs):
        """Return Solver object from Dora signature, i.e. to play with it from a notebook.
        See `get_solver_from_xp` for more information.
        """
        xp = main.get_xp_from_sig(sig)
        return SonicSealTrainer.get_solver_from_xp(xp, *args, **kwargs)

    def init_seed_and_system(self):
        import numpy as np
        import torch
        import random
        from audiocraft.modules.transformer import set_efficient_attention_backend

        multiprocessing.set_start_method(self.cfg.mp_start_method)
        logger.debug('Setting mp start method to %s', self.cfg.mp_start_method)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.set_num_threads(self.cfg.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.cfg.num_threads)
        os.environ['OMP_NUM_THREADS'] = str(self.cfg.num_threads)
        logger.debug('Setting num threads to %d', self.cfg.num_threads)
        set_efficient_attention_backend(self.cfg.efficient_attention_backend)
        logger.debug('Setting efficient attention backend to %s', self.cfg.efficient_attention_backend)
        if 'SLURM_JOB_ID' in os.environ:
            tmpdir = Path('/scratch/slurm_tmpdir/' + os.environ['SLURM_JOB_ID'])
            if tmpdir.exists():
                logger.info("Changing tmpdir to %s", tmpdir)
                os.environ['TMPDIR'] = str(tmpdir)

    def run(self):
        self.init_seed_and_system()

        # Setup logging both to XP specific folder, and to stderr.
        log_name = '%s.log.{rank}' % self.cfg.execute_only if self.cfg.execute_only else 'solver.log.{rank}'
        flashy.setup_logging(level=str(self.cfg.logging.level).upper(), log_name=log_name)
        flashy.distrib.init()  # Initialize distributed training

        solver = self.get_solver()

        if self.cfg.show:
            solver.show()
            return

        if self.cfg.execute_only:
            assert self.cfg.execute_inplace or self.cfg.continue_from is not None, \
                "Please specify the checkpoint to continue from with continue_from=<sig_or_path> " + \
                "or set execute_inplace to True."
            solver.restore(replay_metrics=False)
            solver.run_one_stage(self.cfg.execute_only)
            return

        return solver.run()

# Correctly set the Dora directory using a global variable
DORA_DIR = AudioCraftEnvironment.get_dora_dir()

@hydra.main(config_path='/app/config/model/sonicseal', config_name='sonicseal', version_base='1.1')
def main(cfg):
    try:
        print(f"Current Working Directory: {os.getcwd()}")
        print(f"Config Search Path: {hydra.core.global_hydra.GlobalHydra.instance().config_loader.get_search_path()}")
        print(f"Config contents: {cfg}")
        
        # Initialize seed and system settings
        init_seed_and_system(cfg)
        
        # Setup logging both to XP specific folder, and to stderr.
        log_name = '%s.log.{rank}' % cfg.execute_only if cfg.execute_only else 'solver.log.{rank}'
        flashy.setup_logging(level=str(cfg.logging.level).upper(), log_name=log_name)
        flashy.distrib.init()  # Initialize distributed training
        
        # Get the solver
        solver = get_solver(cfg)
        
        # If cfg.show is set, display solver information
        if cfg.show:
            solver.show()
            return
        
        # If executing a specific stage, restore checkpoint and run one stage
        if cfg.execute_only:
            assert cfg.execute_inplace or cfg.continue_from is not None, \
                "Please specify the checkpoint to continue from with continue_from=<sig_or_path> " + \
                "or set execute_inplace to True."
            solver.restore(replay_metrics=False)
            solver.run_one_stage(cfg.execute_only)
            return
        
        # Run the full training pipeline
        return solver.run()
    except hydra.errors.MissingConfigException as e:
        print(f"Error: {e}")
        print("Make sure that the referenced config files are present and accessible.")
        sys.exit(1)

if __name__ == '__main__':
    main()
