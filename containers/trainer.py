import os
from tqdm import tqdm
import numpy as np
from .utils import *
class Trainer(object):
    def __init__(self,
        args,
        cfg,
        model,
        model_fn,
        optimizer,
        checkpoint_name="ckpt",
        best_name="best",
        lr_scheduler=None,
        bnm_scheduler=None,
        viz=None,
        writer=None):
        self.args = args
        self.cfg = cfg

        self.model = model
        self.model_fn = model_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.bnm_scheduler = bnm_scheduler

        self.checkpoint_name = checkpoint_name
        self.best_name = best_name

        self.training_best, self.eval_best = {}, {}
        self.viz = viz

        self._writer = writer
    def train(
        self,
        start_it,
        args,
        start_epoch,
        n_epochs,
        train_loader,
        train_sampler,
        test_loader=None,
        best_loss=0.0,
        log_epoch_f=None,
        tot_iter=1,
        clr_div=6,
    ):
        r"""
           Call to begin training the model

        Parameters
        ----------
        start_epoch : int
            Epoch to start at
        n_epochs : int
            Number of epochs to train for
        test_loader : torch.utils.data.DataLoader
            DataLoader of the test_data
        train_loader : torch.utils.data.DataLoader
            DataLoader of training data
        best_loss : float
            Testing loss of the best model
        """

        print("Totally train %d iters per gpu." % tot_iter)

        it = start_it
        #_, eval_frequency = is_to_eval(0, it)
        cfg = self.cfg
        lr = get_lr(self.optimizer)
        print()
        print(f"current lr:{lr}")
        print(cfg.n_total_epoch)
        print("start training     ++++++++++++++++++++++++")
        with tqdm(range(cfg.n_total_epoch), desc="%s_epochs" % cfg.cls_type) as \
                tbar, tqdm(1, leave=False, desc="train") as pbar:

            for epoch in tbar:
                if epoch > cfg.n_total_epoch:
                    break
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                # Reset numpy seed.
                # REF: https://github.com/pytorch/pytorch/issues/5059
                np.random.seed()
                if log_epoch_f is not None:
                    os.system("echo {} > {}".format(epoch, log_epoch_f))
                lr = get_lr(self.optimizer)
                print()
                print(f"current lr:{lr}")
                running_loss =0.0
                n = len(train_loader)
                
                for batch in train_loader:
                    self.model.train()
                    self.optimizer.zero_grad()

                    
                    _, loss, res = self.model_fn(self.model, batch, it=it)

                    lr = get_lr(self.optimizer)
                    running_loss += loss.item()/n
                    if args.local_rank == 0 and self._writer:
                        self._writer.write({"train/lr": lr}, it)
                    #if (it+1) % 16 ==0:
                    self.optimizer.step()
                    #self.optimizer.zero_grad()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    if self.bnm_scheduler is not None:
                        self.bnm_scheduler.step(it)

                    it += 1

                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.refresh()

                    if self.viz is not None:
                        self.viz.update("train", it, res)

                    #eval_flag, eval_frequency = is_to_eval(epoch, it)
                    #if eval_flag:
                #self.lr_scheduler.step()    
                print(f"training loss:{running_loss}\n")
                if self._writer is not None:
                    self._writer.write({"train/running loss":running_loss},it)

                if test_loader is not None:
                    #fix evaluation
                    val_loss, res = self.eval_epoch(test_loader, it=it, epoch=epoch)
                    print("val_loss", val_loss)

                    is_best = val_loss < best_loss
                    best_loss = min(best_loss, val_loss)
                    if args.local_rank == 0:
                        save_checkpoint(
                            checkpoint_state(
                                self.model, self.optimizer, val_loss, epoch, it
                            ),
                            is_best,
                            filename=self.checkpoint_name,
                            bestname=self.best_name + f'_{epoch}',
                            bestname_pure=self.best_name
                        )
                        info_p = self.checkpoint_name.replace(
                            '.pth.tar', '_epoch.txt'
                        )
                        os.system(
                            'echo {} {} >> {}'.format(
                                it, val_loss, info_p
                            )
                        )

        return best_loss
    def eval_epoch(self, d_loader, is_test=False, test_pose=False, it=None, epoch=0):
        self.model.eval()
        np.random.seed(2333)
        eval_dict = {}
        total_loss = 0.0
        count = 1
        log_iters = len(d_loader) // self.cfg.wandb.log_img_per_eval
        for i, data in enumerate(tqdm.tqdm(d_loader, leave=False, desc="val")):
            count += 1
            self.optimizer.zero_grad()

            if self.cfg.wandb.enable and self.args.local_rank == 0 \
                    and self.cfg.wandb.log_imgs \
                    and (i % log_iters) == 0:
                log_wandb = True
            else:
                log_wandb = False

            _, loss, eval_res = self.model_fn(
                self.model,
                data,
                it=it,
                epoch=epoch,
                is_eval=True,
                is_test=is_test,
                test_pose=test_pose,
                log_wandb=log_wandb
            )

            if 'loss_target' in eval_res.keys():
                total_loss += eval_res['loss_target']
            else:
                total_loss += loss.item()
            for k, v in eval_res.items():
                if v is not None:
                    eval_dict[k] = eval_dict.get(k, []) + [v]
        mean_eval_dict = {}
        for k,v in eval_dict.items():
            mean_eval_dict[k] = np.mean(v)
        for k, v in mean_eval_dict.items():
            print(k, v)

        if is_test:
            mean_eval_dict['rot_diff_std'] =  np.std(eval_dict['rot_diff'])
            mean_eval_dict['trans_diff_std'] =  np.std(eval_dict['trans_diff'])
            self._writer.write({"{}/{}".format(
                "test" if is_test else "val", k): v for k, v in mean_eval_dict.items()})
        else:
            if self.args.local_rank == 0 and self._writer:
                self._writer.write({"{}/{}".format(
                "test" if is_test else "val", k): v for k, v in mean_eval_dict.items()},
                iter=it)
        return total_loss / count, eval_dict
class Trainer_ref(Trainer):
    def __init__(self,
        args,
        cfg,
        model,
        model_fn,
        optimizer,
        checkpoint_name="ckpt",
        best_name="best",
        lr_scheduler=None,
        bnm_scheduler=None,
        viz=None,
        writer=None):
        super(Trainer_ref, self).__init__(args,
            cfg,
            model,
            model_fn,
            optimizer,
            checkpoint_name,
            best_name,
            lr_scheduler,
            bnm_scheduler,
            viz,
            writer)
    def train(
        self,
        start_it,
        start_epoch,
        n_epochs,
        train_loader,
        train_sampler,
        test_loader=None,
        best_loss=0.0,
        log_epoch_f=None,
        tot_iter=1,
        clr_div=6,
    ):
        r"""
           Call to begin training the model

        Parameters
        ----------
        start_epoch : int
            Epoch to start at
        n_epochs : int
            Number of epochs to train for
        test_loader : torch.utils.data.DataLoader
            DataLoader of the test_data
        train_loader : torch.utils.data.DataLoader
            DataLoader of training data
        best_loss : float
            Testing loss of the best model
        """

        print("Totally train %d iters per gpu." % tot_iter)

        def is_to_eval(epoch: int, it: int):
            """
            Defining the evaluation frequecies during training at a given
            epoch and iteration for CyclicLR
            Returns:
                boolean to_eval: if current iteration has to be evaluated
                int eval_frequency: new evaluation frequency at this iteration
            """
            # always do a first evaluation at 100 iterations
            if it == 100:
                return True, 1
            wid = tot_iter // clr_div
            # frequency at rising LR
            if (it // wid) % 2 == 1:
                eval_frequency = wid // self.cfg.evals_per_clr_down
            # frequency at declining LR
            else:
                eval_frequency = wid // self.cfg.evals_per_clr_up
            to_eval = (it % eval_frequency) == 0
            return to_eval, eval_frequency

        it = start_it
        #_, eval_frequency = is_to_eval(0, it)
        cfg = self.cfg
        lr = get_lr(self.optimizer)
        print()
        print(f"current lr:{lr}")
        print(cfg.n_total_epoch)
        print("start training     ++++++++++++++++++++++++")
        with tqdm(range(cfg.n_total_epoch), desc="%s_epochs" % cfg.cls_type) as \
                tbar, tqdm(1, leave=False, desc="train") as pbar:

            for epoch in tbar:
                if epoch > cfg.n_total_epoch:
                    break
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                # Reset numpy seed.
                # REF: https://github.com/pytorch/pytorch/issues/5059
                np.random.seed()
                if log_epoch_f is not None:
                    os.system("echo {} > {}".format(epoch, log_epoch_f))
                lr = get_lr(self.optimizer)
                print()
                print(f"current lr:{lr}")
                running_loss =0.0
                n = len(train_loader)
                
                for batch in train_loader:
                    self.model.train()
                    self.optimizer.zero_grad()

                    
                    _, loss, res = self.model_fn(self.model, batch, it=it)

                    lr = get_lr(self.optimizer)
                    running_loss += loss.item()/n
                    if self.args.local_rank == 0 and self._writer:
                        self._writer.write({"train/lr": lr}, it)
                    #if (it+1) % 16 ==0:
                    self.optimizer.step()
                    #self.optimizer.zero_grad()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    if self.bnm_scheduler is not None:
                        self.bnm_scheduler.step(it)

                    it += 1

                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.refresh()

                    if self.viz is not None:
                        self.viz.update("train", it, res)

                    #eval_flag, eval_frequency = is_to_eval(epoch, it)
                    #if eval_flag:
                #self.lr_scheduler.step()    
                print(f"training loss:{running_loss}\n")
                if self._writer is not None:
                    self._writer.write({"train/running loss":running_loss},it)

                if test_loader is not None:
                    #fix evaluation
                    val_loss, res = self.eval_epoch(test_loader, it=it, epoch=epoch)
                    print("val_loss", val_loss)

                    is_best = val_loss < best_loss
                    best_loss = min(best_loss, val_loss)
                    if self.args.local_rank == 0:
                        save_checkpoint(
                            checkpoint_state(
                                self.model, self.optimizer, val_loss, epoch, it
                            ),
                            is_best,
                            filename=self.checkpoint_name,
                            bestname=self.best_name + f'_{epoch}',
                            bestname_pure=self.best_name
                        )
                        info_p = self.checkpoint_name.replace(
                            '.pth.tar', '_epoch.txt'
                        )
                        os.system(
                            'echo {} {} >> {}'.format(
                                it, val_loss, info_p
                            )
                        )

        return best_loss
    def eval_epoch(self, d_loader, is_test=False, test_pose=False, it=None, epoch=0):
        self.model.eval()
        np.random.seed(1111)
        eval_dict = {}
        total_loss = 0.0
        count = 0
        log_iters = len(d_loader) // self.cfg.wandb.log_img_per_eval
        for i, data in enumerate(tqdm(d_loader, leave=False, desc="val")):
            count += 1
            self.optimizer.zero_grad()

            if self.cfg.wandb.enable and self.args.local_rank == 0 \
                    and self.cfg.wandb.log_imgs \
                    and (i % log_iters) == 0:
                log_wandb = True
            else:
                log_wandb = False

            _, loss, eval_res = self.model_fn(
                self.model,
                data,
                it=it,
                epoch=epoch,
                is_eval=True,
                is_test=is_test,
                test_pose=test_pose,
                log_wandb=log_wandb
            )

            if 'loss_target' in eval_res.keys():
                total_loss += eval_res['loss_target']
            else:
                total_loss += loss.item()
            for k, v in eval_res.items():
                if v is not None:
                    eval_dict[k] = eval_dict.get(k, []) + [v]
        mean_eval_dict = {}
        for k,v in eval_dict.items():
            mean_eval_dict[k] = np.mean(v)
        for k, v in mean_eval_dict.items():
            print(k, v)
        if is_test:
            mean_eval_dict['rot_diff_std'] =  np.std(eval_dict['rot_diff'])
            mean_eval_dict['trans_diff_std'] =  np.std(eval_dict['trans_diff'])
            self._writer.write({"{}/{}".format(
                "test" if is_test else "val", k): v for k, v in mean_eval_dict.items()})
        else:
            if self.args.local_rank == 0 and self._writer:
                self._writer.write({"{}/{}".format(
                "test" if is_test else "val", k): v for k, v in mean_eval_dict.items()},
                iter=it)
        return total_loss / count, eval_dict