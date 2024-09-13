# import numpy as np
import torch as th
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.optim import Adam
from tqdm import tqdm

from config import ConfigFactory
from dataset import get_loaders
from logger import TrainLogger, TrainLoggerFactory
# from models.pqformer import WrapPQFormer
# from baselines import get_baselines
from time import time

# th.autograd.set_detect_anomaly(True)


def train_val(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    logger: TrainLogger,
    device,
    args,
):
    LR = [args.lr]
    EPOCHS = args.epochs

    STR_EARLY = False
    EARLY = args.early_stop
    patience = args.patience + 1
    optimizer = Adam(model.parameters(), lr=LR[0], weight_decay=5e-5)
    loss_fn = nn.SmoothL1Loss()
    model = model.train()
    best_criteria = -1 * th.inf
    END = False
    for epoch in range(EPOCHS):
        if END:
            break
        # if (best_criteria > 0) and (epoch < EARLY):
        #     LR[0] = 5e-4
        if (epoch >= EARLY) and (not STR_EARLY):
            logger.info("\n" + "-" * 20 + "\nStart early stopping\n" +
                        "-" * 20)
            STR_EARLY = True
            LR[0] = 1e-3
        running_loss = 0.0
        train_str_time = time()
        with tqdm(train_loader) as pbar:
            model = model.train()
            for i, batch_data in enumerate(pbar):
                x, y = batch_data
                if args.model != "pqformer":
                    x = x.permute(0, 2, 1).to(th.float32)
                    y = y.permute(0, 2, 1).to(th.float32)
                    x = x.to(device)
                    y = y.to(device)
                    with autocast():
                        # Location Embedding
                        x_mark = None
                        y_mark = None
                        # decoder
                        # dec_inp = th.zeros_like(y[:,
                        #                           -args.out_step:, :]).float()
                        # dec_inp = th.cat([x[:, -args.in_step:, :], dec_inp],
                        #                  dim=1).float().to(device)
                        dec_inp = x[:, -args.in_step:, :].float().to(device)
                        y_pred = model(x, x_mark, dec_inp, y_mark)
                        # Sanity Check
                        y_pred = y_pred[:, -args.out_step:, :]
                        y = y[:, -args.out_step:, :].to(device)
                        loss = loss_fn(y_pred, y.to(device))
                else:
                    with autocast():
                        y_pred = model(x)
                        loss = loss_fn(y_pred, y.to(device))
                        if th.isnan(loss):
                            logger.error(
                                "Training loss is NaN. Terminate training.")
                            END = True
                            break

                optimizer.zero_grad()
                loss.backward()
                # Gradient Check
                # for name, param in model.named_parameters():
                #     if param.requires_grad and param.grad is not None:
                #         if th.isnan(param.grad).any():
                #             print(f"Layer {name} has NaN gradients.")
                #         elif th.isinf(param.grad).any():
                #             print(f"Layer {name} has infinite gradients.")
                #         else:
                #             print(f"Layer {name} gradient stats: Mean={param.grad.mean()}, Max={param.grad.max()}, Min={param.grad.min()}")
                # Gradient clip
                # if args.model in ["Informer", "FEDformer"]:
                #     th.nn.utils.clip_grad_norm_(model.parameters(),
                #                                 max_norm=1.0)
                optimizer.step()

                running_loss = (running_loss * i + loss.item()) / (i + 1)
                pbar.set_description(
                    f"Epoch:{epoch}, train loss:{loss.item()}, avg loss:{running_loss}"
                )
        train_end_time = time()
        # Add memory usage record
        memory = th.cuda.memory_reserved(device)
        logger.info(f"Memory: {memory / 1024**2}")
        logger.info(f"Epoch: {epoch}")
        logger.info(f"Train loss: {running_loss}")
        logger.info(f"Train time: {train_end_time - train_str_time}s")

        val_str_time = time()
        val_mae, val_rmse, val_mape, val_r2 = evaluation(model,
                                                         val_loader,
                                                         device,
                                                         args=args)
        val_end_time = time()
        logger.info(
            f"Val MAE-RMSE-MAPE-R2:\n{val_mae},{val_rmse},{val_mape},{val_r2}")
        logger.info(f"Val time: {val_end_time - val_str_time}s")

        # test_mae, test_rmse, test_mape, test_r2 = evaluation(model,
        #                                                      test_loader,
        #                                                      device,
        #                                                      args=args)
        # logger.info(
        #     f"Test MAE-RMSE-MAPE-R2: {test_mae},{test_rmse},{test_mape},{test_r2}"
        # )

        if val_r2 >= best_criteria:
            best_criteria = val_r2
            patience = args.patience + 1
            logger.info("Save Model Parameters...")
            logger.save_model_parameters(model)
        else:
            if epoch >= EARLY:
                patience = patience - 1
                if patience > 0:
                    logger.info(
                        f"Patience of early stopping: {patience}/{args.patience}"
                    )
                else:
                    logger.info("Early stopping. Training process completed.")
                    break
        logger.info("")
    logger.info("Training process completed.")


def evaluation(model: nn.Module, loader, device, args):
    run_mae = 0.0
    run_rmse = 0.0
    run_mape = 0.0
    run_r_square = 0.0
    model = model.eval()
    with th.no_grad():
        with tqdm(loader) as pbar:
            for i, batch_data in enumerate(pbar):
                x, y = batch_data
                y = y.to(device)
                if args.model != "pqformer":
                    x = x.permute(0, 2, 1)
                    y = y.permute(0, 2, 1)
                    x = x.to(device)
                    with autocast():
                        # Location Embedding
                        x_mark = None
                        y_mark = None
                        # decoder
                        dec_inp = th.zeros_like(y[:,
                                                  -args.out_step:, :]).float()
                        dec_inp = th.cat([x[:, -args.in_step:, :], dec_inp],
                                         dim=1).float().to(device)
                        y_pred = model(x, x_mark, dec_inp, y_mark)
                        # Sanity Check
                        y_pred = y_pred[:, -args.out_step:, :]
                else:
                    with autocast():
                        y_pred = model(x)
                error = y_pred - y
                error_abs = th.abs(error)
                error_square = th.square(error)
                mae = error_abs.mean()
                rmse = error_square.mean().sqrt()
                mape = (error_abs / (th.abs(y) + 1e-5)).mean()
                r_square = 1 - error_square.sum() / th.square(y).sum()

                run_mae = (run_mae * i + mae) / (i + 1)
                run_rmse = (run_rmse * i + rmse) / (i + 1)
                run_mape = (run_mape * i + mape) / (i + 1)
                run_r_square = (run_r_square * i + r_square) / (i + 1)
                desc = "MAE-RMSE-MAPE-R2: ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}), ".format(
                    a=mae.item(),
                    b=rmse.item(),
                    c=mape.item(),
                    d=r_square.item())
                desc += "mean MAE-RMSE-R2: ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f})".format(
                    a=run_mae.item(),
                    b=run_rmse.item(),
                    c=run_mape.item(),
                    d=run_r_square.item())
                pbar.set_description(desc)
    return run_mae, run_rmse, run_mape, run_r_square


def main():
    args, msg = ConfigFactory.build()
    logger = TrainLoggerFactory.build(args.log_name)
    logger.info(msg)
    logger.info(f"Device num: {th.cuda.device_count()}")
    if hasattr(args, 'device'):
        device_id = args.device
    else:
        device_id = 0
    DEVICE = f"cuda:{device_id}" if th.cuda.is_available() else "cpu"
    logger.info(f"Device: {DEVICE}")

    IN_STEP, OUT_STEP, MODEL_NAME = args.in_step, args.out_step, args.model
    X_train_np, train_loader, val_loader, test_loader, CHN = get_loaders(args)
    print("Data Loader Finish!")
    if MODEL_NAME == "pqformer":
        model = WrapPQFormer(
            args,
            CHN,
            DEVICE,
        )
        logger.info("Train Product Quantization...")
        pq_str_time = time()
        model.train_index(
            X_train_np.reshape(-1, IN_STEP,
                               CHN).transpose(1, 0, 2).reshape(IN_STEP, -1).T)
        pq_end_time = time()
        logger.info(f"Initialization time: {pq_end_time - pq_str_time}s")
    else:
        base_args = args
        base_args.seq_len = IN_STEP
        base_args.pred_len = OUT_STEP
        base_args.label_len = IN_STEP
        base_args.enc_in = CHN
        base_args.dec_in = CHN
        base_args.c_out = CHN
        model = get_baselines(base_args)
    model.to(DEVICE)

    train_val(model, train_loader, val_loader, test_loader, logger, DEVICE,
              args)
    model.load_state_dict(logger.load_model_parameters())
    test_mae, test_rmse, test_mape, test_r2 = evaluation(model,
                                                         test_loader,
                                                         DEVICE,
                                                         args=args)
    logger.info(
        f"Test MAE-RMSE-MAPE-R2: {test_mae},{test_rmse},{test_mape},{test_r2}")
    memory = th.cuda.memory_reserved(DEVICE)
    logger.info(f"Memory: {memory / 1024**2}")


if __name__ == "__main__":
    main()
