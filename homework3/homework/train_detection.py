import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.utils.tensorboard as tb
from .models import load_model, save_model
from .datasets.road_dataset import load_data
import torch.nn.functional as F
from .metrics import AccuracyMetric, DetectionMetric  # Import the metric classes

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs).to(device)
    print(model)
    model.train()

    # Use an absolute path for the dataset
    train_data_path = "drive_data/train"
    val_data_path = "drive_data/val"

    # Print the paths to verify
    print(f"Train data path: {train_data_path}")
    print(f"Validation data path: {val_data_path}")

    train_data = load_data(train_data_path, shuffle=True, batch_size=batch_size, num_workers=5)
    val_data = load_data(val_data_path, shuffle=False, num_workers=5)
    print(f"Train_data: {len(train_data.dataset)}, Val_data: {len(val_data.dataset)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_seg = torch.nn.CrossEntropyLoss()
    criterion_depth = torch.nn.MSELoss()

    global_step = 0
    accuracy_metric = AccuracyMetric()  # Initialize accuracy metric
    detection_metric = DetectionMetric(num_classes=3)  # Initialize detection metric

 # training loop
    for epoch in range(num_epoch):
        accuracy_metric.reset()
        detection_metric.reset()
        model.train()
        total_loss = 0
        for batch in train_data:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            images = batch["image"]  # Shape: (B, 3, H, W)
            depths = batch["depth"]  # Shape: (B, 1, H, W)
            tracks = batch["track"]  # Shape: (B, H, W) or (B, C, H, W)

            # print(f"{images.shape = }")
            # print(f"{tracks.shape = }")
            # print(f"{depths.shape = }")
            seg_logits, depth_preds = model(images)
            # print(f"{seg_logits.shape = }")
            # print(f"{depth_preds.shape = }")

            # # # Resize tracks to match the shape of seg_logits
            tracks_resized = F.interpolate(tracks.unsqueeze(1), size=seg_logits.shape[2:], mode='nearest').squeeze(1)

            # # Resize depths to match the shape of depth_preds
            # depths_resized = F.interpolate(depths.unsqueeze(1), size=depth_preds.shape[2:], mode='nearest').squeeze(1)
            depth_preds = depth_preds.squeeze(1)  # Squeeze the singleton dimension

            loss_seg = criterion_seg(seg_logits, tracks_resized)
            loss_depth = criterion_depth(depth_preds, depths)
            loss = loss_seg + loss_depth
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Convert segmentation predictions to class indices
            seg_preds = seg_logits.argmax(dim=1)

            # Update metrics
            accuracy_metric.add(seg_preds, tracks)
            detection_metric.add(seg_preds, tracks, depth_preds, depths)

            logger.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        # Compute and log training metrics
        train_metrics = detection_metric.compute()
        train_accuracy = accuracy_metric.compute()["accuracy"]

        logger.add_scalar("train_accuracy", train_accuracy, epoch)
        logger.add_scalar("train_iou", train_metrics["iou"], epoch)
        logger.add_scalar("train_depth_error", train_metrics["abs_depth_error"], epoch)



        # print(f"Epoch {epoch}: Train Accuracy: {train_accuracy:.4f}, IoU: {train_metrics['iou']:.4f}, Depth Error: {train_metrics['abs_depth_error']:.4f}")
 

        # Validation loop
        with torch.no_grad():
            model.eval()
            accuracy_metric.reset()
            detection_metric.reset()

            for batch in val_data:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                images = batch["image"]
                depths = batch["depth"]
                tracks = batch["track"]

                seg_logits, depth_preds = model(images)
                seg_preds = seg_logits.argmax(dim=1)


                # Resize tracks and depths
                # tracks_resized = F.interpolate(tracks.unsqueeze(1), size=seg_logits.shape[2:], mode='nearest').squeeze(1)
                # depths_resized = F.interpolate(depths.unsqueeze(1), size=depth_preds.shape[2:], mode='nearest').squeeze(1)

                # Update metrics
                accuracy_metric.add(seg_preds, tracks)
                detection_metric.add(seg_preds, tracks, depth_preds, depths)

            # Compute and log validation metrics
            val_metrics = detection_metric.compute()
            val_accuracy = accuracy_metric.compute()["accuracy"]
            val_loss_seg = criterion_seg(seg_logits, tracks)
            val_loss_depth = criterion_depth(depth_preds.squeeze(1), depths)
            val_loss = val_loss_seg + val_loss_depth
            logger.add_scalar("val_loss", val_loss.item(), epoch)

            logger.add_scalar("val_accuracy", val_accuracy, epoch)
            logger.add_scalar("val_iou", val_metrics["iou"], epoch)
            logger.add_scalar("val_depth_error", val_metrics["abs_depth_error"], epoch)
            

            
        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:2d} / {num_epoch:2d}: " f"train_acc={train_accuracy:.4f} " f"val_acc={val_accuracy:.4f}" )
            print(f"Epoch {epoch}: Val Accuracy: {val_accuracy:.4f}, IoU: {val_metrics['iou']:.4f}, Depth Error: {val_metrics['abs_depth_error']:.4f}")

            # print(f"Epoch {epoch}: Val Accuracy: {val_accuracy:.4f}, IoU: {val_metrics['iou']:.4f}, Depth Error: {val_metrics['abs_depth_error']:.4f}")
                # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))