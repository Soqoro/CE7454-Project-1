import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='parsenet', choices=['parsenet'])
    parser.add_argument('--imsize', type=int, default=32)
    parser.add_argument('--version', type=str, default='parsenet')

    # Training setting
    parser.add_argument('--total_step', type=int, default=1000, help='how many times to update the generator')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--g_lr', type=float, default=0.0002)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--seed', type=int, default=0)

    # Testing setting
    parser.add_argument('--test_size', type=int, default=2824)
    parser.add_argument('--model_name', type=str, default='model.pth')

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--img_path', type=str, default='../dev_data/train/images')
    parser.add_argument('--label_path', type=str, default='../dev_data/train/masks')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')

    # Validation paths (for sweep & local model selection)
    parser.add_argument('--val_image_path', type=str, default='../dev_data/val/images')
    parser.add_argument('--val_label_path', type=str, default='../dev_data/val/masks')

    # Test paths
    parser.add_argument('--test_image_path', type=str, default='../dev_data/test/images')
    parser.add_argument('--test_label_path', type=str, default='./test_results')
    parser.add_argument('--test_color_label_path', type=str, default='./test_color_visualize')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=1.0)

    # --------- Sweep controls (comma-separated lists) ----------
    parser.add_argument('--sweep', type=str2bool, default=False)
    parser.add_argument('--sweep_g_lr', type=str, default='')         # e.g. "0.0002,0.0003,0.0005"
    parser.add_argument('--sweep_batch_size', type=str, default='')   # e.g. "4,8"
    parser.add_argument('--sweep_beta1', type=str, default='')        # e.g. "0.5,0.9"
    parser.add_argument('--sweep_lr_decay', type=str, default='')     # e.g. "0.95,0.98"
    parser.add_argument('--sweep_seed', type=str, default='')         # e.g. "0,1"

    return parser.parse_args()
