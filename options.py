import argparse
import os
import util


class TrainOptions:
    def __init__(self):
        self.opt = None
        self.parser = None
        self.initialized = False

    def initialize(self, parser):
        # data augment ----------------------------------------------------------------------
        parser.add_argument('--down_prob', type=float, default=0.1)
        parser.add_argument('--down_ratio', type=list, default=[0.25, 0.5])
        parser.add_argument('--blur_prob', type=float, default=0.1)
        parser.add_argument('--blur_sig', default='0.0,3.0')
        parser.add_argument('--jpg_prob', type=float, default=0.1)
        parser.add_argument('--jpg_method', default='cv2,pil')
        parser.add_argument('--jpg_qual', default='30,100')
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--data_aug', type=bool, default=True, help='use data augmentation, gaussian_blur, jpeg, '
                                                                        'down_sample')
        parser.add_argument('--cropSize', type=int, default=224, help='Scale images to this size')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        # basic options ----------------------------------------------------------------------
        parser.add_argument('--name', type=str, default='Experiment_name',
                            help='Name of the experiment which decides where to store samples and models')
        parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
        parser.add_argument('--classes',
                            default='airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,'
                                    'pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse',
                            help='image classes to train on')
        parser.add_argument('--loadSize', type=int, default=256, help='Scale images to this size')
        parser.add_argument('--patch_num', type=int, default=3, help='Scale images to this size, used in preprocess')
        parser.add_argument('--dataroot', default='D:/dataset/local_val/',
                            help='path to images, should include folders: train, val')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--beta1', type=float, default=0.1, help='momentum term of adam')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--training_method', type=bool, default=False, help='Training network, RPTC with freq '
                                                                                'feature or not,'
                                                                                'True for use, False for not')

        parser.add_argument('--continue_train', type=bool, default=False)
        parser.add_argument('--earlystop_epoch', type=int, default=2, help='the patience of early stop')
        parser.add_argument('--niter', type=int, default=3, help='Total training epochs')
        parser.add_argument('--loss_freq', type=int, default=400, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_latest_freq', type=int, default=2500, help='frequency of saving the latest results')

        return parser

    def gather_options(self):
        # 以默认参数初始化选项
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # 解析命令行，获得选项
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_option=True):
        opt = self.gather_options()
        opt.isTrain = True
        opt.isVal = False
        opt.classes = opt.classes.split(',')

        # result dir, save results and opt
        opt.results_dir = f'./results/'
        util.mkdirs(opt.results_dir)

        if print_option:
            self.print_options(opt)

        # additional
        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt


class TestOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        # data augmentation
        parser.add_argument('--detect_method', type=str, default='RPTC', help='The detect method used in testing '
                                                                                    '[cnn, RPTC, RPTC with freq]')
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_sig', default='1.0')
        parser.add_argument('--jpg_method', default='pil')
        parser.add_argument('--jpg_qual', default='95')

        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--CropSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')

        parser.add_argument('--model_path', type=str, default='./weights/classifier/CNNSpot.pth',
                            help='the path of detection model')
        parser.add_argument('--training_method', type=bool, default=False, help='Training network, RPTC with freq '
                                                                                'feature or not,'
                                                                                'True for use, False for not')
        parser.add_argument('--noise_type', type=str, default=None, help='such as jpg, blur and resize')
        parser.add_argument('--pr_curve', type=bool, default=False, help='If specified, print and save pr_curve')

        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk

        file_name = os.path.join(opt.results_dir, f'{opt.noise_type}opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = False  # train or test
        opt.isVal = False

        # result dir, save results and opt
        opt.results_dir = f"./results/{opt.detect_method}"
        util.mkdir(opt.results_dir)

        if print_options:
            self.print_options(opt)

        # additional

        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt


if __name__ == '__main__':
    print("The default network for training is RPTC, if you want to use RPTC with freq feature "
          " you should set the trainig_method as True")
