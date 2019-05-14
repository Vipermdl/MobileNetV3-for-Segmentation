import os
import time
import pickle
import random

import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
#user
from model import MobileNetV3
from utils.utils import *
from utils.convert_state import convert_state_dict
import torchvision.transforms as transforms
from  dataset.cityscapes import CityscapesTestDataSet, CityscapesTrainInform  # dataset

color_encoding = OrderedDict([
            ('road', (128, 64, 128)),
            ('sidewalk', (244, 35, 232)),
            ('building', (70, 70, 70)),
            ('wall', (102, 102, 156)),
            ('fence', (190, 153, 153)),
            ('pole', (153, 153, 153)),
            ('traffic_light', (250, 170, 30)),
            ('traffic_sign', (220, 220, 0)),
            ('vegetation', (107, 142, 35)),
            ('terrain', (152, 251, 152)),
            ('sky', (70, 130, 180)),
            ('person', (220, 20, 60)),
            ('rider', (255, 0, 0)),
            ('car', (0, 0, 142)),
            ('truck', (0, 0, 70)),
            ('bus', (0, 60, 100)),
            ('train', (0, 80, 100)),
            ('motorcycle', (0, 0, 230)),
            ('bicycle', (119, 11, 32))
    ])

full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

# The values above are remapped to the following
new_classes = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)

label_to_rgb = transforms.Compose([
    LongTensorToRGBPIL(color_encoding),
    transforms.ToTensor()
])

def test(args, test_loader, model, device, data):
    """
    args:
      test_loader: loaded for test set
      model: model
      criterion: loss function
    return: IoU class, and mean IoU
    """
    #evaluation mode
    model.eval()
    total_batches = len(test_loader)

    submission_path = os.path.join(args.save_seg_dir, 'submission')
    visual_path = os.path.join(args.save_seg_dir, 'visual')

    if not os.path.exists(submission_path):
        os.makedirs(submission_path)
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    
    with torch.no_grad():
        for i, (input, size, name) in enumerate(test_loader):
            start_time = time.time()
            input_var = input.to(device)
            output_ = model(input_var)
            _, prediction = torch.max(output_.data, 1)
            tensor_ = prediction.clone()

            tensor_ = tensor_.to('cpu').byte()
            save_png = ToPILImage()(tensor_)
            save_png, prediction = remap(save_png, full_classes, new_classes)
            time_taken = time.time() - start_time
            print('[%d/%d]  time: %.2f' % (i, total_batches, time_taken))
            save_png.save(os.path.join(submission_path, name[0]+'.png'))

            if args.visual:
                image = input.squeeze().numpy().transpose(1,2,0) + data['mean']
                image = torch.from_numpy(image.transpose(2, 0, 1)).int()
                color_prediction =  batch_transform(torch.from_numpy(prediction).unsqueeze(dim=0).long(), label_to_rgb)
                
                imshow_batch(image, color_prediction, os.path.join(visual_path, name[0]+'.png'))

def test_func(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)
    global network_type

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

        device = 'cuda'
    
    args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed) 
    
    print('=====> checking if processed cached_data_file exists')
    if not os.path.isfile(args.inform_data_file):
        dataCollect = CityscapesTrainInform(args.data_dir, args.classes, train_set_file = args.dataset_list, 
                                            inform_data_file = args.inform_data_file) #collect mean std, weigth_class information
        data= dataCollect.collectDataAndSave()
        if data is  None:
            print("error while pickling data, please check")
            exit(-1)
    else:
        data = pickle.load(open(args.inform_data_file, "rb"))
    M = args.M
    N = args.N

    model = MobileNetV3(model_mode="SMALL", num_classes=args.classes)

    network_type = "MobileNetV3"
    print("Arch:  MobileNetV3")

    if args.cuda:
        model = model.to(device) # using GPU for inference
        cudnn.benchmark = True

    print('Dataset statistics')
    print('mean and std: ', data['mean'], data['std'])
    print('classWeights: ', data['classWeights'])

    # validation set
    testLoader = torch.utils.data.DataLoader(CityscapesTestDataSet(args.data_dir, args.test_data_list, mean = data['mean']),
                                             batch_size = 1, shuffle = False, num_workers = args.num_workers, pin_memory = True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=====> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #model.load_state_dict(checkpoint['model'])
            model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.resume))
    
    print("=====> beginning testing")
    print("test set length: ", len(testLoader))
    test(args, testLoader, model, device, data)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', default = "MobileNetV3", help = "model name: Context Guided Network (MobileNetV3)")
    parser.add_argument('--data_dir', default = "/home/madongliang/dataset/cityscape/", help = "data directory")
    parser.add_argument('--dataset_list', default = "cityscapes_trainval_list.txt",
                        help = "train and val data, for computing the ratio of all classes, mean and std")
    parser.add_argument('--test_data_list', default = "./dataset/list/Cityscapes/cityscapes_test_list.txt", help = "test set")
    parser.add_argument('--scaleIn', type = int, default = 1, help = "rescale input image, default is 1, keep fixed size")  
    parser.add_argument('--num_workers', type = int, default= 1, help = "the number of parallel threads") 
    parser.add_argument('--batch_size', type = int, default = 1, help=" the batch_size is set to 1 when evaluating or testing") 
    parser.add_argument('--resume', type = str, default = "./checkpoint/cityscapes/MobileNetV3bs16gpu2_ontrainval/model_343.pth",
                        help = "use the file to load last checkpoint for evaluating or testing ")
    parser.add_argument('--classes', type = int, default = 19, 
                        help = "the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--inform_data_file', default = "./dataset/wtfile/cityscapes_inform.pkl", 
                        help = "storing the classes weights, mean and std")
    parser.add_argument('--cuda', default = True, help = "run on CPU or GPU")
    parser.add_argument('--visual', default=True, help="to visual the results")
    parser.add_argument('--M', type = int, default = 3, help = "the number of blocks in stage 2")
    parser.add_argument('--N', type = int, default = 21, help = "the number of blocks in stage 3")
    parser.add_argument('--save_seg_dir', type = str, default = "./result/cityscapes/test/", help = "saving path of prediction result")
    parser.add_argument("--gpus", default = "7", type = str, help = "gpu ids (default: 2)")

    test_func(parser.parse_args())

