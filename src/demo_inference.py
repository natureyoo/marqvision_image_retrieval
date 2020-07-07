from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from networks import ResNetbasedNet
from datasets import GalleryDataset, trivial_batch_collator

import argparse
import cv2
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def transform_for_sim(inputs, bboxes, need_resize):
    bboxes = [bb.proposal_boxes.tensor[0].int() for bb in bboxes]
    images = [input_data['image'].unsqueeze(0).double() for input_data in inputs]
    if need_resize:
        sizes = [(input_data['height'], input_data['width']) for input_data in inputs]
        images = [F.interpolate(im, size=sz) for (im, sz) in zip(images, sizes)]
    images = [im[:, [2, 1, 0], bb[1].item():bb[3].item(), bb[0].item():bb[2].item()] for (im, bb) in
              zip(images, bboxes)]
    images = [normalize(F.interpolate(im, size=(224, 224)).squeeze(0)) for im in images]
    images = torch.stack(images)
    return images


def get_query_img(img_path):
    query = {}
    query_img = cv2.imread(img_path)
    query['image'] = torch.from_numpy(query_img).permute(2, 0, 1)
    return [query]


def main(args):
    cfg = get_cfg()
    cfg.merge_from_file("./configs/mask_rcnn_R_101_FPN_3x.yaml")

    batch_size = cfg.SOLVER.IMS_PER_BATCH
    params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 8, 'collate_fn': trivial_batch_collator}

    query_path = args.query_path
    gallery_dir = args.gallery_dir

    # Load gallery dataset
    dataset = GalleryDataset(gallery_dir)
    data_loader = torch.utils.data.DataLoader(dataset, **params)
    need_resize = False

    # Load Detection model
    detection_model = build_model(cfg)
    checkpointer = DetectionCheckpointer(detection_model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # Load Sim model
    sim_model = ResNetbasedNet()
    sim_model.load_state_dict(torch.load(args.sim_model_file))
    sim_model = sim_model.double()

    is_cud = torch.cuda.is_available()
    device = torch.device('cuda' if is_cud else 'cpu')

    if is_cud:
        if torch.cuda.device_count() > 1:
            detection_model = nn.DataParallel(detection_model)
            sim_model = nn.DataParallel(sim_model)
        detection_model.to(device)
        sim_model.to(device)
        detection_model.eval()
        sim_model.eval()

    with torch.no_grad():
        query = get_query_img(query_path)
        bbox = detection_model.module.make_proposal(query, False, False)
        image = transform_for_sim(query, bbox, False)
        query_vec = sim_model(image)[0]

        start_time = time.time()
        gallery_vecs = torch.zeros((len(data_loader), 128))
        for idx, batched_inputs in enumerate(data_loader):
            bboxes = detection_model.module.make_proposal(batched_inputs, False, need_resize)
            # Crop the image with proposed bounding box for sim model
            images = transform_for_sim(batched_inputs, bboxes, need_resize)
            # Fed cropped image into the similarity model
            sim_vecs = sim_model(images)
            gallery_vecs[batch_size * idx: batch_size * (idx + 1)] = sim_vecs
            if idx % 100 == 0:
                print('processing {}/{}... elapsed time {}s'.format(idx+1, len(data_loader), time.time() - start_time))
            print('Converting the gallery images into the vectors is finished!')

        distance = ((query_vec - gallery_vecs) ** 2).sum(axis=1) ** 0.5
        similar_set = torch.argsort(distance)[:args.topk]
        similar_dist = distance[similar_set]

    save_dir = args.result_dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir, '{}.txt'.format(query_path.split('/')[-1].split('.')[0])), 'w') as f:
        for i, d in zip(similar_set, similar_dist):
            f.write('{},{:.3f}'.format(dataset.list_ids[i],d))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image retrieval system')
    parser.add_argument('-db', dest='gallery_dir', required=False, help='Directory of Gallery Images')
    parser.add_argument('-q', dest='query_path', required=True, help='File Path of Query Image')
    parser.add_argument('-k', dest='topk', required=False, default=100, help='Has GT Bounding Box or Not')
    parser.add_argument('-sm', dest='sim_model_file', required=False, help='Similarity Compairing Model File')
    parser.add_argument('-r', dest='result_dir', required=True, help='Directory to save the results')

    args = parser.parse_args()

    main(args)
