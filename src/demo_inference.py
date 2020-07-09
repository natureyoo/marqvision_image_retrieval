from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

from networks import ResNetbasedNet
from datasets import GalleryDataset, trivial_batch_collator

import argparse, time, os
import cv2
import numpy as np
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


def get_embed_vec(data_loader, detection_model, sim_model, data_num, vec_dim=128, need_resize=False):
    start_time = time.time()
    embedding_vecs = torch.zeros((data_num, vec_dim))
    idx_ = 0
    for idx, batched_inputs in enumerate(data_loader):
        bboxes = detection_model.module.make_proposal(batched_inputs, False, need_resize)
        # Crop the image with proposed bounding box for sim model
        images = transform_for_sim(batched_inputs, bboxes, need_resize)
        # Fed cropped image into the similarity model
        sim_vecs = sim_model(images)
        embedding_vecs[idx_: idx_ + len(batched_inputs)] = sim_vecs
        idx_ += len(batched_inputs)
        if idx % 100 == 0:
            print('processing {}/{}... elapsed time {:.1f}s'.format(idx + 1, len(data_loader), time.time() - start_time))
    return embedding_vecs.cpu().numpy()


def get_proposals(data_loader, detection_model, need_resize=False):
    start_time = time.time()
    img_set = {}
    for idx, batched_inputs in enumerate(data_loader):
        bboxes = detection_model.module.make_proposal(batched_inputs, False, need_resize)
        # Crop the image with proposed bounding box for sim model
        images = transform_for_sim(batched_inputs, bboxes, need_resize)
        for inp, img in zip(batched_inputs, images):
            img_set[inp['idx']] = img.cpu().numpy()
        if idx % 100 == 0:
            print('processing {}/{}... elapsed time {:.1f}s'.format(idx + 1, len(data_loader), time.time() - start_time)) 
    print('Finish predicting bounding boxes! Elapsed Time: {:.1f}s'.format(time.time() - start_time))
    del detection_model
    return img_set


def get_sim_vecs(data_loader, sim_model, data_num, device, vec_dim=128):
    start_time = time.time()
    embedding_vecs = torch.zeros((data_num, vec_dim))
    idx_ = 0
    for idx, batched_images in enumerate(data_loader):
        sim_vecs = sim_model(batched_images.to(device))
        embedding_vecs[idx_:idx_ + len(batched_images)] = sim_vecs
        idx_ += len(batched_images)
        if idx % 100 == 0:
            print('processing {}/{}... elapsed time {:.1f}s'.format(idx + 1, len(data_loader), time.time() - start_time))
    return embedding_vecs.cpu().numpy()


def calculate_distance(query_vec, gallery_vecs, gallery_file_list, top_k=100):
    distance = (((query_vec - gallery_vecs) ** 2).mean(axis=1) ** 0.5)
    similar_set = np.argsort(distance)[:top_k]
    distance = distance[similar_set]
    similar_file_list = [gallery_file_list[idx] for idx in similar_set]
    return similar_file_list, distance


def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join('/'.join(os.path.realpath(__file__).split('/')[:-2]), "configs/mask_rcnn_R_101_FPN_3x.yaml"))

    params = {'batch_size': cfg.SOLVER.IMS_PER_BATCH, 'shuffle': False, 'num_workers': 8, 'collate_fn': trivial_batch_collator}

    query_path = args.query_path
    gallery_dir = args.gallery_dir
    save_dir = args.result_dir
    is_precomputed = bool(args.is_precomputed)
    top_k = int(args.top_k)
    batch_size = int(args.batch_size)
    vec_dim = 128

    # Load Detection model
    detection_model = build_model(cfg)
    checkpointer = DetectionCheckpointer(detection_model)
    checkpointer.load(args.detection_model_file)

    # Load Sim model
    sim_model = ResNetbasedNet(cfg=cfg, vec_dim=vec_dim)
    sim_model.load_state_dict(torch.load(args.sim_model_file))
    sim_model = sim_model.double()

    is_cud = torch.cuda.is_available()
    # device = torch.device("cuda:1")
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
        query_vec = sim_model(image)[0].cpu().numpy()
        
        if is_precomputed:
            print('Load precomputed vectors for gallery!')
            gallery_vecs = np.load(os.path.join(gallery_dir, 'vecs.npy'))
            gallery_list = [f.strip() for f in open(os.path.join(gallery_dir, 'vecs_idx.txt'), 'r').readlines()]
        else:
            # Load gallery dataset
            dataset = GalleryDataset(gallery_dir=gallery_dir, task='detection')
            data_loader = torch.utils.data.DataLoader(dataset, **params)
            print('Gallery dataset is loaded!')
            need_resize = False

            gallery_list = dataset.list_ids
            # gallery_vecs = get_embed_vec(data_loader, detection_model, sim_model, len(gallery_list), vec_dim=vec_dim, need_resize=need_resize)
            dataset = get_proposals(data_loader, detection_model, need_resize=False)
            dataset = GalleryDataset(image_list=dataset, image_ids=gallery_list, task='similarity')
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=params['num_workers'])
            gallery_vecs = get_sim_vecs(data_loader, sim_model, len(gallery_list), device, vec_dim=128)
            print('Converting the gallery images into the vectors is finished!')
            
            np.save(os.path.join(gallery_dir, 'vecs.npy'), gallery_vecs)
        
            with open(os.path.join(gallery_dir, 'vecs_idx.txt'), 'w') as f:
                for item in gallery_list:
                    f.write('{}\n'.format(item))

    similar_files, distance = calculate_distance(query_vec, gallery_vecs, gallery_list, top_k=top_k)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    query_id = query_path.split('/')[-1].split('.')[0]
    with open(os.path.join(save_dir, '{}.txt'.format(query_id)), 'w') as f:
        for file_name, dist in zip(similar_files, distance):
            f.write('{},{:.3f}\n'.format(file_name, dist))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='image retrieval system')
    parser.add_argument('-db', dest='gallery_dir', required=True, help='Directory of Gallery Images')
    parser.add_argument('-q', dest='query_path', required=True, help='File Path of Query Image')
    parser.add_argument('-b', dest='batch_size', required=False, default=8, help='Batch Size, if the memori is lack, reduce the batch size')
    parser.add_argument('-k', dest='top_k', required=False, default=100, help='Has GT Bounding Box or Not')
    parser.add_argument('-dm', dest='detection_model_file', required=True, help='Detection Model File')
    parser.add_argument('-sm', dest='sim_model_file', required=True, help='Similarity Compairing Model File')
    parser.add_argument('-r', dest='result_dir', required=True, help='Directory to save the results')
    parser.add_argument('-c', dest='is_precomputed', required=False, default=False, help='Precomputed vectors exist or not')

    args = parser.parse_args()

    main(args)
