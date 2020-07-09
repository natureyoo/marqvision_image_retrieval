# marqvision_image_retrieval



## Install necessary libraries

sh install.sh

#### Environment
python3.6

CUDA Version: 10.2




## Download model file
There are two models for object detection and similarity comparison respectively.

All models are being stored in S3 and you can use it after downloading and decompressing it.

In each S3 directory, there can be various versions for the models. You should download the appropriate version of the model.


#### Object detection model file
s3://marq-ai/model/object-detection/detection_model.pth.zip 

#### Similarity comparison model file
s3://marq-ai/model/similar-retrieval/similar_model.pth.zip



## Run demo code

python src/demo_inference.py -db {gallery directory path} -q {query image path} -dm {detection model file} -sm {similarity model file} -r {similarity result save path} -b {batch size fo similarity model} -c {whether to use the precomputed vectors of gallery images or not} -k {how many the most similar images to find}


#### Case1: Converting gallery images to the vectors and calculating the similarity with the query image

ex. python src/demo_inference.py -db '/DeepFashion2/validation/image' -q '/DeepFashion2/validation/image/000001.jpg' -dm 'home/{user}/model/detection_model.pth' -sm 'home/{user}/model/sim_model.pth' -b 16 -r './result'

#### Case2: Loading the precomputed vectors of gallery images and calculating the similarity with the query image
For about 32,000 gallery images and 1 query image, it takes about 33min with 4 RTX2080ti gpus. 

If we pool only 5000 images and use them as gallery images, it will take about 6min.


ex. python src/demo_inference.py -db '/DeepFashion2/validation/image' -q '/DeepFashion2/validation/image/000001.jpg' -dm 'home/{user}/model/detection_model.pth' -sm 'home/{user}/model/sim_model.pth' -b 16 -r './result' -c True


## Tips
If the gpu memory is lack, you can reduce the batch size with '-b' option.
