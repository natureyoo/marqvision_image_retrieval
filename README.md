# marqvision_image_retrieval


## Install necessary libraries
sh install.sh


## Run demo code
python src/demo_inference.py -db {gallery directory path} -q {query image path} -sm {similarity model file} -r {similarity result save path}

ex. python src/demo_inference.py -db '/DeepFashion2/validation/image' -q '/DeepFashion2/validation/image/000001.jpg' -sm './model/sim_model.pth' -r './result'
