GitHub.

markdown
Copy code
# ISBNet Setup Using Docker

## 1. Docker Setup

### Pull the Docker Image

To pull the Docker image for ISBNet, run the following command:

```bash
docker pull rayyoh/oneformer3d
Clone ISBNet Repository
Clone the ISBNet repository using Git:

bash
Copy code
git clone https://github.com/VinAIResearch/ISBNet.git
Run Docker Container
Run the Docker container with the appropriate settings. Ensure that you have a GPU enabled for this setup.

bash
Copy code
docker run --gpus all -it --shm-size=8g --cpus=4 -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 -v C:\Users\Perception-Team\ISBNet:/workspace -p 6008:6008 --name isnbet check
Install Python Requirements Inside the Container
After entering the Docker container, install the required Python dependencies:

bash
Copy code
pip3 install -r requirements.txt
Build Segmentator Inside Docker
Clone the Segmentator repository and build it inside the Docker container:

bash
Copy code
git clone https://github.com/Karbo123/segmentator.git
cd segmentator/csrc
mkdir build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch; print(torch.utils.cmake_prefix_path)'` \
  -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
  -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'`
make && make install
Install Additional Dependencies
Install any additional dependencies required for the setup:

bash
Copy code
apt-get install libsparsehash-dev
Setup PointNet2
Navigate to the pointnet2 directory inside the ISBNet folder and install it:

bash
Copy code
cd isbnet/pointnet2
python3 setup.py bdist_wheel
cd ./dist
pip3 install pointnet2-0.0.0-cp310-cp310-linux_x86_64.whl
Final Build
To complete the final setup and build process, run the following commands:

bash
Copy code
cd ../../..
python3 setup.py build_ext develop
By following these steps, you should have ISBNet successfully set up in Docker and ready for use.
