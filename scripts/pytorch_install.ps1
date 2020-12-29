$TORCH_VERSION = [string]$para1
if [ $TORCH_VERSION = "1.5.0" ]
then
    TORCHVISION_VERSION="0.6.0"
elif [ $TORCH_VERSION = "1.5.1" ]
then
    TORCHVISION_VERSION="0.6.1"
elif [ $TORCH_VERSION = "1.6.0" ]
then
    TORCHVISION_VERSION="0.7"
elif [ $TORCH_VERSION = "1.7.0" ]
then
    TORCHVISION_VERSION="0.8.1"
fi
pip install torch==${TORCH_VERSION}
pip install torchvision==${TORCHVISION_VERSION}
