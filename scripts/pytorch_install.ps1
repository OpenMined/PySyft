$TORCH_VERSION = $para1
If ($TORCH_VERSION -eq "1.5.0") {
  $TORCHVISION_VERSION="0.6.0"
} Elseif ( $TORCH_VERSION -eq "1.5.1" ) {
  $TORCHVISION_VERSION="0.6.1"
} Elseif ($TORCH_VERSION -eq "1.6.0") {
  $TORCHVISION_VERSION="0.7"
} Elseif ($TORCH_VERSION -eq "1.7.0") {
  $TORCHVISION_VERSION="0.8.1"
}
pip install torch==$TORCH_VERSION+cpu torchvision==$TORCHVISION_VERSION+cpu -f https://download.pytorch.org/whl/torch_stable.html

