[string]$TORCH_VERSION=$args[0]
If ($TORCH_VERSION -eq "1.4.0") {
  $TORCHVISION_VERSION="0.5.0"
} Elseif ( $TORCH_VERSION -eq "1.5.0" ) {
  $TORCHVISION_VERSION="0.6.0"
} Elseif ( $TORCH_VERSION -eq "1.5.1" ) {
  $TORCHVISION_VERSION="0.6.1"
} Elseif ($TORCH_VERSION -eq "1.6.0") {
  $TORCHVISION_VERSION="0.7"
  $TORCHCSPRNG_VERSION="0.1.2"
} Elseif ($TORCH_VERSION -eq "1.7.0") {
  $TORCHVISION_VERSION="0.8.1"
  $TORCHCSPRNG_VERSION="0.1.3"
} Elseif ($TORCH_VERSION -eq "1.7.1") {
  $TORCHVISION_VERSION="0.8.2"
  $TORCHCSPRNG_VERSION="0.1.4"
}
pip install torch==$TORCH_VERSION+cpu torchvision==$TORCHVISION_VERSION+cpu -f https://download.pytorch.org/whl/torch_stable.html

If ($TORCH_VERSION -eq "1.4.0") {
  echo "No torchcsprng"
} Elseif ($TORCH_VERSION -eq "1.5.0" ) {
  echo "No torchcsprng"
} Elseif ($TORCH_VERSION -eq "1.5.1" ) {
  echo "No torchcsprng"
} Else {
  pip install torchcsprng==$TORCHCSPRNG_VERSION+cpu -f https://download.pytorch.org/whl/torch_stable.html
}
