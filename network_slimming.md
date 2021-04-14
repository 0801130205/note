$$
x^{l+1}=max\{\gamma^{l}\centerdot BN_{\mu^{l},\sigma^{l},\epsilon^{l}}(W^{l}*x^{l})+\beta,0\}
$$

#### If the subsequent convolution layer does not have batch normalization:

$$
x^{l+2}=max\{W^{l+1}*x^{l+1}+b^{l+1},0\}\\
b_{new}^{l+1}=b^{l+1}+sum\_H\_W(W^{l+1})*ReLU(I(\gamma=0)\centerdot \beta).reshape(-1,1)\\
x^{l+2}=max\{W^{l+1}*_{\gamma}x^{l+1}+b_{new}^{l+1},0\}
$$

$*_{\gamma}$ denotes the convolution operator which is only calculated along channels indexed by non-zeors of $\gamma$. * denote matrix multiplication or convolution.

#### if the subsequent convolution layer has batch normalization:



