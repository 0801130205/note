

Convolution layer
---------------------------------

**Note.** The "convolution" operator in deep learning domain is
"cross-correlation" in signal process domain. In view of industry
practice, we also define "convolution" equals to "cross-correlation",
and "cross-correlation" is "convolution".

The convolution algorithm is ubiquitous in deep learning convolution
neural nets. We describe the basic mathematics that being used in suDNN.

.. TODO: illustrate algorithm

In this chapter, we have following definitions:

Tensors:

+-----------------------+-------------------------------+
|        Tensor         |           describe            |
+=======================+===============================+
| :math:`\vec{x}`       | input tensor                  |
+-----------------------+-------------------------------+
| :math:`\vec{x'}`      | padded input tensor           |
+-----------------------+-------------------------------+
| :math:`\vec{\omega}`  | weight tensor                 |
+-----------------------+-------------------------------+
| :math:`\vec{y}`       | output tensor                 |
+-----------------------+-------------------------------+
| :math:`\vec{dy}`      | input grad tensor             |
+-----------------------+-------------------------------+
| :math:`\vec{dy'}`     | dilated input grad tensor     |
+-----------------------+-------------------------------+
| :math:`\vec{dx}`      | partial derivative for input  |
+-----------------------+-------------------------------+
| :math:`\vec{d\omega}` | partial derivative for weight |
+-----------------------+-------------------------------+
| :math:`\vec{b}`       | bias tensor                   |
+-----------------------+-------------------------------+
| :math:`\vec{db}`      | partial derivative for bias   |
+-----------------------+-------------------------------+

Parameters:

+-----------+---------------------------------------------+
|   Param   |                  describe                   |
+===========+=============================================+
| :math:`P` | pad                                         |
+-----------+---------------------------------------------+
| :math:`S` | stride                                      |
+-----------+---------------------------------------------+
| :math:`D` | dialtion                                    |
+-----------+---------------------------------------------+
| :math:`R` | whether a feature map rotated by 180 degree |
+-----------+---------------------------------------------+
| :math:`N` | tensor batch size                           |
+-----------+---------------------------------------------+
| :math:`C` | tensor channels                             |
+-----------+---------------------------------------------+
| :math:`H` | tensor height                               |
+-----------+---------------------------------------------+
| :math:`W` | tensor width                                |
+-----------+---------------------------------------------+
| :math:`K` | number of filters(only for weight tensor)   |
+-----------+---------------------------------------------+

Suffixes:

+-------------+------------------------------------+------------------+
|   Suffix    |              describe              |     used in      |
+=============+====================================+==================+
| :math:`n`   | batch size                         | tensor           |
+-------------+------------------------------------+------------------+
| :math:`c`   | channels                           | tensor           |
+-------------+------------------------------------+------------------+
| :math:`h`   | height or vertical value           | tensor/parameter |
+-------------+------------------------------------+------------------+
| :math:`w`   | width or horizontal value          | tensor/parameter |
+-------------+------------------------------------+------------------+
| :math:`k`   | number of filters                  | weight tensor    |
+-------------+------------------------------------+------------------+
| :math:`l`   | backward pad parameter at the left | pad parameter    |
+-------------+------------------------------------+------------------+
| :math:`t`   | backward pad parameter at the top  | pad parameter    |
+-------------+------------------------------------+------------------+
| :math:`f`   | forward parameters                 | parameter        |
+-------------+------------------------------------+------------------+
| :math:`b`   | backward parameters                | parameter        |
+-------------+------------------------------------+------------------+
| all tensors | attributes of a tensor             | tensor shapes    |
+-------------+------------------------------------+------------------+

Generally, the number of filter kernel channels should match the number
of input channels :

.. math::
    

    \vec{x}_c = \vec{\omega}_c

The output shape can be calculated from the shapes of input, filter and
the convolution parameters as :

.. math::

    W_{\vec{y}} = 1 + \frac{W_{\vec{x}} + 2 * P_w - ((W_{\vec{\omega}} - 1) * D_w + 1)}{S_w}

.. math::

    H_{\vec{y}} = 1 + \frac{H_{\vec{x}} + 2 * P_h - ((H_{\vec{\omega}} - 1) * D_h + 1)}{S_h}

.. math::

    C_{\vec{y}} = K_{\vec{\omega}}

.. math::

    N_{\vec{y}} = N_{\vec{x}}

In this version of suDNN, we assume the tensors are colum majored, and
the tensor ``NCHW`` formatted, namely, the continuous dimension in
memory is the ``W`` dimension. Transforming APIs shall be provided to
allow transform to other formats.

Conv2d forward

~~~~~~~~~~~~~~
With these parameters, the computation process is then illustrated in
the following figures.

|Elementwise computation|

.. raw:: html

   <center>

**Fig.** Elementwise level computation.

.. raw:: html

   </center>

The convolution for a 2d feature map and a 2d filter-kernel is straight
forward. Each element of the output is obtained from a sliding window
like direct vector dot operation of corresponding elements of the kernel
and the feature map.

|Channel level computation|

.. raw:: html

   <center>

**Fig.** Channel level computation.

.. raw:: html

   </center>

When there are multiple channels, the results obtained from each
channel-wise convolution should be summed up to form a single frame of
feature map.

|Batch level computation|

.. raw:: html

   <center>

**Fig.** Batch level computation.

.. raw:: html

   </center>

At batch level, where there are multiple channels and multiple feature
maps, the ``n``-th input feature map is convoluted with the
:math:`K_{\vec{\omega}}` filters obtaining the ``k``-th channel of the ``n``-th
output feature. This process is repeated for each feature map in the
batch [1].

We only support zero-padding in suDNN. As the following figure and
equation illustrates, zero-padding adds zeros around a 2D feature map.

.. math::

    x'_{n,c,h,w} =\left\{
    \begin{aligned}
    0 & & (-P_h\leq h<0 & & or & & H_{\vec{x}}<h\leq H_{\vec{x}}+P_h) \\
    0 & & (-P_w\leq w<0 & & or & & W_{\vec{x}}<w\leq W_{\vec{x}}+P_w) \\
    x_{n,c,h,w} & & (0 \leq h \leq H_{\vec{x}} & & and & & 0 \leq w_p \leq W_{\vec{x}})
    \end{aligned}
    \right.

.. raw:: html

   <center>

|Common padding|

**Fig.** Common padding, pad = 1.

.. raw:: html

   </center>

The final formula reads:    

.. math::

    \vec{y}_{n,k,h,w} = \sum_{c=0}^{C_{\vec{\omega}}-1}\sum_{j=0}^{H_{\vec{\omega}}-1}\sum_{i=0}^{W_{\vec{\omega}}-1}\vec{\omega}_{k,c,j,i}*\vec{x'}_{n,c,S_h*h+j*D_h,S_w*w+i*D_w}

Conv2d backward data
~~~~~~~~~~~~~~~~~~~~

This function presents backward data, also called backward propagation
activation(BPA).

In general, we know its behavior:

.. math::

    \vec{dx} = \vec{dy} \otimes \vec{w^{R}}

Although this pass looks like a common convolution, we need to do dilation and
padding for the :math:`\vec{dy}`, and also need to flip filter.

There is a base rule. :math:`D_b` = :math:`S_f`, we should
inert 0 on :math:`\vec{dy}` by :math:`D_b` size to make dilated matrix at first. Then add
left and top padding, if the dimensions of :math:`\vec{dy}` unequal dimensions of dx
which had padded, add right and down corresponding paddings to arrive
dimensions of dx [2].

The padding formula as following:

.. math::

    P_{bw} = W_{\vec{x}} - W_{\vec{dy}} * S_w + (W_{\vec{\omega}} - 1) * D_w \\
    P_{bh} = H_{\vec{x}} - H_{\vec{dy}} * S_h + (H_{\vec{\omega}} - 1) * D_h \\
    \\
    P_l = \left\{\begin{aligned}
    \quad\frac{P_{bw}}{2}\quad (P_{bw}\ mod\ 2 = 0) \\
    \frac{P_{bw}}{2} + 1 \  (P_{bw}\ mod\ 2 \neq 0)
    \end{aligned}\right. 
    \\
    P_t = \left\{\begin{aligned}
    \quad\frac{P_{bh}}{2}\quad (P_{bh}\ mod\ 2 = 0) \\
    \frac{P_{bh}}{2} + 1 \  (P_{bh}\ mod\ 2 \neq 0)
    \end{aligned}\right.

For example.

+-----------------------------------+-----------+-----------+
|               param               | example 1 | example 2 |
+===================================+===========+===========+
| :math:`S_f`                       | 2         | 2         |
+-----------------------------------+-----------+-----------+
| :math:`P`                         | 1         | 1         |
+-----------------------------------+-----------+-----------+
| dimension of :math:`\vec{dx}`     | 5 x 5     | 8 x 8     |
+-----------------------------------+-----------+-----------+
| dimension of :math:`\vec{dy}`     | 3 x 3     | 5 x 5     |
+-----------------------------------+-----------+-----------+
| dimension of :math:`\vec{\omega}` | 3 x 3     | 3 x 3     |
+-----------------------------------+-----------+-----------+

Get :math:`\vec{dy'}` as the following fig.

.. raw:: html

    <center>

|padding 5x5|

.. raw:: html

   <center>


**Fig.** Convolution backward data 1.

|padding 8x8|

.. raw:: html

   <center>


**Fig.** Convolution backward data 2.

.. raw:: html

    </center>

How to flip: For each channel of a filter, rotate by 180°. There are
two modes for backward data. The **SUDNN_CROSS_CORRELATION** mode is it
necessary to do the flip, and the **SUDNN_CONVOLUTION** mode is to keep
original filter for backward data .

.. raw:: html

    <center>

|Flip weight|

**Fig.** Flip weight

.. raw:: html

    </center>

It is that once we finished the two steps pre-process, the following
steps are common convolution, but set :math:`S_b` to 1.

Conv2d backward filter

~~~~~~~~~~~~~~~~~~~~~~
Backward propagation of filter of conv2d, is to compute the gradient of
weight, namely :math:`dw`. The operation is complicated and confusing,
so we start from a simple case, where channel num and batch equal 1, and
no padding, stride, dilation equal to 1.

As in the notation table, :math:`dw` has the same shape with :math:`w`,
and :math:`dy` has the same shape with :math:`y`, which could be
inferred given :math:`x, w` shapes and convolution parameters (padding,
stride, dilation).

In the first simple case, the operation is like a convolution, taking
:math:`dy` as the kernel. :math:`dy` slides in :math:`x_{pad}`,
generating :math:`dw`. At last, rotate 180° on :math:`dw`. Or, the
rotation is not needed when ``SUDNN_CROSS_CORRELATION`` is specified.

.. math::

    \vec{d\omega}_{k,c,h,w} = \sum_{n = 0}^{N_{\vec{dy}}}\sum_{i = 0}^{H_{\vec{dy}}-1}\sum_{j = 0}^{W_{\vec{dy}}-1}{\vec{x'}_{n,c,i * S_{bh} + h * D_{bh},j * S_{bw} + w * D_{bw}}*\vec{dy}_{n,c,i,j}}

.. raw:: html

    <center>

|Conv bwd filter|

**Fig.** Convolution backward filter 1

.. raw:: html

   </center>

**padding, stride, and dilation**

Taking padding, stride, and dilation into account, as in the next case,
x is padded first. :math:`dy` now has a different shape ``3x3``. See the
backward convolution as a forward convolution, the new stride and
dilation are the converse of the original forward convolution
parameters. The dilation will be the original stride and the stride will
be the original dilation.

.. raw:: html

    <center>

|Conv bwd filter2|

**Fig.** Convolution backward filter 2

.. raw:: html

   </center>

**multi-batch, multi-channel**

More generally, when multi-batch and multi-channel are considered,
backward convolution is different from forwarding convolution. As you
know, :math:`x, dy` have the same batch, while the batch of :math:`dw`
equals to channel the number of :math:`dy`, the channel number of
:math:`dw` equals to the channel number of :math:`x`. Within a batch of
:math:`x, dy`, one channel of :math:`dy` convolutes with each channel of
:math:`x`, resulting in multiple channels of :math:`dw_{tmp}`. Another
channel of :math:`dy` convoluted with :math:`x` results in another batch
of :math:`dw`. At last, sum the corresponding batches of
:math:`dw_{tmp}` to get the :math:`dw`.

.. raw:: html

    <center>

|Conv bwd filter3|

**Fig.** Convolution backward filter 3

.. raw:: html

   </center>

`Stanford CS231N <http://cs231n.stanford.edu/>`__

Conv2d backward bias
~~~~~~~~~~~~~~~~~~~~

In general, convolution is usually followed by a ``bias`` operation. As its name,
after convolution, we need to add a :math:`\vec{b}` over each output channel.

And the convolution backward bias is to compute the gradient for the
``bias`` during a convolution backward pass.

The formula is easy to write:

.. math::

    \vec{db}_{c} = \sum\limits_{n=0}^{N_{\vec{dy}}-1}\sum\limits_{j=0}^{H_{\vec{dy}}-1}\sum\limits_{i=0}^{W_{\vec{dy}}-1}\vec{dy}_{n,c,j,i}


.. |Elementwise computation| image:: img/theory/conv3.PNG
.. |Channel level computation| image:: img/theory/conv2.PNG
.. |Batch level computation| image:: img/theory/conv1.PNG
.. |Common padding| image:: img/theory/common_padding.png
.. |padding 5x5| image:: img/theory/dilation_and_padding_5x5.png
.. |padding 8x8| image:: img/theory/dilation_and_padding_8x8.png
.. |Flip weight| image:: img/theory/flip_filter.png
.. |Conv bwd filter| image:: img/theory/conv_bwd_filter1.PNG
.. |Conv bwd filter2| image:: img/theory/conv_bwd_filter2.PNG
.. |Conv bwd filter3| image:: img/theory/conv_bwd_filter3.PNG

.. [1] `Y. LeCun. Gradient-Based Learning Applied to Document Recognition. 1998 <https://pdfs.semanticscholar.org/62d7/9ced441a6c78dfd161fb472c5769791192f6.pdf>`__
.. [2] `Stanford CS231N <http://cs231n.stanford.edu/>`__