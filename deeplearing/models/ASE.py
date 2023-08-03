import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    # print(in_channels)
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

def max_pooling(kernel_size):
    return nn.MaxPool2d(kernel_size)

def hw_flatten(x):
    return x.view(x.size(0), x.size(1), -1)

class attention_with_pooling(nn.Module):
    def __init__(self, channels, ksize=4, use_bias=True, sn=False, down_scale=2, pool_scale=2,
                 name='attention_pooling', padding='reflection'):
        super(attention_with_pooling, self).__init__()
        self.channels = channels
        self.ksize = ksize
        self.use_bias = use_bias
        self.sn = sn
        self.down_scale = down_scale
        self.pool_scale = pool_scale
        self.name = name
        self.padding = padding
        # self.conv_f = conv(channels, channels , kernel_size=1, stride=1, bias=use_bias)
        # self.conv_g = conv(channels, channels , kernel_size=1, stride=1, bias=use_bias)
        # self.conv_h = conv(channels, channels , kernel_size=1, stride=1, bias=use_bias)
        self.conv_f = conv(channels, channels // 16, kernel_size=1, stride=1, bias=use_bias)
        self.conv_g = conv(channels, channels // 16, kernel_size=1, stride=1, bias=use_bias)
        self.conv_h = conv(channels, channels // 16, kernel_size=1, stride=1, bias=use_bias)
        self.gamma = nn.Parameter(torch.zeros(1))
        if down_scale > 1:
            self.down_sample = conv(channels, channels, kernel_size=ksize, stride=down_scale,padding=0)
            self.up_sample = nn.ConvTranspose2d(channels // 16, channels , kernel_size=ksize, stride=down_scale)

    def forward(self, x):


        x_origin = x

        # down sampling
        # print(x.shape)
        if self.down_scale > 1:
            x = self.down_sample(x)

        # attention
        # print(x.shape)torch.Size([1, 12, 112, 112])

        f = self.conv_f(x) # [bs, c', h, w]
        f = max_pooling(self.pool_scale)(f)
        # f = F.avg_pool2d(f, self.pool_scale)

        g = self.conv_g(x) # [bs, c', h, w]

        h = self.conv_h(x) # [bs, c, h, w]
        h = max_pooling(self.pool_scale)(h)
        # h = F.avg_pool2d(h, self.pool_scale)

        # N = h * w
        s = torch.matmul(hw_flatten(g).transpose(1, 2), hw_flatten(f)) # [bs, N, N]

        beta = F.softmax(s, dim=-1) # attention map

        o = torch.matmul(beta, hw_flatten(h).transpose(1, 2)) # [bs, N, c']

        o = hw_flatten(o).transpose(1, 2) # [bs, c', N]
        # print(o_list.size())
        # o = o.reshape(o_list[0],o_list[1],224,224)
        # print("o", o.shape)
        # o = torch.nn.Conv2d(o.size()[1],out_channels=self.channels,kernel_size=1,stride=1)(o)

        o = o.view(x.size(0), self.channels // 16, x.size(2), x.size(3)) # [bs, c', h, w]
        # o = o.view(-1, self.channels , x.size(2), x.size(3)) # [bs, c', h, w]
        # print("o1",o.shape)
        # up sampling 上采样
        if self.down_scale > 1:
            o = self.up_sample(o)
        # o = torch.nn.Conv2d(o.size(1), out_channels=self.channels, kernel_size=1, stride=1)(o)
        # print("o", o.shape)
        o = self.gamma * o + x_origin
        return o


def gen_deconv(x, cnum, ksize=4, stride=2, rate=1, method='deconv', IN=True,
               activation=F.relu, name='upsample', padding='SAME', sn=False, training=True, reuse=False):
    with torch.nn.parameter.Parameter(torch.Tensor([1.0])):
        if method == 'nearest':
            x = F.interpolate(x, scale_factor=stride, mode='nearest')
            x = gen_conv(
                x, cnum, 3, 1, name=name+'_conv', padding=padding,
                training=training, IN=IN)
        elif method == 'bilinear':
            x = F.interpolate(x, scale_factor=stride, mode='bilinear')
            x = gen_conv(
                x, cnum, 3, 1, name=name + '_conv', padding=padding,
                training=training, IN=IN)
        elif method == 'bicubic':
            x = F.interpolate(x, scale_factor=stride, mode='bicubic', align_corners=False)
            x = gen_conv(
                x, cnum, 3, 1, name=name + '_conv', padding=padding,
                training=training, IN=IN)
        else:
            # if padding == 'SYMMETRIC' or padding == 'REFLECT':
            #     p = int(rate * (ksize - 1) / 2)
            #     p = 0
            #     x = F.pad(x, (p, p, p, p), mode=padding)
            padding = 1 if padding == 'SAME' else 0
            x = nn.ConvTranspose2d(in_channels=x.shape[1], out_channels=cnum, kernel_size=ksize, stride=stride,
                                   padding=padding, bias=not IN)
            if IN:
                x = nn.InstanceNorm2d(cnum)
            if activation is not None:
                x = activation(x)
    return x

def l2_norm(v):
    return v / torch.norm(v)

def spectral_norm(w, iteration=1):
    w_shape = w.shape
    w = w.view(-1, w_shape[-1])

    u = torch.randn(1, w_shape[-1], requires_grad=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = torch.matmul(u_hat, w.t())
        v_hat = l2_norm(v_)

        u_ = torch.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = torch.matmul(torch.matmul(v_hat, w), u_hat.t())
    w_norm = w / sigma

    with torch.no_grad():
        w_norm = w_norm.view(w_shape)

    return w_norm

def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv', IN=True, reuse=False,
             padding='SAME', activation=F.elu, use_bias=True, training=True, sn=False):
    assert padding in ['SYMMETRIC', 'SAME', 'REFLECT']
    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        """ 
        Padding layer.
        Dilated kernel size: k_r = ksize + (rate - 1)*(ksize - 1)
        Padding size: o = i + 2p - k_r and o = i, so p = rate * (ksize - 1) / 2 (when i and o has the same image shape)
        """
        p = int(rate * (ksize - 1) / 2)
        x = F.pad(x, (p, p, p, p), mode=padding)
        padding = 'VALID'

    # if spectrum normalization
    if sn:
        with torch.no_grad():
            w = nn.Parameter(torch.randn(cnum, x.shape[-1], ksize, ksize))
            nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')

            x = F.conv2d(x, spectral_norm(w),
                         stride=stride, padding=0, dilation=rate)
            if use_bias:
                bias = nn.Parameter(torch.zeros(cnum))
                x = F.bias_add(x, bias)
    else:
        # print(x.shape[-1],cnum)
        x = nn.Conv2d(in_channels=cnum, out_channels=x.shape[-1],
                      kernel_size=ksize, stride=stride, dilation=rate, padding=2,
                      bias=use_bias)(x)
        # x = nn.Conv2d(in_channels=x.shape[-1], out_channels=cnum,
        #               kernel_size=ksize, stride=stride, dilation=rate, padding='valid',
        #               bias=use_bias)(x)

    if IN:
        x = nn.InstanceNorm2d(cnum)(x)  # if instance norm? before non-linear activation!!!
    if activation is not None:
        x = activation(x)
    return x

def attention_with_neighbors(x, channels, ksize=3, use_bias=True, sn=False, stride=2,
                             down_scale=2, pool_scale=2, name='attention_pooling',
                             training=True, padding='REFLECT', reuse=False):
    x1 = x
    # downsample input feature maps if needed due to limited GPU memory
    # down sampling
    if down_scale > 1:
        conv = nn.Conv2d(x1.shape[1], channels, ksize, stride=down_scale, bias=use_bias)
        x1 = conv(x1)
        x1 = F.relu(x1)

    # get shapes
    int_x1s = list(x1.shape)

    # extract patches from high-level feature maps for matching and attending
    x1_groups = torch.split(x1, 1, dim=0)
    w = F.unfold(x1, ksize, stride=stride, padding=0)
    w = w.view(int_x1s[0], -1, ksize, ksize, int_x1s[1])
    w = w.permute(0, 2, 3, 4, 1)  # transpose to [b, ksize, ksize, c, hw/4]
    w_groups = torch.split(w, 1, dim=0)

    # matching and attending hole and non-hole patches
    y = []
    scale = 10.
    # high level patches: w_groups, low level patches: raw_w_groups, x2_groups: high level feature map
    for xi, wi in zip(x1_groups, w_groups):
        # matching on high-level feature maps
        wi = wi[0]
        wi_normed = wi / torch.clamp(torch.sqrt(torch.sum(wi ** 2, dim=[0, 1, 2])), min=1e-4)
        yi = F.conv2d(xi, wi_normed, stride=1, padding=1)
        yi = yi.view(1, int_x1s[1], int_x1s[2], (int_x1s[1] // stride) * (int_x1s[2] // stride))
        yi = F.softmax(yi * scale, dim=3)
        # non local mean
        wi_center = wi.permute(0, 1, 3, 2)
        yi = F.conv2d(yi, wi_center, stride=1, padding="SAME") / 4.

        # filter: [height, width, output_channels, in_channels]
        y.append(yi)
    y = torch.cat(y, dim=0)
    y = y.view(int_x1s)

    # up sampling
    if down_scale > 1:
        deconv = nn.ConvTranspose2d(y.shape[1], channels, ksize, stride=down_scale, bias=use_bias)
        y = deconv(y)
        y = F.relu(y)

    gamma = nn.Parameter(torch.zeros(1))
    x = gamma * y + x
    x = nn.Conv2d(x.shape[1], channels, 3, 1, dilation=1, padding=1)
    x = F.relu(x)
    return x


def attention(x, channels, neighbors=1, use_bias=True, sn=False, down_scale = 2, pool_scale=2,
              name='attention_pooling', training=True, padding='REFLECT', reuse=False):
    # print(channels)
    if neighbors > 1:
        
        an = attention_with_neighbors(x,channels=channels).cuda()
        x = an(x)
    else:

        # x = attention_with_pooling(x, channels, down_scale=down_scale, pool_scale=pool_scale, name=name)

        # print("-------------------------")
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        ap = attention_with_pooling(channels=channels).cuda()

        # print(ap.cuda())
        x = ap(x)

    return x

def ase(x):
    cnum = list(x.size())[1] // 4
    # x_64 = x
    x = attention(x, neighbors=1, channels=4 * cnum, down_scale=2, pool_scale=2, name='attention_pooling_64')
    # gc = gen_conv_c(cnum=cnum,ksize=5).cuda()
    # x_64 = nn.Conv2d(in_channels=cnum * 4, out_channels=x.shape[-1],
    #           kernel_size=5, stride=1, dilation=1, padding=2,
    #           bias=True)
    # x_64 = gc(x)
    # x_cat = torch.cat([x, x_64], dim=1)

    # x = torch.nn.Conv2d(in_channels=x.size(1) + x_64.size(1), out_channels=4 * cnum, kernel_size=1, stride=1).cuda()(x_cat)
    return x

class gen_conv_c(nn.Module):
    def __init__(self, cnum, ksize, stride=1, rate=1, name='conv', IN=True, reuse=False,
             padding='SAME', activation=F.elu, use_bias=True, training=True, sn=False):
        super(gen_conv_c, self).__init__()
        self.cnum = cnum
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.name = name
        self.IN = IN
        self.reuse = reuse
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.training = training
        self.sn = sn
    def forward(self,x):
        assert self.padding in ['SYMMETRIC', 'SAME', 'REFLECT']
        if self.padding == 'SYMMETRIC' or self.padding == 'REFLECT':
            """ 
            Padding layer.
            Dilated kernel size: k_r = ksize + (rate - 1)*(ksize - 1)
            Padding size: o = i + 2p - k_r and o = i, so p = rate * (ksize - 1) / 2 (when i and o has the same image shape)
            """
            p = int(self.rate * (self.ksize - 1) / 2)
            x = F.pad(x, (p, p, p, p), mode=self.padding)
            padding = 'VALID'
        # if spectrum normalization
        if self.sn:
            with torch.no_grad():
                w = nn.Parameter(torch.randn(self.cnum, x.shape[-1], self.ksize, self.ksize))
                nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')

                x = F.conv2d(x, spectral_norm(w),
                             stride=self.stride, padding=0, dilation=self.rate)
                if self.use_bias:
                    bias = nn.Parameter(torch.zeros(cnum))
                    x = F.bias_add(x, bias)
        else:
            x = nn.Conv2d(in_channels=self.cnum*4, out_channels=x.shape[-1],
                          kernel_size=self.ksize, stride=self.stride, dilation=self.rate, padding=2,
                          bias=self.use_bias).cuda()(x)

        if self.IN:
            x = nn.InstanceNorm2d(self.cnum)(x)  # if instance norm? before non-linear activation!!!
        if self.activation is not None:
            x = F.elu(x)
        return x
