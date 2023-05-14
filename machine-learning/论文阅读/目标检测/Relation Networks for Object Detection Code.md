# Relation Networks for Object Detection Code

代码梳理流程。

# 大致流程

主要思路是建议一个关系模块，然后再Region Feature Extraction 后 fc --> relation --> fc --> relation --> Duplicate Removal --> relation --> 结果。

Dumplicate Removal 可以加可以不加，仍可选用nms。暂定不加Dumplicate Removal

<img src="..\..\pics\CV\ISG\Relation Networks for Object Detection Code\image-20210528115648078.png">

# 代码从后向前梳理

> 在Region Feature Extraction后进行fc，relation，fc，relation

```python
# 提取感兴趣的特征图
self.rpn = RegionProposalNetwork(in_channels=512, mid_channels=512, feat_stride=self.feat_stride)

# fc + relation RoIHead 中加入的关系模块。Mask R-CNN也在里面加。 classifier=classifier 代表是否是分类任务。因为检测是多任务的，一个分类，一个检测。
# in_channels 是输入的特征图的通道数；fc_features是全连接后输出的通道数。n_relations是关系模块的数目
self.roi_head = RoIHead(n_class=self.n_class, roi_size=7, spatial_scale=(1. / self.feat_stride), n_relations=0,
                        in_channels=512, fc_features=4096, classifier=classifier)
```

> 我们需要在RoIHead里增加关系模块RelationMpdule

```python
    def __init__(self, n_class, roi_size, spatial_scale,
                 in_channels=128, fc_features=1024, n_relations=0, classifier=None):
        # n_class includes the background
        super(RoIHead, self).__init__()
        # 如果不是分类任务。 是bounding boxes predict
        if classifier is None:
            self.n_relations = n_relations
            fully_connected1 = nn.Linear(7 * 7 * in_channels, fc_features)
            relu1 = nn.ReLU(inplace=True)

            fully_connected2 = nn.Linear(fc_features, fc_features)
            relu2 = nn.ReLU(inplace=True)
            # 在全连接+relu后加入关系模块
            if n_relations > 0:
                self.dim_g = int(fc_features / n_relations)
                relation1 = RelationModule(n_relations=n_relations, appearance_feature_dim=fc_features,
                                           key_feature_dim=self.dim_g, geo_feature_dim=self.dim_g)

                relation2 = RelationModule(n_relations=n_relations, appearance_feature_dim=fc_features,
                                           key_feature_dim=self.dim_g, geo_feature_dim=self.dim_g)
                self.classifier = nn.Sequential(fully_connected1, relu1, relation1,
                                                fully_connected2, relu2, relation2)
            else:
                self.classifier = nn.Sequential(fully_connected1, relu1,
                                                fully_connected2, relu2)
        else:
            self.classifier = classifier

        self.cls_loc = nn.Linear(fc_features, n_class * 4)
        self.score = nn.Linear(fc_features, n_class)
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)
```

> RelationModule代码

```python
class RelationModule(nn.Module):
    def __init__(self, n_relations=16, appearance_feature_dim=1024, key_feature_dim=64, geo_feature_dim=64,
                 isDuplication=False):
        super(RelationModule, self).__init__()
        self.isDuplication = isDuplication
        self.Nr = n_relations
        self.dim_g = geo_feature_dim
        self.relation = nn.ModuleList()
        for N in range(self.Nr):
            self.relation.append(RelationUnit(appearance_feature_dim, key_feature_dim, geo_feature_dim))

    def forward(self, input_data):
        if (self.isDuplication):
            f_a, embedding_f_a, position_embedding = input_data
        else:
            f_a, position_embedding = input_data
        isFirst = True
        for N in range(self.Nr):
            if (isFirst):
                if (self.isDuplication):
                    concat = self.relation[N](embedding_f_a, position_embedding)
                else:
                    concat = self.relation[N](f_a, position_embedding)
                isFirst = False
            else:
                if (self.isDuplication):
                    concat = torch.cat((concat, self.relation[N](embedding_f_a, position_embedding)), -1)
                else:
                    concat = torch.cat((concat, self.relation[N](f_a, position_embedding)), -1)
        return concat + f_a
```

> RelationUnit代码

```python
class RelationUnit(nn.Module):
    def __init__(self, appearance_feature_dim=1024, key_feature_dim=64, geo_feature_dim=64):
        super(RelationUnit, self).__init__()
        self.dim_g = geo_feature_dim
        self.dim_k = key_feature_dim
        self.WG = nn.Linear(geo_feature_dim, 1, bias=True)
        self.WK = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WQ = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.WV = nn.Linear(appearance_feature_dim, key_feature_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_a, position_embedding):
        N, _ = f_a.size()

        position_embedding = position_embedding.view(-1, self.dim_g)

        w_g = self.relu(self.WG(position_embedding))
        w_k = self.WK(f_a)
        w_k = w_k.view(N, 1, self.dim_k)

        w_q = self.WQ(f_a)
        w_q = w_q.view(1, N, self.dim_k)

        scaled_dot = torch.sum((w_k * w_q), -1)
        scaled_dot = scaled_dot / np.sqrt(self.dim_k)

        w_g = w_g.view(N, N)
        w_a = scaled_dot.view(N, N)

        w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        w_mn = torch.nn.Softmax(dim=1)(w_mn)

        w_v = self.WV(f_a)

        w_mn = w_mn.view(N, N, 1)
        w_v = w_v.view(N, 1, -1)

        output = w_mn * w_v

        output = torch.sum(output, -2)
        return output
```

打印的关系模块的内容如下：

```shell
RelationModule(
  (relation): ModuleList(
    (0): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (1): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (2): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (3): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (4): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (5): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (6): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (7): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (8): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (9): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (10): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (11): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (12): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (13): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (14): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
    (15): RelationUnit(
      (WG): Linear(in_features=64, out_features=1, bias=True)
      (WK): Linear(in_features=1024, out_features=64, bias=True)
      (WQ): Linear(in_features=1024, out_features=64, bias=True)
      (WV): Linear(in_features=1024, out_features=64, bias=True)
      (relu): ReLU(inplace=True)
    )
  )
)
```

查看单个关系单元的内容。

```shell
RelationUnit(
    (WG): Linear(in_features=64, out_features=1, bias=True)
    (WK): Linear(in_features=1024, out_features=64, bias=True)
    (WQ): Linear(in_features=1024, out_features=64, bias=True)
    (WV): Linear(in_features=1024, out_features=64, bias=True)
    (relu): ReLU(inplace=True)
)
```

----

```python
    def train_step(self, data):
        self.optimizer.zero_grad()
        losses = self.my_model.get_loss(
            [data[0].cuda().float(), data[1].cuda().float(), data[2].cuda().float(), data[3].cuda().float()],
            opt.isLearnNMS)
        if losses[0] == 0.:
            return 1.
        losses[0].backward()
        torch.nn.utils.clip_grad_norm_(self.my_model.parameters(), 0.1)

        self.optimizer.step()

        curr_loss = losses[0].item()
        return curr_loss
```

```python
appearance_features 是 duplicate_remover需要用到的，我暂时不加duplicate_remover网络
```