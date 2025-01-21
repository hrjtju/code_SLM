> 以串行二维装箱模型作为基准，在其他两个模型中标识与基准模型不同的地方
# 1 环境模型
## 1.1 串行二维装箱模型（基准环境）

* 从零开始，一个箱子装满了再装下一箱，然后封住这一箱不再放进工件
* 状态为（1）现阶段装箱中的等高线图（2）装箱尺寸（3）现阶段所有工件的剩余量和所有工件的尺寸参数
* 动作为选择零件种类，选择零件朝向。为一长向量，然后转变为矩阵。
    * 可优化为：选择零件种类，及该零件的朝向

## 1.2 并行一维装箱模型

> 很松的约束，没有二维装箱问题的影响，通过此环境检查强化学习算法及MDP定义是否存在问题

* 估计零件所占总空间量，一次性开 $K$ 个空箱，确保可以装下所有零件
* 状态（1）现阶段所有箱子中所占用的总底面积及其与箱子总底面积的比值（占用率矩阵）（2）现阶段所有工件的剩余量和所有工件尺寸 **（3）当前箱子中最高的零件** #DIFF 
* 动作为选择零件种类，选择零件朝向，该零件的朝向，该零件应该放到哪个箱子

## 1.3 并行二维装箱模型

> 更加实际的模型，但依然不考虑后面的scheduling的等等问题

* 估计零件所占总空间量，一次性开 $K$ 个空箱，确保可以装下所有零件
* 状态（1）现阶段装箱中的等高线图（2）装箱尺寸（3）现阶段所有工件的剩余量和所有工件的尺寸参数
* 动作为选择零件种类，选择零件朝向，该零件的朝向，该零件应该放到哪个箱子

# 2 智能体规则
## 2.1 串行二维装箱模型

1. 环境在接受动作后进行前处理：
    1. 使用$O_{\text{mask}}$对动作矩阵进行过滤
        * 将零件个数为零的种类对应的行归零
        * 将所有不合法的打印方向对应的分量归零
    1. 首先对第一维（零件种类）做argmax，选出分数总和最大的那一个零件种类确定为本次操作的零件种类（orientation数目的问题？）
    2. 然后对动作矩阵中对应这一零件种类的行向量做argsort，得到这一零件种类的打印方向分数排序
    3. 令此时的打印方向为排序第一的那一个打印方向
2. 尝试按照上述确定的零件种类和打印方向
    * 成功：更新状态
        1. 将$V_{t}$更换为当前Batch的新视图$V_{t}$
        2. 将零件信息矩阵中对应种类的零件个数$-1$
        3. *如果此时所有零件都被分配，则结束环境*
    * 失败：
        1. 尝试更换分数下一高的打印方向重新尝试
            * 如果成功，执行上面成功的流程
            * 如果失败，更换下一高的打印方向==（惩罚？）==
        2. 如果没有打印方向可以更换
            1. Solution中开启新Batch
            2. 重新按照分数第一大的打印方向尝试装箱
                * 如果成功，执行上面成功的流程
                * 如果失败，掐断环境（理论上不会失败）


## 2.2 并行一维装箱模型

1. 环境在接受动作后进行前处理：
    1. 使用$O_{\text{mask}}$对动作矩阵进行过滤
        * 将零件个数为零的种类对应的行归零
        * 将所有不合法的打印方向对应的分量归零
    2. 切分动作向量为三个域，分别为零件种类，零件朝向和batch分布，然后argsort
2. 尝试按照上述确定的零件种类和打印方向
    * 成功：更新状态
        1. 将$V_{t}$更换为当前Batch的新视图$V_{t}$
        2. 将零件信息矩阵中对应种类的零件个数$-1$
        3. *如果此时所有零件都被分配，则结束环境*
    * 失败：按照batch、零件朝向和零件种类的顺序依次尝试输出值下一高的分配情况，直到可以成功放进箱子为止

## 2.3 并行二维装箱模型

和2.2相同

# 3 API文档
## 3.1 串行二维装箱模型
### 3.1.1 基本数据元素类
#### 3.1.1 `ItemFromJson`

基类。用于装载Json文件中的工件、机器和加工参数

| 方法                                               | 类型                                 | 解释                                                            |
| ------------------------------------------------ | ---------------------------------- | ------------------------------------------------------------- |
| `get_from_dict(self, d: Dict[str, Any]) -> None` | `Callable[[Dict[str, Any]], None]` | 从字典中按照键值对提取为类实例参数。如 `{"asd": 123}` 变成类中有一参数为 `asd`，其值为 `123`。 |
| `show(self) -> None`                             | `Callable[[None], str]`            | 用于输出该类中的所有参数及其值。                                              |

> 修改时间：2025年1月11日20:26:33

#### 3.1.1.2 `Instance`

数据类。用于装载所有的实例参数。

| 参数                      | 类型    | 解释             |
| ----------------------- | ----- | -------------- |
| `self.num_parts`        | `int` | 该算例中的所有零件数量    |
| `self.num_orientations` | `int` | 该算例中零件朝向数量的最大值 |
| `self.type_parts`       | `int` | 该算例中零件的种类数目    |
> 修改时间：2025年1月11日20:31:50

#### 3.1.1.3 `Machine`

类。用于装载机器参数

| 参数或方法                    | 类型                                             | 解释                 |
| ------------------------ | ---------------------------------------------- | ------------------ |
| `Machine.A`              | `float`                                        | 耗能模型中用于计算*加工时间*的参数 |
| `Machine.B`              | `float`                                        | 耗能模型中用于计算*加工时间*的参数 |
| `Machine.C`              | `float`                                        | 耗时模型中用于计算*充氧时间*的参数 |
| `Machine.D`              | `float`                                        | 耗时模型中用于计算*充氧时间*的参数 |
| `Machine.E`              | `float`                                        | 耗时模型中用于计算*加热时间*的参数 |
| `Machine.F`              | `float`                                        | 耗时模型中用于计算*加热时间*的参数 |
| `Machine.G`              | `float`                                        | 耗时模型中用于计算*加热时间*的参数 |
| `Machine.H`              | `float`                                        | 耗时模型中用于计算*冷却时间*的参数 |
| `Machine.I`              | `float`                                        | 耗时模型中用于计算*冷却时间*的参数 |
| `Machine.J`              | `float`                                        | 耗时模型中用于计算*冷却时间*的参数 |
| `get_lwh(self)`          | `Callable[[None], Tuple[float, float, float]]` | 获取机器工作区域的长宽高       |
| `self.power_subsystems`  | `Dict[str, float]`                             | 诸子系统功率参数           |
| `self.power_coefficient` | `DataFrame`                                    | 能源系统的功率矩阵          |

**诸子系统功率参数**

* `basic_subsystem`
* `platform_heater`
* `water_circulation_unit`
* `water_cooling_unit`
* `laser_scanning_border`
* `laser_filling_contour`
* `laser_volume_hatching`
* `laser_supports_building`
* `recoater_motor`
* `electric_valves`
* `gas_circulation_pump_motor`

**功率矩阵**

是形如下方的矩阵，每行每列有特定的名称

|                               | preheating | scanning<br>border | filling<br>contour | volume<br>hatching | supports<br>soliding | recoating | cooling |
| ----------------------------- | :--------: | :----------------: | :----------------: | :----------------: | :------------------: | :-------: | :-----: |
| basic<br>subsystem            |   $1.0$    |       $1.0$        |       $1.0$        |       $1.0$        |        $1.0$         |   $1.0$   |  $1.0$  |
| platform<br>heater            |   $1.0$    |       $0.5$        |       $0.5$        |       $0.5$        |        $0.5$         |   $0.5$   |  $0.0$  |
| water<br>circulation unit     |   $1.0$    |       $1.0$        |       $1.0$        |       $1.0$        |        $1.0$         |   $1.0$   |  $1.0$  |
| water <br>coolingunit         |   $0.2$    |       $0.4$        |       $0.4$        |       $0.4$        |        $0.4$         |   $0.4$   |  $0.2$  |
| laser<br>scanning_border      |   $0.0$    |       $1.0$        |       $0.0$        |       $0.0$        |        $0.0$         |   $0.0$   |  $0.0$  |
| laser<br>filling_contour      |   $0.0$    |       $0.0$        |       $1.0$        |       $0.0$        |        $0.0$         |   $0.0$   |  $0.0$  |
| laser<br>volume_hatching      |   $0.0$    |       $0.0$        |       $0.0$        |       $1.0$        |        $0.0$         |   $0.0$   |  $0.0$  |
| laser<br>supports_building    |   $0.0$    |       $0.0$        |       $0.0$        |       $0.0$        |        $1.0$         |   $0.0$   |  $0.0$  |
| recoater<br>motor             |   $0.0$    |       $0.0$        |       $0.0$        |       $0.0$        |        $0.0$         |   $1.0$   |  $0.0$  |
| electric<br>valves            |   $1.0$    |       $1.0$        |       $1.0$        |       $1.0$        |        $1.0$         |   $1.0$   |  $0.0$  |
| gas_circulation<br>pump_motor |   $0.0$    |       $1.0$        |       $1.0$        |       $1.0$        |        $1.0$         |   $1.0$   |  $0.0$  |


> 修改时间：2025年1月11日20:38:42

#### 3.1.1.4 `Process`

数据类。用于装载工艺参数

| 参数                                | 类型      | 解释            |
| --------------------------------- | ------- | ------------- |
| `self.min_distance_parts`         | `float` | 零件间的最短距离      |
| `self.min_distance_part_platform` | `float` | 零件与工作台边缘的最短距离 |
| `self.num_laser`                  | `float` | 激光束数量         |
| `self.hatch_distance_volume`      | `float` | *？            |
| `self.hatch_distance_support`     | `float` | *？            |
| `self.laser_speed_border`         | `float` | *激光烧结速度？      |
| `self.laser_speed_contour`        | `float` | *激光烧结速度？      |
| `self.laser_speed_volume`         | `float` | *激光烧结速度？      |
| `self.layer_thickness`            | `float` | 粉料单层厚度        |
| `self.recoater_time_single`       | `float` | *单次铺粉时间？      |
| `self.heat_time`                  | `float` | 加热时间          |
| `self.cool_time`                  | `float` | 冷却时间          |
> 修改时间：2025年1月11日22:04:56

#### 3.1.1.5 `Part`

类。用于装载每种零件的参数并提供简单的计算服务

| 参数或方法                                   | 类型                            | 解释              |
| --------------------------------------- | ----------------------------- | --------------- |
| `self.part_type`                        | `int`                         | 该算例内的零件种类序号     |
| `self.num_part`                         | `int`                         | 该算例中该种零件的总数     |
| `self.volume`                           | `float`                       | 该种零件的体积         |
| `self.surface_area`                     | `float`                       | 该种零件的表面积        |
| `self.build_params`                     | `List[Dict[str, float\|int]]` | 该种零件的           |
| `self.gap`                              | `float`                       | 零件间的距离          |
| `get_proj_area(self, orientation: int)` | `Callable[[int], float]`      | 获取零件的投影面积（占地面积） |
**零件的成形参数**

列表，里面装着字典。每个字典包含下面的键值对

| Key   | Value   | Value Type |
| ----- | ------- | ---------- |
| `"O"` | `int`   | 零件朝向编号     |
| `"L"` | `float` | 零件投影长      |
| `"W"` | `float` | 零件投影宽      |
| `"H"` | `float` | 零件高度       |
| `"S"` | `float` | 零件支撑材料体积   |
> 修改时间：2025年1月11日22:05:06

#### 3.1.1.6 `MetaData`

总算例数据类。包括机器参数、工艺参数、算例参数、零件参数。并提供一些帮助函数。

| 成员                                                                                                | 类型                         | 解释               |
| ------------------------------------------------------------------------------------------------- | -------------------------- | ---------------- |
| `self.max_part_type`                                                                              | `int`                      | 允许最大零件种类数        |
| `self.max_orientation_num`                                                                        | `int`                      | 允许最大零件朝向数        |
| `self.instance`                                                                                   | `Inatance`                 | 算例类              |
| `self.machine`                                                                                    | `Machine`                  | 机器类              |
| `self.parts`                                                                                      | `List[Part]`               | 算例中所有零件种类的数据     |
| `self.mask_mtx`                                                                                   | `Tensor`                   | 掩码矩阵（用于获取智能体动作）  |
| `load(self, instance: Instance, machine: Machine, process: Process, parts: List[Part],) -> None:` | `Callable[[...], None]`    | 将算例、机器等类打包       |
| `get_from_dict(self, d: dict) -> None:`                                                           | Deprecated                 | \                |
| `part_vec(self, part_id: int) -> Tensor:`                                                         | `Callable[[int], Tensor]`  | 获取对应种类零件的状态向量    |
| `mask_vec(self, part_id: int) -> Tensor:`                                                         | `Callable[[int], Tensor]`  | 获取对应种类零件的掩码矩阵    |
| `init_state(self) -> Tensor:`                                                                     | `Callable[[None], Tensor]` | 获取所有零件的初始状态矩阵    |
| `mask_matrix(self) -> Tensor:`                                                                    | `Callable[[None], Tensor]` | 获取该算例所有零件当前的掩码矩阵 |

**`part_vec(self, part_id: int) -> Tensor:`**

状态中需要包含json文件中每个零件的所有信息，每个零件的状态向量固定为$1 + 2 + 4 \times 7 = 31$维，其中零件数量一维，零件本身参数（体积和表面积）两维，剩余为每个打印方向的四个数据（长宽高、支撑体积）乘以打印方向总数。并同样使用$0$填补不适用的打印方向和零件。如只有$10$个零件，那么状态矩阵的后$20$行均为零，假设某零件只有$2$个打印方向，则矩阵中该行的后$4 \times 5 = 20$维分量均为零。
为了网络鲁棒性，可暂时将零件类别数和摆放方向开大，例如均以算例中的最大值设定：$v_{\text{parts}} \in \mathbb{R}^{20}$, $v_{\text{ori}} \in \mathbb{R}^{7}$，以适应所有零件类别和摆放方向小于对应值的设定。

**`mask_vec(self, part_id: int) -> Tensor:`**

对于零件类别，构造类别掩码向量$p_{\text{mask}} = [1, 1, \dots, 1, 0, \dots, 0] \in \mathbb{R}^{20}$，其中$1$的个数等于该算例中零件类别数

**`init_state(self) -> Tensor:`**

将 part_vec 堆叠起来

**`mask_matrix(self) -> Tensor:`**

对于每个零件的摆放方向，以类似方法定义$o_{\text{mask}}^{(i)} = [1, 1, \dots, 1, 0, \dots, 0] \in \mathbb{R}^{7}$，然后将所有的向量堆叠起来：$$O_{\text{mask}} = \left[ \begin{matrix}1 & \cdots & 1 & 0 & \cdots &  0\\1 & \cdots & 1 & 0 & \cdots &  0\\\vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\1 & \cdots & 1 & 0 & \cdots &  0\end{matrix} \right] \in \mathbb{R}^{n_{\text{parts}} \times 7}$$由于矩阵$O_{\text{mask}}$的形状会随着算例中类别数量改变而改变，因此可不断填$0$至矩阵形状固定为$20 \times 7$: $$O_{\text{mask}} = \left[ \begin{matrix}1 & \cdots & 1 & 0 & \cdots &  0\\1 & \cdots & 1 & 0 & \cdots &  0\\\vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\1 & \cdots & 1 & 0 & \cdots &  0 \\\vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\ 0 & 0 & 0 & 0 & 0 & 0\end{matrix} \right] \in \mathbb{R}^{20 \times 7}$$

> 修改时间：2025年1月11日22:05:06

### 3.1.2 规划可行解类
#### 3.1.2.1 `Batch`

**功能概述：**
`Batch` 类主要用于管理和操作一批物品的相关信息，它结合了 2D 装箱算法的实际实例、用于神经网络的离散化视图以及已分配到该批物品的信息容器。它提供了各种方法来操作和获取有关这批物品的信息，包括空间信息、物品添加、视图生成、面积和体积计算等，同时还支持将信息输出到文件或显示为图像。

**类属性：**
- `L`：从 `Machine` 类实例中获取的*工作区域长度*，减去了零件距离工作台两侧边缘的最小距离（`machine.build_l - 2 * process.min_distance_part_platform`）。
- `W`：从 `Machine` 类实例中获取的*工作区域宽度*，减去了零件距离工作台两侧边缘的最小距离（`machine.build_w - 2 * process.min_distance_part_platform`）。
- `H`：从 `Machine` 类实例中获取的*工作区域高度*（`machine.build_h`）。
- `machine`：存储 `Machine` 类的实例，方便访问其参数。
- `process`：存储 `Process` 类的实例，方便访问其参数。
- `view_shape`：用于重塑批次视图以进行神经网络处理的形状元组。
- `bin_true`：使用 `newPacker` 创建的装箱算法实例，可能依赖于 `PackingMode.Online` 模式且允许旋转。
- `bin_view`：最终应为固定大小的张量，初始化为 `None`。
- `parts_info`：存储包含已指定方向的部件信息的字典列表。


**方法：**
- `__init__(self, machine: Machine, process: Process, view_shape: tuple) -> None`：
    - **功能**：类的构造函数，初始化 `Batch` 类的实例。
    - **参数**：
        - `machine`：`Machine` 类的实例，用于获取机器的构建参数。
        - `process`：`Process` 类的实例，可能涉及到一些处理过程的参数。
        - `view_shape`：用于神经网络处理的视图形状元组。
    - **实现细节**：
        - 从 `machine` 实例获取 `L`、`W`、`H` 信息并进行边界处理。
        - 存储 `machine` 和 `process` 实例。
        - 初始化 `view_shape`。
        - 创建 `bin_true` 装箱算法实例并添加相应的箱子信息。
        - 初始化 `bin_view` 为 `None`，`parts_info` 为空列表。
- `@property slice_number(self) -> float`：
    - **功能**：计算构建的切片数，根据 `parts_info` 中部件的最大高度除以 `process` 的层厚度向上取整，若批次为空则返回 0。
    - **实现细节**：使用 `max` 函数和 `lambda` 表达式找到 `parts_info` 中 `H` 最大的元素，然后将其除以 `process.layer_thickness` 并向上取整，若批次为空则直接返回 0。
- `get_current_view(self, stretch: bool = True, show: bool = False) -> Tensor`：
    - **功能**：返回批次的当前视图作为神经网络的输入之一，同时包含高度信息。
    - **参数**：
        - `stretch`：是否将视图拉伸到标准大小，默认为 `True`。
        - `show`：是否为可视化添加边框，默认为 `False`。
    - **实现细节**：
        - 创建一个零张量 `grid`，其大小根据 `L` 和 `W` 向上取整。
        - 遍历 `bin_true.rect_list()`，将部件的高度信息添加到 `grid` 中。
        - 根据 `show` 参数添加额外的可视化边框。
        - 根据 `stretch` 参数使用 `torch.nn.functional.interpolate` 进行双线性插值将视图拉伸到 `view_shape` 或保留原始形状。
- `get_rest_area(self) -> float`：
    - **功能**：计算批次的剩余面积。
    - **实现细节**：计算总可用面积（`L * W`）并减去已被部件占用的面积（通过 `map` 和 `lambda` 表达式计算部件的 `L * W` 之和）。
- `get_occupied_ratio(self) -> float`：
    - **功能**：计算部件占用的面积在机器可用面积中的比例，结果范围在 0 到 1 之间。
    - **实现细节**：若批次为空返回 0，否则用 1 减去剩余面积占比。
- `get_total_surface_area(self) -> float`：
    - **功能**：计算该批次中所有部件的总表面积。
    - **实现细节**：使用 `map` 和 `lambda` 表达式将 `parts_info` 中每个部件的表面积相加。
- `get_total_part_volume(self) -> float`：
    - **功能**：计算该批次中所有部件的总体积。
    - **实现细节**：使用 `map` 和 `lambda` 表达式将 `parts_info` 中每个部件的体积相加。
- `get_total_support_volume(self) -> float`：
    - **功能**：计算该批次中所有部件的总支撑体积。
    - **实现细节**：使用 `map` 和 `lambda` 表达式将 `parts_info` 中每个部件的支撑体积相加。
- `add_part(self, part: Part, orientation: int) -> Tuple[Tensor, bool]`：
    - **功能**：将部件添加到批次中，根据添加结果返回相应的视图和布尔值表示是否添加成功。
    - **参数**：
        - `part`：要添加的部件。
        - `orientation`：部件的方向。
    - **实现细节**：
        - 首先检查部件的投影面积是否小于批次的剩余面积，若不满足则返回 `None` 和 `False`。
        - 尝试使用 `allocate_bin_packing_2d` 函数将部件添加到 `bin_true` 中。
        - 若添加成功，更新 `parts_info` 列表并更新 `bin_view`，返回更新后的视图和 `True`，否则不更新并返回原视图和 `False`。
- `empty(self) -> bool`：
    - **功能**：检查批次是否为空。
    - **实现细节**：通过检查 `parts_info` 列表的长度是否小于 1 来判断。
- `show_parts(self, fp)`：
    - **功能**：将 `parts_info` 打印到文件中，同时输出批次的各种信息，如占用比例、剩余面积、切片数等。
    - **参数**：
        - `fp`：文件对象。
    - **实现细节**：
        - 遍历 `parts_info` 和 `bin_true.rect_list()` 并打印部件信息。
        - 打印批次的各种计算信息，如占用比例、剩余面积、切片数等，还包括计算得到的批次时间和能量信息。
- `show_view(self, dir: str) -> None`：
    - **功能**：将批次的视图以矩阵形式作为图像打印出来。
    - **参数**：
        - `dir`：保存图像的目录。
    - **实现细节**：
        - 获取未拉伸且带有显示边框的当前视图。
        - 使用 `matplotlib` 绘制图像，设置 `dpi` 和 `figsize`，显示颜色条和网格，保存图像并尝试关闭图像，删除 `ax` 对象。

**注意事项：**
- 对于 `bin_true` 实例，使用的是 `newPacker` 算法，其 `PackingMode.Online` 模式和旋转功能可能会根据不同的场景有不同的效果，在实际使用中需要根据具体情况进行调整。
- 在调用 `get_current_view` 方法时，根据 `stretch` 和 `show` 参数的不同设置，生成的视图会有所不同，要根据需求选择合适的设置。
- 在添加部件时，`allocate_bin_packing_2d` 函数的行为会直接影响部件是否能成功添加到 `bin_true` 中，需要确保该函数的正确实现和理解其返回结果。
- 在使用 `show_parts` 方法时，文件对象 `fp` 需要在调用该方法前正确创建和打开，以保证信息能正确输出。
- `show_view` 方法使用 `matplotlib` 进行图像绘制，确保系统已正确安装和配置 `matplotlib` 库，并且在保存图像后正确关闭相关资源，以避免资源泄漏。

#### 3.1.2.2 `Solution`

**功能概述：**
`Solution` 类用于管理多个 `Batch` 实例，提供了操作这些 `Batch` 的一系列方法，包括添加 `Batch`、添加部件、获取当前 `Batch` 视图、计算时间和能量，以及展示整个 `Solution` 的信息。

**类属性：**
- `instance_name`：存储实例的名称，默认为 `"None"`。
- `view_shape`：存储视图形状的元组。
- `batches`：存储 `Batch` 实例的列表，初始为空列表。

**方法：**
- `__init__(self, view_shape: Tuple[int, int], instance_name: str = "None") -> None`：
    - **功能**：类的构造函数，初始化 `Solution` 类的实例。
    - **参数**：
        - `view_shape`：视图形状的元组。
        - `instance_name`：实例名称，默认为 `"None"`。
    - **实现细节**：
        - 初始化 `instance_name` 和 `view_shape` 属性。
        - 创建一个空的 `batches` 列表。
- `add_batch(self, machine: Machine, process: Process) -> None`：
    - **功能**：向 `Solution` 中添加一个空的 `Batch`。
    - **参数**：
        - `machine`：`Machine` 类的实例，用于创建 `Batch` 时传递给 `Batch` 的构造函数。
        - `process`：`Process` 类的实例，用于创建 `Batch` 时传递给 `Batch` 的构造函数。
    - **实现细节**：
        - 调用 `Batch` 的构造函数，将 `machine`、`process` 和 `self.view_shape` 作为参数，创建一个新的 `Batch` 实例并添加到 `batches` 列表中。
- `get_batch(self) -> Batch`：
    - **功能**：获取当前的 `Batch`。
    - **实现细节**：
        - 首先使用 `assert` 语句确保 `batches` 列表不为空，若为空会引发异常。
        - 返回 `batches` 列表中的最后一个 `Batch` 实例。
- `get_current_view(self, stretch: bool = True, show: bool = False) -> Tensor`：
    - **功能**：获取当前 `Batch` 的视图。
    - **参数**：
        - `stretch`：是否拉伸视图，默认为 `True`。
        - `show`：是否显示视图，默认为 `False`。
    - **实现细节**：
        - 调用 `get_batch` 方法获取当前 `Batch`，然后调用 `Batch` 的 `get_current_view` 方法，将 `stretch` 和 `show` 参数传递给它。
- `add_part(self, part: Part, orientation: int) -> Tuple[Tensor, bool]`：
    - **功能**：尝试将部件添加到当前 `Batch` 中。
    - **参数**：
        - `part`：要添加的部件。
        - `orientation`：部件的方向。
    - **实现细节**：
        - 调用 `get_batch` 方法获取当前 `Batch`，然后调用 `Batch` 的 `add_part` 方法将部件添加到其中，并返回相应的结果。
- `calculate_time(self) -> float`：
    - **功能**：计算直到现在该 `Solution` 所需的时间。
    - **实现细节**：
        - 若 `Solution` 为空（即 `batches` 列表为空或所有 `Batch` 都为空），返回 0。
        - 否则，使用 `map` 和 `lambda` 表达式计算每个 `Batch` 的时间并求和，其中调用 `calculate_batch_time` 函数。
- `calculate_energy(self) -> float`：
    - **功能**：计算直到现在该 `Solution` 所需的能量。
    - **实现细节**：
        - 若 `Solution` 为空（即 `batches` 列表为空或所有 `Batch` 都为空），返回 0。
        - 否则，使用 `map` 和 `lambda` 表达式计算每个 `Batch` 的能量，调用 `calculate_batch_energy` 函数，并将结果求和后除以 `1e6`。
- `show(self, out_dir: str = f"./solution/") -> None`：
    - **功能**：展示 `Solution` 中所有 `Batch` 的部件列表和视图信息。
    - **参数**：
        - `out_dir`：输出目录，默认为 `"./solution/"`。
    - **实现细节**：
        - 首先检查 `out_dir` 是否存在，若不存在则创建该目录。
        - 打开 `contains.txt` 文件，打印实例名称、解决方案的时间和能量。
        - 遍历 `batches` 列表，为每个 `Batch` 调用 `show_parts` 方法将部件信息打印到文件中，并调用 `show_view` 方法将视图保存为 `batch_<编号>.jpg` 图像。
- `empty(self) -> bool`：
    - **功能**：检查 `Solution` 是否为空。
    - **实现细节**：
        - 检查 `batches` 列表的长度是否小于 1 或者 `batches` 列表中的所有 `Batch` 实例是否都为空，使用 `all` 函数和 `lambda` 表达式进行判断。


**注意事项：**
- 在调用 `get_batch` 方法时，确保 `batches` 列表不为空，否则会引发 `AssertionError`，可以在调用之前使用 `empty` 方法进行检查。
- `add_batch` 方法会添加一个新的 `Batch` 实例，需要注意 `Machine` 和 `Process` 实例的正确传递，它们会影响 `Batch` 的属性和行为。
- `calculate_time` 和 `calculate_energy` 方法依赖于 `calculate_batch_time` 和 `calculate_batch_energy` 函数，确保这些函数已正确实现和调用。
- `show` 方法会在指定目录下创建文件和目录，确保有相应的权限进行文件和目录操作，同时要注意存储的文件和图像的信息，避免信息丢失或覆盖。

### 3.1.4 能量函数和时间函数
#### 3.1.4.1 `calculate_batch_time`
#### 3.1.4.2 `calculate_batch_energy`

### 3.1.3 二维装箱算法

### 3.1.4 强化学习环境

**类功能概述**
`SingleSLMEnv` 类用于构建SLM机器优化的强化学习环境。它负责管理环境的状态、动作空间，处理零件分配和装箱操作，并根据操作结果计算奖励和更新环境状态，在训练和测试阶段与外部数据和模型进行交互，为强化学习算法提供运行环境。

**构造函数 `__init__`**
- **功能**：初始化 `SingleSLMEnv` 类的实例，设置环境的各种参数和初始状态。
- **参数**：
    - `in_path`（`str`）：训练阶段为实例目录路径，测试阶段为 `.json` 文件路径，用于加载数据。
    - `device`（`torch.device`）：指定计算设备。
    - `phase`（`Literal["Train", "Test"]`，默认 `"Train"`）：环境所处阶段，决定数据加载方式。
    - `view_shape`（`Tuple[int, int]`，默认 `(224, 224)`）：用于神经网络处理的视图形状。
    - `max_part_type`（`int`，默认 `20`）：可接受的最大零件类型数。
    - `max_orientation_num`（`int`，默认 `7`）：可接受的最大零件方向数。
    - `seed`（`float`，默认 `0`）：随机种子。
    - `**kwargs`：其他可选参数。
- **实现细节**：
    - 设置随机种子，初始化设备、名称、阶段、路径、视图形状等属性。
    - 根据阶段加载数据并转换为状态，创建 `Solution` 实例并添加初始批次。
    - 初始化当前状态、参考标准和环境空间。

**方法**
* `get_unavailable_mask`
    - **功能**：获取零件类型的不可用掩码，根据零件剩余数量和合法方向判断。
    - **实现细节**：通过对零件状态和掩码矩阵的计算生成掩码向量。
- `check_geo_constraints`
    - **功能**：目前未实现，用于检查分配是否满足几何约束（若强化学习模型需确定零件位置和方向时使用）。
* `reset`
    - **功能**：根据环境阶段重置环境，训练阶段重新初始化，测试阶段退出。
    - **返回值**：`self.curr_state`（当前状态）和 `self.load_path`（加载路径）。
- `update_state`
    - **功能**：在成功装箱后根据零件 ID 和视图更新当前状态。
    - **参数**：
        - `view`（`Tensor`）：新的视图张量。
        - `part_id`（`int`）：已分配零件的 ID。
- `done`
    - **功能**：检查是否所有零件都已分配。
    - **返回值**：若所有零件都已分配则返回 `True`，否则返回 `False`。
* `step`
    - **功能**：根据输入动作更新环境，执行零件分配、状态更新、奖励计算等操作，并返回更新后的状态、奖励和环境终止标志。
    - **参数**：
        - `action`（`Tensor`）：包含零件选择和方向信息的动作张量。
    - **返回值**：
        - `self.curr_state`（更新后的当前状态）。
        - `reward`（计算得到的奖励）。
        - `terminated`（是否终止）。
        - `truncated`（是否截断）。
        - `info`（其他信息，目前为 `None`）。
    - **实现细节**：
        - 对动作进行处理，选择零件 ID 和方向 ID。
        - 尝试分配零件，根据分配结果更新状态和环境标志。
        - 计算奖励，基于能量成本的变化。

**注意事项**
- 在 `step` 方法中，零件分配和方向选择的逻辑较为复杂，涉及到多个条件判断和循环，需确保对零件状态、掩码矩阵和 `Solution` 类的操作正确无误。
- `check_geo_constraints` 方法未实现，若后续需要使用该功能，需按照特定的几何约束规则进行编写。
- 在训练和测试阶段的切换中，`reset` 方法的行为不同，需注意在不同场景下的正确使用，尤其是测试阶段直接退出的设定。

## 3.2 并行一维装箱模型`
### 3.2.1 规划可行解类
#### 3.2.1.1 `Batch`

**功能概述：**
`Batch` 类主要用于管理和操作一批物品的相关信息，它结合了 1D 装箱算法的实际实例、用于神经网络的离散化视图以及已分配到该批物品的信息容器。它提供了各种方法来操作和获取有关这批物品的信息，包括占用信息、物品添加、视图生成、面积和体积计算等，同时还支持将信息输出到文件或显示为图像。

**类属性：**
- `L`：从 `Machine` 类实例中获取的*工作区域长度*，减去了零件距离工作台两侧边缘的最小距离（`machine.build_l - 2 * process.min_distance_part_platform`）。
- `W`：从 `Machine` 类实例中获取的*工作区域宽度*，减去了零件距离工作台两侧边缘的最小距离（`machine.build_w - 2 * process.min_distance_part_platform`）。
- `H`：从 `Machine` 类实例中获取的*工作区域高度*（`machine.build_h`）。
- `machine`：存储 `Machine` 类的实例，方便访问其参数。
- `process`：存储 `Process` 类的实例，方便访问其参数。
- `view_shape`：被删除 #DIFF 
- `bin_true`：被删除 #DIFF 
- `bin_view`：被删除 #DIFF 
- `parts_info`：存储包含已指定方向的部件信息的字典列表。


**方法：**
- `__init__(self, machine: Machine, process: Process) -> None`： #DIFF
    - **功能**：类的构造函数，初始化 `Batch` 类的实例。
    - **参数**：
        - `machine`：`Machine` 类的实例，用于获取机器的构建参数。
        - `process`：`Process` 类的实例，可能涉及到一些处理过程的参数。
    - **实现细节**：
        - 从 `machine` 实例获取 `L`、`W`、`H` 信息并进行边界处理。
        - 存储 `machine` 和 `process` 实例。
        - 初始化 `view_shape`。
        - 初始化`parts_info` 为空列表。 #DIFF 
- `@property slice_number(self) -> float`：
    - **功能**：计算构建的切片数，根据 `parts_info` 中部件的最大高度除以 `process` 的层厚度向上取整，若批次为空则返回 0。
    - **实现细节**：使用 `max` 函数和 `lambda` 表达式找到 `parts_info` 中 `H` 最大的元素，然后将其除以 `process.layer_thickness` 并向上取整，若批次为空则直接返回 0。
- `get_current_view(self, stretch: bool = True, show: bool = False) -> Tensor`：
    - 如果 show 为真，则返回当前占比，否则返回二元组，第一个元素为剩余表面积，第二个元素为占比
- `get_rest_area(self) -> float`：
    - **功能**：计算批次的剩余面积。
    - **实现细节**：计算总可用面积（`L * W`）并减去已被部件占用的面积（通过 `map` 和 `lambda` 表达式计算部件的 `L * W` 之和）。
- `get_occupied_ratio(self) -> float`：
    - **功能**：计算部件占用的面积在机器可用面积中的比例，结果范围在 0 到 1 之间。
    - **实现细节**：若批次为空返回 0，否则用 1 减去剩余面积占比。
- `get_total_surface_area(self) -> float`：
    - **功能**：计算该批次中所有部件的总表面积。
    - **实现细节**：使用 `map` 和 `lambda` 表达式将 `parts_info` 中每个部件的表面积相加。
- `get_total_part_volume(self) -> float`：
    - **功能**：计算该批次中所有部件的总体积。
    - **实现细节**：使用 `map` 和 `lambda` 表达式将 `parts_info` 中每个部件的体积相加。
- `get_total_support_volume(self) -> float`：
    - **功能**：计算该批次中所有部件的总支撑体积。
    - **实现细节**：使用 `map` 和 `lambda` 表达式将 `parts_info` 中每个部件的支撑体积相加。
- `add_part(self, part: Part, orientation: int) -> Tuple[Tensor, bool]`：
    - **功能**：将部件添加到批次中，根据添加结果返回相应的视图和布尔值表示是否添加成功。
    - **参数**：
        - `part`：要添加的部件。
        - `orientation`：部件的方向。
    - **实现细节**：
        - 检查部件的投影面积是否小于批次的剩余面积
            - 若不满足则返回 `None` 和 `False`。
            - 若满足，更新 `parts_info` 列表并更新 `bin_view`，返回更新后的视图和 `True`，否则不更新并返回原视图和 `False`。
- `empty(self) -> bool`：
    - **功能**：检查批次是否为空。
    - **实现细节**：通过检查 `parts_info` 列表的长度是否小于 1 来判断。

- `show_parts(self, fp)`：
    - **功能**：将 `parts_info` 打印到文件中，同时输出批次的各种信息，如占用比例、剩余面积、切片数等。
    - **参数**：
        - `fp`：文件对象。
    - **实现细节**：
        - 遍历 `parts_info` 和 `bin_true.rect_list()` 并打印部件信息。
        - 打印批次的各种计算信息，如占用比例、剩余面积、切片数等，还包括计算得到的批次时间和能量信息。
- `show_view(self, dir: str) -> None`：
    - **功能**：什么都不做
    - 如果写好其他部分之后这个函数没有被引用就把他删了

#### 3.2.1.2 `Solution`

**功能概述：**
`Solution` 类用于管理多个 `Batch` 实例，提供了操作这些 `Batch` 的一系列方法，包括添加 `Batch`、添加部件、获取当前 `Batch` 视图、计算时间和能量，以及展示整个 `Solution` 的信息。

**类属性：**
- `instance_name`：存储实例的名称，默认为 `"None"`。
- `batches`：存储 `Batch` 实例的列表，初始为空列表。

**方法：**
- `__init__(self, instance_name: str = "None", batch_num: int = 100) -> None`：
    - **功能**：类的构造函数，初始化 `Solution` 类的实例。
    - **参数**：
        - `instance_name`：实例名称，默认为 `"None"`。
        - `batch_num`：本Solution中的所有Batch数量
    - **实现细节**：
        - 初始化 `instance_name` 属性和 `batch_num` 属性。
        - 创建一个空的 `batches` 列表。
- `init_batches(self, machine: Machine, process: Process) -> None`：
    - **功能**：初始化，向 `Solution` 中重置 `self.batch_num` 个空的 `BatchParallel1D`。
    - **参数**：
        - `machine`：`Machine` 类的实例，用于创建 `Batch` 时传递给 `Batch` 的构造函数。
        - `process`：`Process` 类的实例，用于创建 `Batch` 时传递给 `Batch` 的构造函数。
- `def get_batch(self, idx: None|int) -> BatchParallel1D:`：
    - **功能**：获取当前的 `Batch`。
    - **参数**：
        - `idx`：整数或 `None`，表示索取的 Batch 的索引
    - **实现细节**：
        - 首先使用 `assert` 语句确保 `batches` 列表不为空，若为空会引发异常。
        - 如果 `idx` 的值为 `None`，则返回 `batches` 列表中的最后一个 `Batch` 实例。
        - 否则按照 `idx` 的值索引对应的 Batch。
- `def get_current_view(self, idx: None|int, show: bool = False) -> Tensor`：
    - **功能**：获取当前所有 Batch 的视图。
    - **参数**：
        - `show`：是否显示视图，默认为 `False`。
    - **实现细节**：
        - 遍历 `self.batches` 中的所有 Batch，并调用各自的 `get_current_view()` 方法，最后将得到的 Tensor 堆叠起来，最终输出的形状为 `[self.batch_num, 2]`
- `add_part(self, part: Part, orientation: int, idx: None|int) -> Tuple[Tensor, bool]`：
    - **功能**：尝试将部件添加到当前 `Batch` 中。
    - **参数**：
        - `part`：要添加的部件。
        - `orientation`：部件的方向。
        - `idx`：整数或 `None`，表示索取的 Batch 的索引
    - **实现细节**：
        - 调用 `get_batch` 方法获取目标 Batch ，然后调用 `BatchParallel1D`  的 `add_part` 方法将部件添加到其中，并返回相应的结果。
- `calculate_time(self) -> float`：
    - **功能**：计算直到现在该 `Solution` 所需的时间。
    - **实现细节**：
        - 若 `Solution` 为空（即 `batches` 列表为空或所有 `Batch` 都为空），返回 0。
        - 否则，使用 `map` 和 `lambda` 表达式计算每个 `Batch` 的时间并求和，其中调用 `calculate_batch_time` 函数。
- `calculate_energy(self) -> float`：
    - **功能**：计算直到现在该 `Solution` 所需的能量。
    - **实现细节**：
        - 若 `Solution` 为空（即 `batches` 列表为空或所有 `Batch` 都为空），返回 0。
        - 否则，使用 `map` 和 `lambda` 表达式计算每个 `Batch` 的能量，调用 `calculate_batch_energy` 函数，并将结果求和后除以 `1e6`。
- `show(self, out_dir: str = f"./solution/") -> None`：
    - **功能**：展示 `Solution` 中所有 `Batch` 的部件列表和视图信息。
    - **参数**：
        - `out_dir`：输出目录，默认为 `"./solution/"`。
    - **实现细节**：
        - 首先检查 `out_dir` 是否存在，若不存在则创建该目录。
        - 打开 `contains.txt` 文件，打印实例名称、解决方案的时间和能量。
        - 遍历 `batches` 列表，获取每个batch的占用情况，然后输出占用条形图
### 3.2.2 强化学习环境 `SingleSLMEnvParallel1D`

**功能概述**
`SingleSLMEnvParallel1D` 类是 `SingleSLMEnv` 的子类，用于实现一维并行的选择性激光熔化（SLM）机器优化的强化学习环境。它在父类的基础上，对环境状态、动作空间以及 `step` 操作进行了修改，以适应一维并行的场景，处理零件在多个批次中的并行分配和状态更新，同时计算相应的奖励和环境终止条件。

**构造函数** `__init__`
- **功能**：初始化 `SingleSLMEnvParallel1D` 类的实例，为一维并行环境设置各种参数和初始状态。
- **参数**：
    - `in_path`（`str`）：训练阶段为实例目录路径，测试阶段为 `.json` 文件路径，用于加载数据。
    - `phase`（`Literal["Train", "Test"]`，默认 `"train"`）：环境所处阶段，决定数据加载方式。
    - `max_part_type`（`int`，默认 `20`）：可接受的最大零件类型数。
    - `max_orientation_num`（`int`，默认 `7`）：可接受的最大零件方向数。
    - `max_batch_num`（`int`，默认 `20`）：可接受的最大批次数量。
    - `seed`（`float`，默认 `0`）：随机种子。
    - `**kwargs`：其他可选参数。
- **实现细节**：
    - 初始化随机种子、名称、阶段、路径等属性。
    - 根据阶段加载数据并转换为状态，使用 `SolutionParallel1D` 实例初始化批次。
    - 初始化当前状态、参考标准和环境空间，状态是三个张量的堆叠。

**方法** 
* `update_state`
    - **功能**：在成功分配零件后根据零件 ID 和视图更新当前状态。
    - **参数**：
        - `view`（`Tensor`）：新的视图张量。
        - `part_id`（`int`）：已分配零件的 ID。
    - **实现细节**：
        - 保存旧状态，检查零件数量，更新零件状态矩阵，更新当前状态。
* `done`
    - **功能**：检查是否所有零件都已分配。
    - **实现细节**：根据零件状态矩阵判断是否还有未分配的零件。
* `slice_action`
    - **功能**：将输入动作拆分为三个部分，分别为零件分布、方向分布和批次分布。
    - **返回值**：三个张量组成的元组，分别表示零件分布、方向分布和批次分布。
* `get_distribution_masks`
    - **功能**：获取分布掩码，目前未完成，可能用于过滤不合法的动作。
    - **返回值**：三个张量组成的列表，分别表示零件掩码、方向掩码和批次掩码。
     `step`
    - **功能**：根据输入动作更新环境，尝试将零件分配到不同批次和方向，更新状态、计算奖励和检查环境终止。
    - **参数**：
        - `action`（`Tensor`）：包含零件分布、方向分布和批次分布的动作张量。
    - **返回值**：
        - `self.curr_state`（更新后的当前状态）。
        - `reward`（计算得到的奖励）。
        - `terminated`（是否终止）。
        - `truncated`（是否截断）。
        - `info`（其他信息，目前为 `None`）。
    - **实现细节**：
        - 调用 `slice_action` 拆分动作，遍历可能的零件、方向和批次进行分配尝试。
        - 根据分配结果更新状态，计算奖励，考虑惩罚项，更新参考标准。

**注意事项**
- `get_distribution_masks` 方法尚未确定，使用时需注意其功能可能未完整实现。
- 在 `step` 方法中，零件分配涉及到多个嵌套循环，性能可能会受到一定影响，尤其是在零件、方向和批次数量较大时。
- 确保 `SolutionParallel1D` 类的 `add_part` 方法能正确处理传递的参数，并返回所需结果（如视图、是否成功和惩罚值）。
- 注意 `phase` 属性对数据加载的影响，在训练和测试阶段有不同的操作，确保路径的正确性和处理逻辑的适用性。
- 奖励计算包含惩罚项，确保惩罚项的计算逻辑符合预期的环境行为。

## 3.3 并行二维装箱模型
### 3.3.1 规划可行解类
#### 3.3.1.1 `Batch`
#### 3.3.1.2 `Solution`
### 3.3.2 强化学习环境



