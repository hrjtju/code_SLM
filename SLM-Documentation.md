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
* 状态（1）现阶段所有箱子中所占用的总底面积及其与箱子总底面积的比值（占用率矩阵）（2）现阶段所有工件的剩余量和所有工件尺寸
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

### 3.1.2 规划可行解类
#### 3.1.2.1 `Batch`
#### 3.1.2.2 `Solution`
### 3.1.4 能量函数和时间函数
#### 3.1.4.1 `calculate_batch_time`
#### 3.1.4.2 `calculate_batch_energy`
### 3.1.3 二维装箱算法
## 3.2 并行一维装箱模型
### 3.2.1 基本数据元素类
#### 3.2.1 `ItemFromJson`
#### 3.2.1.2 `Instance`
#### 3.2.1.3 `Machine`
#### 3.2.1.4 `Process`
#### 3.2.1.5 `Part`
#### 3.2.1.6 `MetaData`
### 3.2.2 规划可行解类
#### 3.2.2.1 `Batch`
#### 3.2.2.2 `Solution```
### 3.2.4 能量函数和时间函数
#### 3.2.4.1 `calculate_batch_time`
#### 3.2.4.2 `calculate_batch_energy`
### 3.2.3 二维装箱算法
## 3.3 并行二维装箱模型
### 3.3.1 基本数据元素类
#### 3.3.1 `ItemFromJson`
#### 3.2.1.2 `Instance`
#### 3.3.1.3 `Machine`
#### 3.3.1.4 `Process`
#### 3.3.1.5 `Part`
#### 3.3.1.6 `MetaData`
### 3.3.2 规划可行解类
#### 3.3.2.1 `Batch`
#### 3.3.2.2 `Solution```
### 3.3.4 能量函数和时间函数
#### 3.3.4.1 `calculate_batch_time`
#### 3.3.4.2 `calculate_batch_energy`

