# """研究 detach 对 计算图的影响。 """
"""如果想要计算各个Variable的梯度，只需调用根节点variable的backward方法，autograd会自动沿着计算图反向传播，计算每一个叶子节点的梯度。"""
# import torch
# x = torch.tensor([1.0,2.0],requires_grad=True)
# x1 = torch.tensor([3.0,4.0],requires_grad=True)
# y1 = x**2
# y2 = y1.detach()*2
# y3 = y2*2
# y4 = y1 +y3+y2
# y5 = y4**2 + x1
#
# print(y1, y1.requires_grad) # grad_fn=<PowBackward0>
# print(y2, y2.requires_grad) # None
# print(y3, y3.requires_grad) # None
# print(y4, y4.requires_grad) # grad_fn=<AddBackward0>

# x.grad: tensor([ 28., 224.])
# x1.grad: tensor([ 28., 224.])
# y5.backward(torch.ones(y3.shape))
# print('grad')
# print('x.grad:',x.grad) # 只有叶子节点有 grad 值。
# print('x1.grad:',x1.grad)
# print('y1.grad:', y1.grad) # 都是 x 产生的复合函数。即都是叶子节点。
# print('y2.grad:', y2.grad) # 非叶子节点grad计算完之后自动清空
# print('y3.grad:', y3.grad)
# print('y4.grad:', y4.grad)

""" 研究 将 原计算图中叶子节点剥离出来并重新计算，是否对原 计算图的 反向传播有影响。"""
""" PyTorch使用的是动态图，它的计算图在每次前向传播时都是从头开始构建，所以它能够使用Python控制语句（如for、if等）根据需求创建计算图。 """
import torch
# 简化网络，没有 detach 影响
x = torch.tensor([1.0,2.0],requires_grad=True)
x1 = torch.tensor([3.0,4.0],requires_grad=True)
y1 = x**2
y2 = y1*2
y3 = y2*2
y4 = y1 +y3+y2
y5 = y4**2 + x1

print('剥离前的 计算图状态')
print(x, x.requires_grad,x.grad)
print(x1, x1.requires_grad,x1.grad)
print(y1, y1.requires_grad,y1.grad) # 非叶子节点grad计算完之后自动清空
print(y2, y2.requires_grad,y2.grad)
print(y3, y3.requires_grad,y3.grad)
print(y4, y4.requires_grad,y4.grad)
# 如果 y5.backward(torch.ones(y5.shape)) ，输出：
# tensor([1., 2.], requires_grad=True) True tensor([ 196., 1568.])
# tensor([3., 4.], requires_grad=True) True tensor([1., 1.])
# tensor([1., 4.], grad_fn=<PowBackward0>) True None
# tensor([2., 8.], grad_fn=<MulBackward0>) True None
# tensor([ 4., 16.], grad_fn=<MulBackward0>) True None
# tensor([ 7., 28.], grad_fn=<AddBackward0>) True None



y_1 = y3
# 将 叶子节点从 计算图中剥离出来。
p_list = [x,x1,y1,y2,y3,y4,y5]
for p in p_list:
    if p.grad is not None:  # 计算的时候，只有 叶子节点的  p.grad 为 not None。
        # 当我们再训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；
        # 或者只 训练部分分支网络，并不让其梯度对主网络的梯度造成影响，
        # 这时候我们就需要使用detach()函数来切断一些分支的反向传播
        p.grad.detach_()
        p.grad.zero_()

# 剥离后的各个计算图的节点状态：
# 无论是否经历：y5.backward(torch.ones(y5.shape))，输出结果不变。即 grad_fn 自动记录 中间节点的 计算过程。即 动态构建 计算图 。
print('detach_and_zero_grad')
print('x.grad:',x,x.grad)
print('x1.grad:',x1,x1.grad)
print('y1.grad:', y1,y1.grad) # 都是 x 产生的复合函数。即都是叶子节点。
print('y2.grad:', y2,y2.grad)
print('y3.grad:', y3,y3.grad)
print('y4.grad:', y4,y4.grad)
print('y5.grad:', y5,y5.grad)

# 输出：
# detach_and_zero_grad
# x.grad: tensor([1., 2.], requires_grad=True) None
# x1.grad: tensor([3., 4.], requires_grad=True) None
# y1.grad: tensor([1., 4.], grad_fn=<PowBackward0>) None
# y2.grad: tensor([2., 8.], grad_fn=<MulBackward0>) None
# y3.grad: tensor([ 4., 16.], grad_fn=<MulBackward0>) None
# y4.grad: tensor([ 7., 28.], grad_fn=<AddBackward0>) None
# y5.grad: tensor([ 52., 788.], grad_fn=<AddBackward0>) None
# 可以看到剥离后的 点中，计算图依然存在。 只是梯度值消失。 梯度值依然可以根据 自身值 与 计算图 进行梯度计算。

# x = torch.tensor([1.0,2.0],requires_grad=True)
# x1 = torch.tensor([3.0,4.0],requires_grad=True) # 如果直接用 上计算图使用过的 叶子节点，则会将 原计算图的 叶子节点破坏掉，导致 backward  叶子节点的 grad == None，虽然进行了p.grad.detach_() p.grad.zero_()

# x_det = torch.tensor([1.0,2.0],requires_grad=True)
# x1_det = torch.tensor([3.0,4.0],requires_grad=True)
# x = x_det
# x1 = x1_det # 名字相同都会出现 x.grad == None

"""实验 当 叶子节点变量名不同，中间变量名相同，不会影响 backward 的结果"""
# 只有这时（叶子节点的变量名不同），才会在 y5.backward 时，回溯到 已经记录的 x,x1 的叶子节点上。
# x_det = torch.tensor([1.0,2.0],requires_grad=True)
# x1_det = torch.tensor([3.0,4.0],requires_grad=True)
# # 与上一个计算图的计算方式不一样
# y1 = x_det*2
# y2 = y1**2
# y3 = y2*2
# y4 = y1 +y3+y2
# y6 = y4**2 + x1_det*2
#
# y_2 = y3
#
# # 在将计算图中叶子节点剥离后，仍然可以反向传播。即 结果 y5 的 grad_fn 保存了 其后向传播的函数，与 可追溯的中间计算过程(参数名可被其他计算图利用，即y1,y2,y3可以多图使用。)，
# # 和 叶子节点的名字(x,x1)。（叶子节点被重置 或者 参与其他计算图，会导致后向传播无法继续。） 应该是 在同一存储空间，产生冲突。 考虑 clean_image_l.detach().clone()
# y5.backward(torch.ones(y5.shape))
# print('y5 在 detach_and_zero 后，依然对原结果进行 backward')
# print('grad')
# print('x.grad:',x,x.grad) # x.grad: tensor([1., 2.], requires_grad=True) tensor([ 196., 1568.])
# print('x1.grad:',x1,x1.grad) # x1.grad: tensor([3., 4.], requires_grad=True) tensor([1., 1.])
# print('y1.grad:', y1,y1.grad) # 都是 x 产生的复合函数。即都是叶子节点。
# print('y2.grad:', y2,y2.grad)
# print('y3.grad:', y3,y3.grad)
# print('y4.grad:', y4,y4.grad)

# 从结果可以看到 上下两个的
#
# y6.backward(torch.ones(y5.shape))
# print('grad')
# print('x_det.grad:',x_det,x_det.grad) # x_det.grad: tensor([1., 2.], requires_grad=True) tensor([ 728., 5200.])
# print('x1_det.grad:',x1_det,x1_det.grad) # x1_det.grad: tensor([3., 4.], requires_grad=True) tensor([2., 2.])
# print('y1.grad:', y1,y1.grad) # 都是 x 产生的复合函数。即都是叶子节点。
# print('y2.grad:',y2, y2.grad)
# print('y3.grad:', y3,y3.grad)
# print('y4.grad:', y4,y4.grad)
#
# print('y_1 == y_2 ,{}'.format(y_2==y_1))
# print('y3 == y3 ,{}'.format(y3==y3))
# # y6.backward(torch.ones(y3.shape))  同样会计入 重复图的后向传播 （Trying to backward through the graph a second time）

"""使用 clone 操作，替换原 x1,x 等 叶子节点 ，即换一个存储空间。"""
x_ = x.detach().clone() # 默认 required_grad 为 false
x_1 = x1.detach().clone() # 可以通过xxx.requires_grad_()将默认的Flase修改为True
x_.requires_grad_()
x_1.requires_grad_()

y1 = x_*2
y2 = y1**2
y3 = y2*2
y4 = y1 +y3+y2
y6 = y4**2 + x_1*2

y_2 = y3

# 在将计算图中叶子节点剥离后，仍然可以反向传播。即 结果 y5 的 grad_fn 保存了 其后向传播的函数，与 可追溯的中间计算过程(参数名可被其他计算图利用，即y1,y2,y3可以多图使用。)，
# 和 叶子节点的名字(x,x1)。（叶子节点被重置 或者 参与其他计算图，会导致后向传播无法继续。） 应该是 在同一存储空间，产生冲突。clean_image_l.detach().clone()
y5.backward(torch.ones(y5.shape))
print('y5 在 detach_and_zero 后，依然对原结果进行 backward')
print('grad')
print('x.grad:',x,x.grad) # x.grad: tensor([1., 2.], requires_grad=True) tensor([ 196., 1568.])
print('x1.grad:',x1,x1.grad) # x1.grad: tensor([3., 4.], requires_grad=True) tensor([1., 1.])
print('y1.grad:', y1,y1.grad)
print('y2.grad:', y2,y2.grad)
print('y3.grad:', y3,y3.grad)
print('y4.grad:', y4,y4.grad)

y6.backward(torch.ones(y6.shape))
print('y5 在 detach_and_zero 后，依然对原结果进行 backward')
print('grad')
print('x_.grad:',x_,x_.grad) # x_.grad: tensor([1., 2.], requires_grad=True) tensor([ 728., 5200.])
print('x_1.grad:',x_1,x_1.grad) # x_1.grad: tensor([3., 4.], requires_grad=True) tensor([2., 2.])
print('y1.grad:', y1,y1.grad)
print('y2.grad:', y2,y2.grad)
print('y3.grad:', y3,y3.grad)
print('y4.grad:', y4,y4.grad)

"""总结如下 ： 
            1. 当使用 detach 将中间节点从计算图中脱离时，后续的计算中，计算图不在记录其 grad_fn ，即为 None ，也即 loss.backward 时，当追溯到 与其相关的中间计算时，此节点 会自动被忽略。不再计算其梯度。
            2. 当建立好一个计算图后，如果，将其叶子节点全部 detach 后，再利用叶子节点重新建立新的计算图时， 只要不改变 叶子节点在 detach 前的 存储位置，如利用 detach().clone() ，前一个计算图依然可以进行 loss.backward。并计算梯度。
            3. 每一个计算图的建立，中间节点都是零时创建，尽管使用相同参数名。存储位置不重复利用。这也是虽然创建了几个新的计算图，但是每一个都可以进行自己的 loss.backward 回溯。"""

"""在PyTorch中计算图的特点可总结如下：
    1.autograd根据用户对variable的操作构建其计算图。对变量的操作抽象为Function。
    2.对于那些不是任何函数(Function)的输出，由用户创建的节点称为叶子节点，叶子节点的grad_fn为None。叶子节点中需要求导的variable，具有AccumulateGrad标识，因其梯度是累加的。
    3.variable默认是不需要求导的，即requires_grad属性默认为False，如果某一个节点requires_grad被设置为True，那么所有依赖它的节点requires_grad都为True。(如： a = b +c ，则为 a 依赖  b 和 c ， 即计算图中，一个节点requires_grad 为 True，所有建立在其之上的运算，requires_grad 都为 True.)
    4.variable的volatile属性默认为False，如果某一个variable的volatile属性被设为True，那么所有依赖它的节点volatile属性都为True。volatile属性为True的节点不会求导，volatile的优先级比requires_grad高。
    5.多次反向传播时，梯度是累加的。反向传播的中间缓存会被清空，为进行多次反向传播需指定retain_graph=True来保存这些缓存。
    6.非叶子节点的梯度计算完之后即被清空，可以使用autograd.grad或hook技术获取非叶子节点的值。
    7.variable的grad与data形状一致，应避免直接修改variable.data，因为对data的直接操作无法利用autograd进行反向传播
    8.反向传播函数backward的参数grad_variables可以看成链式求导的中间结果，如果是标量，可以省略，默认为1
    9.PyTorch采用动态图设计，可以很方便地查看中间层的输出，动态的设计计算图结构。
"""

""" loss.backward(grad_loss ) 的情况 """
def f(x):
    '''计算y'''
    y = x**2 * torch.exp(x)
    return y
x = torch.randn(3,4, requires_grad = True)
y = f(x)
print(y)
# 结果：
# tensor([[1.6681e-01, 2.9650e+00, 9.1634e+00, 4.9143e-01],
#         [7.4560e-02, 3.3950e+00, 1.8273e+01, 2.8271e-01],
#         [7.8892e+00, 4.2957e-04, 4.1004e-01, 1.2708e-02]], grad_fn=<MulBackward0>)

# y.backward() 如果以此执行 backward()，无 x.grad 结果。 即 backward() 函数中，参数 grad_variables 出错。
# 参数 grad_variables 应为 结果值 对 当前值 的 导数。 相当于 dy/dy。 结果为 1 ，但形状为 y.size()

y.backward(torch.ones(y.size()),retain_graph=True) # gradient形状与y一致
# t.ones(y.size())相当于grad_variables：形状与variable一致，对于y.backward()，grad_variables相当于链式法则 𝑑𝑧𝑑𝑥=𝑑𝑧𝑑𝑦×𝑑𝑦𝑑𝑥 中的 𝐝𝐳𝐝𝐲 。
# 不能单独运行两次 RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed. Specify retain_graph=True when calling backward the first time.
print(x.grad)

# 如果 ：
z = torch.sum(y) # 单独的 sum函数 不行，需要 使用 torch.sum() 因为 还有 grad_fn 等后向传播函数。
z.backward() # 此时的 grad_variables =dz/dz ，因为 z 是标量， dz/dz =1 ,可以省略不写。
print(x.grad)

""" 计算图 的 研究。"""
import torch as t
x = t.ones(1)
b = t.rand(1, requires_grad = True)
w = t.rand(1, requires_grad = True)
y = (w*3) * x**2 # 等价于y=w.mul(x)

# y.requires_grad = False
# RuntimeError: you can only change requires_grad flags of leaf variables.

z = y + b # 等价于z=y.add(b)

y.backward()
# y.grad 中间节点 y 无 grad
print(w.grad)
print('x.grad',x.grad)
print(b.grad)

print('x.requires_grad：',x.requires_grad)
print('b.requires_grad：',b.requires_grad)
print('w.requires_grad：',w.requires_grad)
print('y.requires_grad：',y.requires_grad) # 虽然未指定 y.requires_grad为True，但因为 y.grad 的计算 需要 w ，而 w.requires_grad=True

""" next_functions保存grad_fn的输入，是一个tuple，tuple的元素也是Function
    第一个是y，它是乘法(mul)的输出，所以对应的反向传播函数y.grad_fn是MulBackward
    第二个是b，它是叶子节点，由用户创建，grad_fn为None，但是有
"""
print('z.grad_fn.next_functions:',z.grad_fn.next_functions) # 计算图中 对应的 后向传播 无环有向图。

# variable的grad_fn对应着和图中的function相对应
print('z.grad_fn.next_functions[0][0] == y.grad_fn :',z.grad_fn.next_functions[0][0] == y.grad_fn)

print('z.grad_fn.next_functions[0][0].next_functions:',z.grad_fn.next_functions[0][0].next_functions)
print('y.grad_fn.next_functions:',y.grad_fn.next_functions)


""" 关闭自动求导功能。
    有些时候我们可能不希望autograd对tensor求导。认为求导需要缓存许多中间结构，增加额外的内存/显存开销，那么我们可以关闭自动求导。对于不需要反向传播的情景（如inference，即测试推理时），
    关闭自动求导可实现一定程度的速度提升，并节省约一半显存，因其不需要分配空间计算梯度。
"""

with t.no_grad():
    x = t.ones(1)
    w = t.rand(1, requires_grad = True)
    y = x * w
# y依赖于w和x，虽然w.requires_grad = True，但是y的requires_grad依旧为False
print('x.requires_grad:',x.requires_grad)
print('w.requires_grad:',w.requires_grad)
print('y.requires_grad:',y.requires_grad) # 可以看到  # y.requires_grad: False
# 或者 通过 t.set_grad_enabled(False) 设置 ，并通过 t.set_grad_enabled(True) 恢复。

"""只要你对需要求导的叶子张量使用了这些操作，马上就会报错"""
"""所谓动态图，就是每次当我们搭建完一个计算图，然后在反向传播结束之后，整个计算图就在内存中被释放了。如果想再次使用的话，必须从头再搭一遍，"""