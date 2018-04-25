# -*- coding: utf-8 -*-
#    base_lr
#    这个参数是用来表示网络的初始学习率的。这个值是一个浮点型实数。
#    lr_policy
#    这个参数是用来表示学习率随着时间是如何变化的。值是字符串，需要加""。学习率变化的可选参数有：
#    “step”——需要设置stepsize。根据gamma参数和stepsize参数来降低学习率，base_lr * gamma ^ (floor(iter / stepsize))。iter是当前迭代次数。学习率每迭代stepsize次变化一次。
#    “multistep”——与step类似，需要设置stepvalue，学习率根据stepvalue进行变化。
#    “fixed”——学习率base_lr保持不变。
#    “inv”——学习率变化公式为base_lr * (1 + gamma * iter) ^ (- power)
#    “exp”——学习率变化公式为base_lr * gamma ^ iter}
#    “poly”——学习率以多项式形式衰减，到最大迭代次数时降为0。学习率变化公式为base_lr * (1 - iter/max_iter) ^ (power)。
#    “sigmoid”——学习率以S型曲线形式衰减，学习率变化公式为base_lr * (1 / (1 + exp(-gamma * (iter - stepsize))))。
#    gamma
#    这个参数表示学习率每次的变化程度，值为实数。
#    stepsize
#    这个参数表示什么时候应该进行训练的下一过程，值为正整数。主要用在lr_policy为step的情况。
#    stepvalue
#    这个参数表示什么时候应该进行训练的下一过程，值为正整数。主要用在lr_policy为multistep的情况。
#    max_iter
#    这个参数表示训练神经网络迭代的最大次数，值为正整数。
#    momentum
#    这个参数表示在新的计算中要保留的前面的权重数量，值为真分数，通常设为0.9。
#    weight_decay
#    这个参数表示对较大权重的惩罚（正则化）因子。值为真分数。
#    This parameter indicates the factor of (regularization) penalization of large weights. This value is a often a real fraction.
#    solver_mode
#    这个参数用来表示求解神经网络的模式——值为CPU or GPU。
#    snapshot
#    这个参数用来表示每迭代多少次就应该保存snapshot的model和solverstate，值为正整数。
#    snapshot_prefix:
#    这个参数用来表示保存snapshot时model和solverstate的前缀，值为带引号的字符串。
#    net:
#    这个参数表示训练网络所在的位置，值为带引号的字符串。
#    test_iter
#    这个参数表示
#    这个参数表示每个test_interval进行多少次test迭代，值为正整数。
#    test_interval
#    这个参数表示什么时候进行数据的测试，值为正整数。
#    display
#    这个参数用来表示什么时候将输出结果打印到屏幕上，值为正整数，表示迭代次数。
#    type
#    这个参数表示训练神经网络采用的反向传播算法，值为带引号的字符串。可选的值有：
#    Stochastic Gradient Descent “SGD”——随机梯度下降，默认值。
#    AdaDelta “AdaDelta”——一种”鲁棒的学习率方法“，是基于梯度的优化方法。
#    Adaptive Gradient “AdaGrad”——自适应梯度方法。
#    Adam “Adam”——一种基于梯度的优化方法。
#    Nesterov’s Accelerated Gradient “Nesterov”——Nesterov的加速梯度法，作为凸优化中最理想的方法，其收敛速度非常快。
#    RMSprop “RMSProp”——一种基于梯度的优化方法。
#    假设有N个训练样本，batch的大小为batch_size，将所有样本处理完一次为一个epoch，则需要迭代次数为N/batch_size < test_interval
#    如果有M个训练样本，则需要迭代次数为：M/batch_size < test_iter
path = "prototxt/"
solver_file = path + 'solver.prototxt'     #solver文件保存位置
sp={}
sp['train_net']='"' + path + 'train_net.prototxt' + '"'  # 训练配置文件
sp['test_net']= '"' + path + 'test_net.prototxt' + '"'     # 测试配置文件
sp['test_iter']='325'                  # 测试迭代次数
sp['test_interval']='650'              # 测试间隔
sp['base_lr']='0.001'                  # 基础学习率
sp['display']='500'                    # 屏幕日志显示间隔
sp['max_iter']='10000'                 # 最大迭代次数
sp['lr_policy']='"'+'step'+'"'                 # 学习率变化规律
sp['gamma']='0.1'                      # 学习率变化指数
sp['momentum']='0.9'                   # 动量
sp['weight_decay']='0.0005'            # 权值衰减
sp['stepsize']='3000'                 # 学习率变化频率
sp['snapshot']='1000'                   # 保存model间隔
sp['snapshot_prefix']='"snapshot"'      # 保存的model前缀              # 是否使用gpu
# sp['solver_type']='SGD'                # 优化算法

with open(solver_file, 'w') as f:
    for key, value in sorted(sp.items()):
        if not(type(value) is str):
            raise TypeError('All solver parameters must be strings')
        f.write('%s: %s\n' % (key, value))