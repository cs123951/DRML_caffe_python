import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from layers import multilabel_datalayer
import caffe
def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net, num_batches, batch_size=64):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        #        batch_size, label_num
        gts = np.array(net.blobs['label'].data)
        score_array = np.array(net.blobs['score'].data)
        ests = np.zeros([score_array.shape[0], score_array.shape[1]])
        for i in range(gts.shape[0]):
            for j in range(gts.shape[1]):
                if score_array[i, j] > 0:
                    ests[i, j] = 1
                else:
                    ests[i, j] = -1
        for gt, est in zip(gts, ests):  # for each ground truth and
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)


def run_solvers(solver, niter=10000, disp_interval=10, test_interval=10,
                test_iter=20):
    train_loss = np.zeros(niter  // disp_interval)
    test_loss = np.zeros(niter // test_interval)
    # test_acc = np.zeros(np.ceil(niter * 1.0 / test_interval))

    _train_loss = 0;_test_loss = 0;_accuracy = 0

    solver.step(1)

    for it in range(niter):
        solver.step(1)
        _train_loss += solver.net.blobs['loss'].data

        if it % disp_interval == 0:
            train_loss[it//disp_interval] = _train_loss / disp_interval
            _train_loss = 0

        if it % test_interval == 0:
            for test_it in range(test_iter):
                solver.test_nets[0].forward()
                _test_loss += solver.test_nets[0].blobs['loss'].data

            test_loss[it // test_interval] = _test_loss / test_iter
            _test_loss = 0
    return train_loss, test_loss




solver_file = 'prototxt/solver.prototxt'
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.AdamSolver(solver_file)
disp_interval = 10
test_interval = 10

train_loss, test_loss = run_solvers(solver)
plt.plot(disp_interval * np.arange(len(train_loss)), train_loss,'r')
plt.plot(test_interval * np.arange(len(test_loss)), test_loss, 'g')
plt.title('Trainloss  VS Iters')
plt.xlabel('Iters')
plt.ylabel('Trainloss')
plt.savefig('loss.png')
# solver.solve()