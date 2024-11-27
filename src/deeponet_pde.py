from __future__ import absolute_import, division, print_function

import argparse
import itertools

import deepxde as dde
import numpy as np
import tensorflow as tf
from spaces import GRF, FiniteChebyshev, FinitePowerSeries
from system import ADVDSystem, CVCSystem, DRSystem, LTSystem, ODESystem
from utils import (mean_squared_error_outlier, merge_values, safe_test,
                   trim_to_65535)


def test_u_lt(nn, system, T, m, model, data, u, fname):
    """Test Legendre transform"""
    sensors = np.linspace(-1, 1, num=m)
    sensor_value = u(sensors)
    s = system.eval_s(sensor_value)
    ns = np.arange(system.npoints_output)[:, None]
    X_test = [np.tile(sensor_value, (system.npoints_output, 1)), ns]
    y_test = s
    if nn != "deeponet":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt("test/u_" + fname, sensor_value)
    np.savetxt("test/s_" + fname, np.hstack((ns, y_test, y_pred)))


def test_u_ode(nn, system, T, m, model, data, u, fname, num=100):
    """Test ODE"""
    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values = u(sensors)
    x = np.linspace(0, T, num=num)[:, None]
    X_test = [np.tile(sensor_values.T, (num, 1)), x]
    y_test = system.eval_s_func(u, x)
    if nn != "deeponet":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt(fname, np.hstack((x, y_test, y_pred)))
    print("L2relative error:", dde.metrics.l2_relative_error(y_test, y_pred))


def test_u_dr(nn, system, T, m, model, data, u, fname):
    """Test Diffusion-reaction"""
    sensors = np.linspace(0, 1, num=m)
    sensor_value = u(sensors)
    s = system.eval_s(sensor_value)
    xt = np.array(list(itertools.product(range(m), range(system.Nt))))
    xt = xt * [1 / (m - 1), T / (system.Nt - 1)]
    X_test = [np.tile(sensor_value, (m * system.Nt, 1)), xt]
    y_test = s.reshape([m * system.Nt, 1])
    if nn != "deeponet":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt(fname, np.hstack((xt, y_test, y_pred)))


def test_u_cvc(nn, system, T, m, model, data, u, fname):
    """Test Advection"""
    sensors = np.linspace(0, 1, num=m)
    sensor_value = u(sensors)
    s = system.eval_s(sensor_value)
    xt = np.array(list(itertools.product(range(m), range(system.Nt))))
    xt = xt * [1 / (m - 1), T / (system.Nt - 1)]
    X_test = [np.tile(sensor_value, (m * system.Nt, 1)), xt]
    y_test = s.reshape([m * system.Nt, 1])
    if nn != "deeponet":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt("test/u_" + fname, sensor_value)
    np.savetxt("test/s_" + fname, np.hstack((xt, y_test, y_pred)))


def test_u_advd(nn, system, T, m, model, data, u, fname):
    """Test Advection-diffusion"""
    sensors = np.linspace(0, 1, num=m)
    sensor_value = u(sensors)
    s = system.eval_s(sensor_value)
    xt = np.array(list(itertools.product(range(m), range(system.Nt))))
    xt = xt * [1 / (m - 1), T / (system.Nt - 1)]
    X_test = [np.tile(sensor_value, (m * system.Nt, 1)), xt]
    y_test = s.reshape([m * system.Nt, 1])
    if nn != "deeponet":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt("test/u_" + fname, sensor_value)
    np.savetxt("test/s_" + fname, np.hstack((xt, y_test, y_pred)))


def lt_system(npoints_output):
    """Legendre transform"""
    return LTSystem(npoints_output)


def ode_system(T):
    """ODE"""

    def g(s, u, x):
        # Antiderivative
        return u
        # Nonlinear ODE
        # return -s**2 + u
        # Gravity pendulum
        k = 1
        return [s[1], - k * np.sin(s[0]) + u]

    #s0 = [0]
    s0 = [0, 0]  # Gravity pendulum
    return ODESystem(g, s0, T)


def dr_system(T, npoints_output):
    """Diffusion-reaction"""
    D = 0.01
    k = 0.01
    Nt = 100
    return DRSystem(D, k, T, Nt, npoints_output)


def cvc_system(T, npoints_output):
    """Advection"""
    f = None
    g = None
    Nt = 100
    return CVCSystem(f, g, T, Nt, npoints_output)


def advd_system(T, npoints_output):
    """Advection-diffusion"""
    # source term
    f = 10
    # boundary condition
    g = [3 , 0]
    Nt = 100
    return ADVDSystem(f, g, T, Nt, npoints_output)


def run(problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test):
    # space_test = GRF(1, length_scale=0.1, N=1000, interp="cubic")

    X_train, y_train = system.gen_operator_data(space, m, num_train)
    X_test, y_test = system.gen_operator_data(space, m, num_test)

    if nn != "deeponet":
        X_train = merge_values(X_train)
        X_test = merge_values(X_test)

    # np.savez_compressed("train.npz", X_train0=X_train[0], X_train1=X_train[1], y_train=y_train)
    # np.savez_compressed("test.npz", X_test0=X_test[0], X_test1=X_test[1], y_test=y_test)
    # return

    # d = np.load("train.npz")
    # X_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]
    # d = np.load("test.npz")
    # X_test, y_test = (d["X_test0"], d["X_test1"]), d["y_test"]

    X_test_trim = trim_to_65535(X_test)[0]
    y_test_trim = trim_to_65535(y_test)[0]
    

    if nn == "deeponet":
        X_train = [arr.astype(dtype=np.float32) for arr in X_train]
        y_train = [arr.astype(dtype=np.float32) for arr in y_train]
            
        y_train = np.concatenate([np.array(i, dtype=np.float32) for i in y_train])
        y_test = np.concatenate([np.array(i, dtype=np.float32) for i in y_test])
        data = dde.data.Triple(
            X_train=X_train, 
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    else:
        data = dde.data.DataSet(
            X_train=X_train, y_train=y_train, X_test=X_test_trim, y_test=y_test_trim
        )

    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=[mean_squared_error_outlier])
    checker = dde.callbacks.ModelCheckpoint(
        "model/model", save_better_only=True, period=1000
    )
    losshistory, train_state = model.train(epochs=epochs, callbacks=[checker])
    print("# Parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    model.restore(f"model/model-{train_state.best_step}.weights.h5", verbose=1)
    safe_test(model, data, X_test, y_test)

    tests = [
        (lambda x: x, "x.dat"),
        (lambda x: np.sin(np.pi * x), "sinx.dat"),
        (lambda x: np.sin(2 * np.pi * x), "sin2x.dat"),
        (lambda x: x * np.sin(2 * np.pi * x), "xsin2x.dat"),
    ]
    for u, fname in tests:
        if problem == "lt":
            test_u_lt(nn, system, T, m, model, data, u, fname)
        elif problem == "ode":
            test_u_ode(nn, system, T, m, model, data, u, fname)
        elif problem == "dr":
            test_u_dr(nn, system, T, m, model, data, u, fname)
        elif problem == "cvc":
            test_u_cvc(nn, system, T, m, model, data, u, fname)
        elif problem == "advd":
            test_u_advd(nn, system, T, m, model, data, u, fname)

    if problem == "lt":
        features = space.random(10)
        sensors = np.linspace(0, 2, num=m)[:, None]
        u = space.eval_u(features, sensors)
        for i in range(u.shape[0]):
            test_u_lt(nn, system, T, m, model, data, lambda x: u[i], str(i) + ".dat")

    if problem == "cvc":
        features = space.random(10)
        sensors = np.linspace(0, 1, num=m)[:, None]
        # Case I Input: V(sin^2(pi*x))
        u = space.eval_u(features, np.sin(np.pi * sensors) ** 2)
        # Case II Input: x*V(x)
        # u = sensors.T * space.eval_u(features, sensors)
        # Case III/IV Input: V(x)
        # u = space.eval_u(features, sensors)
        for i in range(u.shape[0]):
            test_u_cvc(nn, system, T, m, model, data, lambda x: u[i], str(i) + ".dat")

    if problem == "advd":
        features = space.random(10)
        sensors = np.linspace(0, 1, num=m)[:, None]
        u = space.eval_u(features, np.sin(np.pi * sensors) ** 2)
        for i in range(u.shape[0]):
            test_u_advd(nn, system, T, m, model, data, lambda x: u[i], str(i) + ".dat")


def main(args):
    # Problems:
    # - "lt": Legendre transform
    # - "ode": Antiderivative, Nonlinear ODE, Gravity pendulum
    # - "dr": Diffusion-reaction
    # - "cvc": Advection
    # - "advd": Advection-diffusion
    problem = args.problem
    T = args.t
    if problem == "lt":
        npoints_output = 20
        system = lt_system(npoints_output)
    elif problem == "ode":
        system = ode_system(T)
    elif problem == "dr":
        npoints_output = 100
        system = dr_system(T, npoints_output)
    elif problem == "cvc":
        npoints_output = 100
        system = cvc_system(T, npoints_output)
    elif problem == "advd":
        npoints_output = 100
        system = advd_system(T, npoints_output)

    # Function space
    # space = FinitePowerSeries(N=100, M=1)
    # space = FiniteChebyshev(N=20, M=1)
    # space = GRF(2, length_scale=0.2, N=2000, interp="cubic")  # "lt"
    space = GRF(1, length_scale=0.2, N=1000, interp="cubic")
    # space = GRF(T, length_scale=0.2, N=1000 * T, interp="cubic")

    # Hyperparameters
    m = args.m
    num_train = args.num_train
    num_test = args.num_test
    lr = args.lr
    epochs = args.epochs

    # Network
    nn = args.nn
    activation = args.activation
    initializer = args.init  # "He normal" or "Glorot normal"
    dim_x = 1 if problem in ["ode", "lt"] else 2
    if nn == "deeponet":
        net = dde.maps.DeepONet(
            [m, 40, 40],
            [dim_x, 40, 40],
            activation,
            initializer,
        )
    elif nn == "fnn":
        net = dde.maps.FNN([m + dim_x] + [100] * 2 + [1], activation, initializer)
    elif nn == "resnet":
        net = dde.maps.ResNet(m + dim_x, 1, 128, 2, activation, initializer)

    run(problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test)


if __name__ == "__main__":
    
    def process_initializer(value):
        if value == 'Glorot':
            return 'Glorot normal'
        else:
            return 'He normal'
 
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--problem', type=str, choices=['lt','ode', 'dr', 'cvc', 'advd'], help="Type of differential equation", required=True)
    parser.add_argument('-t', type=int, default=1, help="Final time in domain (defualt=1)")
    parser.add_argument('-m', type=int, help="number of sensors", required=True)
    parser.add_argument('--num-train', type=int, help="number of train data", required=True)
    parser.add_argument('--num-test', type=int, help="number of test data", required=True)
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate (default=1e-3)")
    parser.add_argument('--epochs', type=int, help="number of epochs", required=True)
    parser.add_argument('--nn', type=str, choices=['fnn', 'deeponet', 'resnet'], help="Type of nueral network", default='deeponet')
    parser.add_argument('--activation', type=str, choices=['elu', 'gelu', 'relu', 'selu', 'sigmoid', 'silu', 'sin', 'swish', 'tanh'], help="Activation function", default='relu')
    parser.add_argument(
    '--init',
    type=str,
    choices=['He', 'Glorot'],
    default='Glorot',  # Default initializer
    help="Specify the initializer (choose from 'He [normal]', 'Glorot [normal]')"
    )
    parser.add_argument('--stacked',type=str,default=False,help="Specify whether to use stacked architecture (usage for --nn is 'deeponet')")
    args = parser.parse_args()
    args.init = process_initializer(args.init)
    args.stacked = args.stacked.lower() == 'true'
    main(args)


