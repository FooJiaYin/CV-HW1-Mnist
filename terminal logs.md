# Environment Setup
```
[YIn.newLaptop] → ssh foojiayin@140.114.91.198
Last login: Sat Aug 24 14:14:52 2019 from 10.11.11.2
[foojiayin@hades01 ~]$ which python
/usr/local/bin/python
[foojiayin@hades01 ~]$ which python3
/usr/local/bin/python3
[foojiayin@hades01 ~]$ virtualenv -p /usr/local/bin/python3 env
Running virtualenv with interpreter /usr/local/bin/python3
Using base prefix '/usr/local'
New python executable in /home/ELSALab/foojiayin/env/bin/python3
Also creating executable in /home/ELSALab/foojiayin/env/bin/python
Installing setuptools, pip, wheel...
done.
[foojiayin@hades01 ~]$ source env/bin/activate
(env) [foojiayin@hades01 ~]$ pip list
DEPRECATION: Python 2.7 will reach the end of its life on January 1st, 2020. Please upgrade your Python as Python 2.7 won't be maintained after that date. A future version of pip will drop support for Python 2.7. More details about Python 2 support in pip, can be found at https://pip.pypa.io/en/latest/development/release-process/#python-2-support
Package    Version
---------- -------
pip        19.2.2
setuptools 41.2.0
wheel      0.33.6
(env) [foojiayin@hades01 ~]$ pip3 install torch torchvision
Collecting torch
  Downloading https://files.pythonhosted.org/packages/30/57/d5cceb0799c06733eefce80c395459f28970ebb9e896846ce96ab579a3f1/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl (748.8MB)
     |████████████████████████████████| 748.9MB 34kB/s
Collecting torchvision
  Downloading https://files.pythonhosted.org/packages/06/e6/a564eba563f7ff53aa7318ff6aaa5bd8385cbda39ed55ba471e95af27d19/torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl (8.8MB)
     |████████████████████████████████| 8.8MB 3.0MB/s
Collecting numpy (from torch)
  Downloading https://files.pythonhosted.org/packages/19/b9/bda9781f0a74b90ebd2e046fde1196182900bd4a8e1ea503d3ffebc50e7c/numpy-1.17.0-cp36-cp36m-manylinux1_x86_64.whl (20.4MB)
     |████████████████████████████████| 20.4MB 2.5MB/s
Collecting six (from torchvision)
  Downloading https://files.pythonhosted.org/packages/73/fb/00a976f728d0d1fecfe898238ce23f502a721c0ac0ecfedb80e0d88c64e9/six-1.12.0-py2.py3-none-any.whl
Collecting pillow>=4.1.1 (from torchvision)
  Downloading https://files.pythonhosted.org/packages/14/41/db6dec65ddbc176a59b89485e8cc136a433ed9c6397b6bfe2cd38412051e/Pillow-6.1.0-cp36-cp36m-manylinux1_x86_64.whl (2.1MB)
     |████████████████████████████████| 2.1MB 12.5MB/s
Installing collected packages: numpy, torch, six, pillow, torchvision
Successfully installed numpy-1.17.0 pillow-6.1.0 six-1.12.0 torch-1.2.0 torchvision-0.4.0
(env) [foojiayin@hades01 ~]$ pip3 install tensorflow
Collecting tensorflow
  Downloading https://files.pythonhosted.org/packages/de/f0/96fb2e0412ae9692dbf400e5b04432885f677ad6241c088ccc5fe7724d69/tensorflow-1.14.0-cp36-cp36m-manylinux1_x86_64.whl (109.2MB)
     |████████████████████████████████| 109.2MB 8.0MB/s
Collecting wrapt>=1.11.1 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/23/84/323c2415280bc4fc880ac5050dddfb3c8062c2552b34c2e512eb4aa68f79/wrapt-1.11.2.tar.gz
Collecting termcolor>=1.1.0 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz
Collecting keras-preprocessing>=1.0.5 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/28/6a/8c1f62c37212d9fc441a7e26736df51ce6f0e38455816445471f10da4f0a/Keras_Preprocessing-1.1.0-py2.py3-none-any.whl (41kB)
     |████████████████████████████████| 51kB 8.0MB/s
Collecting tensorflow-estimator<1.15.0rc0,>=1.14.0rc0 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/3c/d5/21860a5b11caf0678fbc8319341b0ae21a07156911132e0e71bffed0510d/tensorflow_estimator-1.14.0-py2.py3-none-any.whl (488kB)
     |████████████████████████████████| 491kB 15.6MB/s
Collecting gast>=0.2.0 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/4e/35/11749bf99b2d4e3cceb4d55ca22590b0d7c2c62b9de38ac4a4a7f4687421/gast-0.2.2.tar.gz
Collecting grpcio>=1.8.6 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/30/31/6397193572c081e0fd1fec86a7a6b7ac497c27226281e7cb32f8c3705069/grpcio-1.23.0-cp36-cp36m-manylinux1_x86_64.whl (2.2MB)
     |████████████████████████████████| 2.2MB 15.0MB/s
Collecting protobuf>=3.6.1 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/eb/f4/a27952733796330cd17c17ea1f974459f5fefbbad119c0f296a6d807fec3/protobuf-3.9.1-cp36-cp36m-manylinux1_x86_64.whl (1.2MB)
     |████████████████████████████████| 1.2MB 13.3MB/s
Collecting tensorboard<1.15.0,>=1.14.0 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/91/2d/2ed263449a078cd9c8a9ba50ebd50123adf1f8cfbea1492f9084169b89d9/tensorboard-1.14.0-py3-none-any.whl (3.1MB)
     |████████████████████████████████| 3.2MB 20.1MB/s
Requirement already satisfied: wheel>=0.26 in ./env/lib/python3.6/site-packages (from tensorflow) (0.33.6)
Collecting keras-applications>=1.0.6 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/71/e3/19762fdfc62877ae9102edf6342d71b28fbfd9dea3d2f96a882ce099b03f/Keras_Applications-1.0.8-py3-none-any.whl (50kB)
     |████████████████████████████████| 51kB 3.8MB/s
Collecting astor>=0.6.0 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/d1/4f/950dfae467b384fc96bc6469de25d832534f6b4441033c39f914efd13418/astor-0.8.0-py2.py3-none-any.whl
Collecting google-pasta>=0.1.6 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/d0/33/376510eb8d6246f3c30545f416b2263eee461e40940c2a4413c711bdf62d/google_pasta-0.1.7-py3-none-any.whl (52kB)
     |████████████████████████████████| 61kB 7.5MB/s
Collecting absl-py>=0.7.0 (from tensorflow)
  Downloading https://files.pythonhosted.org/packages/da/3f/9b0355080b81b15ba6a9ffcf1f5ea39e307a2778b2f2dc8694724e8abd5b/absl-py-0.7.1.tar.gz (99kB)
     |████████████████████████████████| 102kB 8.9MB/s
Requirement already satisfied: numpy<2.0,>=1.14.5 in ./env/lib/python3.6/site-packages (from tensorflow) (1.17.0)
Requirement already satisfied: six>=1.10.0 in ./env/lib/python3.6/site-packages (from tensorflow) (1.12.0)
Requirement already satisfied: setuptools in ./env/lib/python3.6/site-packages (from protobuf>=3.6.1->tensorflow) (41.2.0)
Collecting werkzeug>=0.11.15 (from tensorboard<1.15.0,>=1.14.0->tensorflow)
  Downloading https://files.pythonhosted.org/packages/d1/ab/d3bed6b92042622d24decc7aadc8877badf18aeca1571045840ad4956d3f/Werkzeug-0.15.5-py2.py3-none-any.whl (328kB)
     |████████████████████████████████| 337kB 12.7MB/s
Collecting markdown>=2.6.8 (from tensorboard<1.15.0,>=1.14.0->tensorflow)
  Downloading https://files.pythonhosted.org/packages/c0/4e/fd492e91abdc2d2fcb70ef453064d980688762079397f779758e055f6575/Markdown-3.1.1-py2.py3-none-any.whl (87kB)
     |████████████████████████████████| 92kB 8.4MB/s
Collecting h5py (from keras-applications>=1.0.6->tensorflow)
  Downloading https://files.pythonhosted.org/packages/30/99/d7d4fbf2d02bb30fb76179911a250074b55b852d34e98dd452a9f394ac06/h5py-2.9.0-cp36-cp36m-manylinux1_x86_64.whl (2.8MB)
     |████████████████████████████████| 2.8MB 17.0MB/s
Building wheels for collected packages: wrapt, termcolor, gast, absl-py
  Building wheel for wrapt (setup.py) ... done
  Created wheel for wrapt: filename=wrapt-1.11.2-cp36-cp36m-linux_x86_64.whl size=67267 sha256=355282ace6b4e6c92e10ec3612480e8470e99182ca668237436399436a1dd96e
  Stored in directory: /home/ELSALab/foojiayin/.cache/pip/wheels/d7/de/2e/efa132238792efb6459a96e85916ef8597fcb3d2ae51590dfd
  Building wheel for termcolor (setup.py) ... done
  Created wheel for termcolor: filename=termcolor-1.1.0-cp36-none-any.whl size=4832 sha256=45aa07dca4a889fc9fcb373bbcd6dd2274e9e6150413f3f6713b5901313da9d8
  Stored in directory: /home/ELSALab/foojiayin/.cache/pip/wheels/7c/06/54/bc84598ba1daf8f970247f550b175aaaee85f68b4b0c5ab2c6
  Building wheel for gast (setup.py) ... done
  Created wheel for gast: filename=gast-0.2.2-cp36-none-any.whl size=7540 sha256=688699453c12f8833fa806994e3380b7ccf6bcc629b531df15235d50ef9a8dec
  Stored in directory: /home/ELSALab/foojiayin/.cache/pip/wheels/5c/2e/7e/a1d4d4fcebe6c381f378ce7743a3ced3699feb89bcfbdadadd
  Building wheel for absl-py (setup.py) ... done
  Created wheel for absl-py: filename=absl_py-0.7.1-cp36-none-any.whl size=117847 sha256=2cdb70392401b00889e38bf63a64f2062543e51edbbcccdca880bf5ddab84584
  Stored in directory: /home/ELSALab/foojiayin/.cache/pip/wheels/ee/98/38/46cbcc5a93cfea5492d19c38562691ddb23b940176c14f7b48
Successfully built wrapt termcolor gast absl-py
Installing collected packages: wrapt, termcolor, keras-preprocessing, tensorflow-estimator, gast, grpcio, protobuf, werkzeug, absl-py, markdown, tensorboard, h5py, keras-applications, astor, google-pasta, tensorflow
Successfully installed absl-py-0.7.1 astor-0.8.0 gast-0.2.2 google-pasta-0.1.7 grpcio-1.23.0 h5py-2.9.0 keras-applications-1.0.8 keras-preprocessing-1.1.0 markdown-3.1.1 protobuf-3.9.1 tensorboard-1.14.0 tensorflow-1.14.0 tensorflow-estimator-1.14.0 termcolor-1.1.0 werkzeug-0.15.5 wrapt-1.11.2
(env) [foojiayin@hades01 ~]$ pip install matplotlib
Collecting matplotlib
  Downloading https://files.pythonhosted.org/packages/57/4f/dd381ecf6c6ab9bcdaa8ea912e866dedc6e696756156d8ecc087e20817e2/matplotlib-3.1.1-cp36-cp36m-manylinux1_x86_64.whl (13.1MB)
     |████████████████████████████████| 13.1MB 3.2MB/s
Collecting kiwisolver>=1.0.1 (from matplotlib)
  Downloading https://files.pythonhosted.org/packages/f8/a1/5742b56282449b1c0968197f63eae486eca2c35dcd334bab75ad524e0de1/kiwisolver-1.1.0-cp36-cp36m-manylinux1_x86_64.whl (90kB)
     |████████████████████████████████| 92kB 8.6MB/s
Requirement already satisfied: numpy>=1.11 in ./env/lib/python3.6/site-packages (from matplotlib) (1.17.0)
Collecting python-dateutil>=2.1 (from matplotlib)
  Downloading https://files.pythonhosted.org/packages/41/17/c62faccbfbd163c7f57f3844689e3a78bae1f403648a6afb1d0866d87fbb/python_dateutil-2.8.0-py2.py3-none-any.whl (226kB)
     |████████████████████████████████| 235kB 16.1MB/s
Collecting cycler>=0.10 (from matplotlib)
  Downloading https://files.pythonhosted.org/packages/f7/d2/e07d3ebb2bd7af696440ce7e754c59dd546ffe1bbe732c8ab68b9c834e61/cycler-0.10.0-py2.py3-none-any.whl
Collecting pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 (from matplotlib)
  Downloading https://files.pythonhosted.org/packages/11/fa/0160cd525c62d7abd076a070ff02b2b94de589f1a9789774f17d7c54058e/pyparsing-2.4.2-py2.py3-none-any.whl (65kB)
     |████████████████████████████████| 71kB 11.4MB/s
Requirement already satisfied: setuptools in ./env/lib/python3.6/site-packages (from kiwisolver>=1.0.1->matplotlib) (41.2.0)
Requirement already satisfied: six>=1.5 in ./env/lib/python3.6/site-packages (from python-dateutil>=2.1->matplotlib) (1.12.0)
Installing collected packages: kiwisolver, python-dateutil, cycler, pyparsing, matplotlib
Successfully installed cycler-0.10.0 kiwisolver-1.1.0 matplotlib-3.1.1 pyparsing-2.4.2 python-dateutil-2.8.0
(env) [foojiayin@hades01 ~]$ deactivate
```

# Testing: linear regression
```
[foojiayin@hades01 ~]$ ssh -X hades02
Warning: Permanently added 'hades02,10.11.11.2' (ECDSA) to the list of known hosts.
Last login: Sat Aug 24 14:23:27 2019 from 10.11.11.1
[foojiayin@hades02 ~]$ source env/bin/activate
(env) [foojiayin@hades02 ~]$ cd test
(env) [foojiayin@hades02 test]$ python3 linear-regression.py
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
...
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
WARNING: Logging before flag parsing goes to stderr.
W0824 14:32:09.743089 139800290920256 deprecation_wrapper.py:119] From linear-regression.py:14: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.

W0824 14:32:09.779193 139800290920256 deprecation_wrapper.py:119] From linear-regression.py:20: The name tf.train.GradientDescentOptimizer is deprecated. Please use tf.compat.v1.train.GradientDescentOptimizer instead.

W0824 14:32:09.821597 139800290920256 deprecation_wrapper.py:119] From linear-regression.py:24: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

W0824 14:32:09.822438 139800290920256 deprecation_wrapper.py:119] From linear-regression.py:27: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

2019-08-24 14:32:09.822788: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-08-24 14:32:09.827353: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600335000 Hz
2019-08-24 14:32:09.828357: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x44aab90 executing computations on platform Host. Devices:
2019-08-24 14:32:09.828383: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2019-08-24 14:32:09.835173: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
0 [-0.62733316] [0.2944061]
20 [-0.21928023] [0.46622002]
40 [-0.07885867] [0.39311555]
60 [-0.00019547] [0.35216275]
80 [0.04387115] [0.32922125]
100 [0.06855698] [0.31636956]
120 [0.08238583] [0.30917013]
140 [0.09013265] [0.30513707]
160 [0.09447237] [0.30287775]
180 [0.09690347] [0.3016121]
200 [0.09826535] [0.30090308]
```
# Mnist_1.py

# Mnist_2.py
```
(env) [foojiayin@hades02 test]$ python3 mnist_2.py
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
WARNING: Logging before flag parsing goes to stderr.
W0824 15:52:16.909361 139692672677696 deprecation.py:323] From mnist_2.py:8: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
W0824 15:52:16.909807 139692672677696 deprecation.py:323] From /home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please write your own downloading logic.
W0824 15:52:16.910296 139692672677696 deprecation.py:323] From /home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting MNIST_data/train-images-idx3-ubyte.gz
W0824 15:52:17.143629 139692672677696 deprecation.py:323] From /home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
W0824 15:52:17.146102 139692672677696 deprecation.py:323] From /home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.one_hot on tensors.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
W0824 15:52:17.214493 139692672677696 deprecation.py:323] From /home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
W0824 15:52:17.349701 139692672677696 deprecation_wrapper.py:119] From mnist_2.py:15: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

W0824 15:52:17.354034 139692672677696 deprecation.py:323] From /home/ELSALab/foojiayin/env/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py:193: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
W0824 15:52:17.355036 139692672677696 deprecation_wrapper.py:119] From mnist_2.py:19: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

2019-08-24 15:52:17.355527: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-08-24 15:52:17.362355: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600335000 Hz
2019-08-24 15:52:17.362967: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55672b0 executing computations on platform Host. Devices:
2019-08-24 15:52:17.363001: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2019-08-24 15:52:17.366678: W tensorflow/compiler/jit/mark_for_compilation_pass.cc:1412] (One-time warning): Not using XLA:CPU for cluster because envvar TF_XLA_FLAGS=--tf_xla_cpu_global_jit was not set.  If you want XLA:CPU, either set that envvar, or use experimental_jit_scope to enable XLA:CPU.  To confirm that XLA is active, pass --vmodule=xla_compilation_cache=1 (as a proper command-line flag, not via TF_XLA_FLAGS) or set the envvar XLA_FLAGS=--xla_hlo_profile.
W0824 15:52:17.367291 139692672677696 deprecation_wrapper.py:119] From mnist_2.py:24: The name tf.truncated_normal is deprecated. Please use tf.random.truncated_normal instead.

W0824 15:52:17.387392 139692672677696 deprecation_wrapper.py:119] From mnist_2.py:35: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

W0824 15:52:17.422527 139692672677696 deprecation.py:506] From mnist_2.py:59: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
W0824 15:52:17.449953 139692672677696 deprecation_wrapper.py:119] From mnist_2.py:67: The name tf.log is deprecated. Please use tf.math.log instead.

W0824 15:52:17.453883 139692672677696 deprecation_wrapper.py:119] From mnist_2.py:68: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

step 0, training accuracy 0.16
step 100, training accuracy 0.84
step 200, training accuracy 0.96
step 300, training accuracy 0.84
step 400, training accuracy 0.98
step 500, training accuracy 0.96
step 600, training accuracy 0.96
step 700, training accuracy 0.94
step 800, training accuracy 0.94
step 900, training accuracy 0.94
step 1000, training accuracy 0.94
step 1100, training accuracy 1
step 1200, training accuracy 0.9
step 1300, training accuracy 1
step 1400, training accuracy 0.96
step 1500, training accuracy 0.98
step 1600, training accuracy 0.94
step 1700, training accuracy 0.98
step 1800, training accuracy 0.96
step 1900, training accuracy 0.96
step 2000, training accuracy 0.94
step 2100, training accuracy 0.98
step 2200, training accuracy 0.96
step 2300, training accuracy 0.96
step 2400, training accuracy 0.98
step 2500, training accuracy 1
step 2600, training accuracy 0.98
step 2700, training accuracy 1
step 2800, training accuracy 0.94
step 2900, training accuracy 0.96
step 3000, training accuracy 0.96
step 3100, training accuracy 0.94
step 3200, training accuracy 1
step 3300, training accuracy 0.98
step 3400, training accuracy 1
step 3500, training accuracy 0.96
step 3600, training accuracy 1
step 3700, training accuracy 0.98
step 3800, training accuracy 1
step 3900, training accuracy 0.98
step 4000, training accuracy 0.94
step 4100, training accuracy 1
step 4200, training accuracy 1
step 4300, training accuracy 1
step 4400, training accuracy 1
step 4500, training accuracy 1
step 4600, training accuracy 0.98
step 4700, training accuracy 1
step 4800, training accuracy 1
step 4900, training accuracy 1
step 5000, training accuracy 0.98
step 5100, training accuracy 0.96
step 5200, training accuracy 1
step 5300, training accuracy 1
step 5400, training accuracy 0.96
step 5500, training accuracy 0.98
step 5600, training accuracy 0.98
step 5700, training accuracy 1
step 5800, training accuracy 1
step 5900, training accuracy 0.98
step 6000, training accuracy 1
step 6100, training accuracy 1
step 6200, training accuracy 1
step 6300, training accuracy 0.96
step 6400, training accuracy 1
step 6500, training accuracy 1
step 6600, training accuracy 1
step 6700, training accuracy 1
step 6800, training accuracy 0.98
step 6900, training accuracy 1
step 7000, training accuracy 1
step 7100, training accuracy 1
step 7200, training accuracy 1
step 7300, training accuracy 0.98
step 7400, training accuracy 1
step 7500, training accuracy 1
step 7600, training accuracy 0.94
step 7700, training accuracy 1
step 7800, training accuracy 1
step 7900, training accuracy 1
step 8000, training accuracy 1
step 8100, training accuracy 1
step 8200, training accuracy 1
step 8300, training accuracy 1
step 8400, training accuracy 1
step 8500, training accuracy 1
step 8600, training accuracy 1
step 8700, training accuracy 1
step 8800, training accuracy 1
step 8900, training accuracy 1
step 9000, training accuracy 1
step 9100, training accuracy 1
step 9200, training accuracy 1
step 9300, training accuracy 1
step 9400, training accuracy 1
step 9500, training accuracy 1
step 9600, training accuracy 0.98
step 9700, training accuracy 1
step 9800, training accuracy 1
step 9900, training accuracy 1
step 10000, training accuracy 1
step 10100, training accuracy 0.98
step 10200, training accuracy 1
step 10300, training accuracy 1
step 10400, training accuracy 1
step 10500, training accuracy 1
step 10600, training accuracy 1
step 10700, training accuracy 1
step 10800, training accuracy 1
step 10900, training accuracy 1
step 11000, training accuracy 1
step 11100, training accuracy 1
step 11200, training accuracy 1
step 11300, training accuracy 0.96
step 11400, training accuracy 1
step 11500, training accuracy 1
step 11600, training accuracy 1
step 11700, training accuracy 1
step 11800, training accuracy 1
step 11900, training accuracy 1
step 12000, training accuracy 1
step 12100, training accuracy 1
step 12200, training accuracy 1
step 12300, training accuracy 1
step 12400, training accuracy 1
step 12500, training accuracy 1
step 12600, training accuracy 1
step 12700, training accuracy 1
step 12800, training accuracy 0.96
step 12900, training accuracy 0.98
step 13000, training accuracy 1
step 13100, training accuracy 1
/step 13200, training accuracy 1
step 13300, training accuracy 1
step 13400, training accuracy 1
step 13500, training accuracy 1
step 13600, training accuracy 1
step 13700, training accuracy 1
step 13800, training accuracy 1
step 13900, training accuracy 1
step 14000, training accuracy 1
step 14100, training accuracy 1
step 14200, training accuracy 1
step 14300, training accuracy 1
step 14400, training accuracy 1
step 14500, training accuracy 0.98
step 14600, training accuracy 1
step 14700, training accuracy 0.98
step 14800, training accuracy 0.98
step 14900, training accuracy 1
step 15000, training accuracy 1
step 15100, training accuracy 1
step 15200, training accuracy 1
step 15300, training accuracy 1
step 15400, training accuracy 1
step 15500, training accuracy 1
step 15600, training accuracy 1
step 15700, training accuracy 1
step 15800, training accuracy 1
step 15900, training accuracy 1
step 16000, training accuracy 1
step 16100, training accuracy 1
step 16200, training accuracy 1
step 16300, training accuracy 1
step 16400, training accuracy 1
step 16500, training accuracy 1
step 16600, training accuracy 1
step 16700, training accuracy 1
step 16800, training accuracy 1
step 16900, training accuracy 1
step 17000, training accuracy 1
step 17100, training accuracy 1
step 17200, training accuracy 1
step 17300, training accuracy 1
step 17400, training accuracy 1
step 17500, training accuracy 1
step 17600, training accuracy 1
step 17700, training accuracy 1
step 17800, training accuracy 1
step 17900, training accuracy 1
step 18000, training accuracy 1
step 18100, training accuracy 1
step 18200, training accuracy 1
step 18300, training accuracy 1
step 18400, training accuracy 1
step 18500, training accuracy 1
step 18600, training accuracy 0.98
step 18700, training accuracy 1
step 18800, training accuracy 1
step 18900, training accuracy 1
step 19000, training accuracy 1
step 19100, training accuracy 1
step 19200, training accuracy 1
step 19300, training accuracy 1
step 19400, training accuracy 1
step 19500, training accuracy 1
step 19600, training accuracy 1
step 19700, training accuracy 1
step 19800, training accuracy 1
step 19900, training accuracy 1
2019-08-24 16:05:57.271553: W tensorflow/core/framework/allocator.cc:107] Allocation of 1003520000 exceeds 10% of system memory.
test accuracy 0.9915
```
