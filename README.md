This project is partially based on [**pytorch-complex-tensor **](https://github.com/williamFalcon/pytorch-complex-tensor). 

Basicly, this new class is a wrapper of (...,2) shape.

Different from original complex datastructue, I use (...,2) tensor shape as complex data structure. Sp

- The real part for a complex tentor T is T[...,0]
- The image part is T[...,1]

This shape is consistent to the native pytorch complex operation [`torch.fft`](https://pytorch.org/docs/stable/torch.html). 

Technically, such a complex implament is a good interface by showing complex form but calculating through  pytorch tensor operation.

For Monomer operator, below list finished:

.exp()           ![img](https://latex.codecogs.com/gif.latex?z%3Dx+iy%20%5Cquad%20e%5Ez%3De%5E%7Bx+iy%7D%3De%5Ex%5Ccos%28y%29+ie%5Exsin%28y%29)

.log()           ![img](https://latex.codecogs.com/gif.latex?z%3D%5Crho%20e%5E%7Bi%5Ctheta%7D%20%5Cquad%20%5Clog%28z%29%3D%5Clog%28%5Crho%20e%5E%7Bi%5Ctheta%7D%29%3D%5Clog%20%5Crho+i%20%5Ctheta)

.conj()          ![img](https://latex.codecogs.com/gif.latex?z%3Dx+iy%20%5Cquad%20z%5E*%3Dx-iy)

For Binary operator, both requirement satisfied 

- one is \< complex tensor, complex tensor \> 
- another is \< complex tensor, real tensor \> 

| Operation   | status |
| ----------- | :----: |
| addition    | Finish |
| subtraction | Finish |
| multiply    | Finish |
| division    | Finish |
| mm          | Finish |
| mv          | Finish |



##### Gradient

​The numerical gradient of complex function z -> f(z) can be achieved by using real Automatic differentiation.  

​Set z=x+i y and f(z)=u(x,y)+iv(x,y) is holomorphic function, according to [Cauchy–Riemann equations](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations), the gradient of f(z) is 

![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20f%28z%29%7D%7B%5Cpartial%20z%7D%3D%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20x%7D-i%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20y%7D%3D%5Cfrac%7B%5Cpartial%20v%7D%7B%5Cpartial%20y%7D+i%5Cfrac%7B%5Cpartial%20v%7D%7B%5Cpartial%20x%7D)

So, in order to get the differentiation, we only need the Automatic differentiation of real part of f(z).

The  [**pytorch-complex-tensor **](https://github.com/williamFalcon/pytorch-complex-tensor) version set the complex graident is 

![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20f%28z%29%7D%7B%5Cpartial%20z%7D%3D%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20x%7D+i%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20y%7D)

which is wrong for complex number.

​For real number X, the gradient of f(z,X) is 

![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20f%28z%2CX%29%7D%7B%5Cpartial%20x%7D%3D%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20x%7D+i%5Cfrac%7B%5Cpartial%20v%7D%7B%5Cpartial%20y%7D)

##### Example

```
from pytorch_complex_tensor import ComplexScalar,ComplexTensor
import numpy as np
import torch
```

You can use any shape complex numpy tensor as import or just convert  (...,2) shape torch tensor to ComplexTensor.

```python
a=np.array([[1+1j,2+2j,4+4j],[3+3j,4+4j,4+4j],[3+3j,4+4j,4+4j]])
b=torch.randint(3,[3,3])
v=torch.randint(3,[3])
c=torch.randint(3,[3,3,2])
```

The ComplexTensor will auto change the (...,2) form tensor to its (...) shape complex form.

```python
ca=ComplexTensor(a)
#tensor([[1.+1.j, 2.+2.j, 4.+4.j],
#        [3.+3.j, 4.+4.j, 4.+4.j],
#        [3.+3.j, 4.+4.j, 4.+4.j]], dtype=complex64)
```

All the operation is available and return complex form 

```python
cc=ComplexTensor(c)
xy=ca.mm(cc)
#tensor([[-3.+15.j,  3.+23.j, -3.+11.j],
#        [-5.+25.j,  7.+31.j, -3.+19.j],
#        [-5.+25.j,  7.+31.j, -3.+19.j]], dtype=complex64)
```

The operation between complex number and real number is available.

```python
xy=ca.mm(b.float())#required float number 
#tensor([[12.+12.j, 13.+13.j, 12.+12.j],
#        [16.+16.j, 19.+19.j, 16.+16.j],
#        [16.+16.j, 19.+19.j, 16.+16.j]], dtype=complex64)
```

The gradient for a complex scalar is available.

```
ca=ComplexTensor(a)
ca.requires_grad = True
cc=ComplexTensor(c)
xy=ca.mm(cc).sum()
#(-5+199j)
xy.backward()
print(ca.grad)
#tensor([[3.+2.j, 4.+4.j, 3.+4.j],
#       [3.+2.j, 4.+4.j, 3.+4.j],
#       [3.+2.j, 4.+4.j, 3.+4.j]], dtype=complex64)
```

