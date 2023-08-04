import torch 
import sympy

N, D_in, H, D_out = 64, 1000, 100, 10


x = torch.randn(N, D_in, device="cpu", dtype=None)
y = torch.randn(N, D_out, device="cpu", dtype=None)


w1 = torch.randn(D_in, H, device="cpu", dtype=None)
w2 = torch.randn(H, D_out, device="cpu", dtype=None)

learning_rate = 1e-6
for t in range(500):
    
    # mm is matrix multiplication
    h = x.mm(w1)

    #clamps so that the min input value is 0
    
    h_relu = h.clamp(min=0)

    #input h_relu
    y_pred = h_relu.mm(w2)

    
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
