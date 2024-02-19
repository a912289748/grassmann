import torch
from torch.autograd import Function as F

class QRs(F):
    @staticmethod
    def forward(ctx,P):


        [Q,R] = torch.qr(P)
        # n1,n2,w1,w2 = P.shape
        # for i in range(n1):
        #     for j in range(n2):
        #         P[i][j] = 1
        ctx.save_for_backward(Q,R)
        return Q

    @staticmethod
    def backward(ctx,dx):
        [Qs,Rs] = ctx.saved_variables
        n1,n2,w1,w2 = dx.shape
        Y = torch.zeros(n1,n2,w2,w1)
        for i in range(n1):
            for j in range(n2):
                T = dx[i][j]
                Q = Qs[i][j]
                R = Rs[i][j]

                m = T.shape[0]
                S = torch.eye(m) - torch.matmul(Q, Q.T)
                dzdx_to = torch.matmul(Q.T, T)
                dzdx_t1 = torch.tril(dzdx_to) - torch.diag_embed(torch.diag(dzdx_to))
                dzdx_t2 = torch.tril(dzdx_to.T) - torch.diag_embed(torch.diag(dzdx_to.T))

                dzdx = (torch.matmul(S.T, T) + torch.matmul(torch.matmul(Q, (dzdx_t1 - dzdx_t2)), torch.inverse(R))).T
                # dzdx = (torch.matmul(S.T, T) + torch.matmul(torch.matmul(Q, (dzdx_t1 - dzdx_t2)), torch.inverse(R+torch.diag_embed(torch.trace(R).repeat(1,R.shape[0]))))).T
                Y[i][j] = dzdx
                # dzdx = (S'*dLdQ+Q*(dzdx_t1-dzdx_t2))*(inv(R))';

        # Q = ctx.saved_variables[1]
        # R = ctx.saved_variables[2]

        return Y.transpose(2,3)

class Orthmap(F):
    @staticmethod
    def forward(ctx, X):

        U,S,V = torch.svd(X)
        ctx.save_for_backward(X,U,S)

        return U[:, :, :, :10]

    @staticmethod
    def calculate_grad_svd(U, S, p, D, dzdy):
        pass
    @staticmethod
    def backward(ctx,P):
        X = ctx.saved_variables[0]
        U = ctx.saved_variables[1]
        S = ctx.saved_variables[2]
        n1,n2,w,h =  P.shape
        n11,n22,w1,h1 = X.shape
        Y = torch.zeros(n11,n22,w1,h1)
        for i3 in range(n1):
            for i4 in range(n2):
                U_t = U[i3,i4,:,:]
                S_t = torch.diag(S[i3,i4,:])

                D = S_t.shape[0]
                p = 10
                dzdy = P[i3,i4,:,:]


                diagS = torch.diag(S_t)
                Dmin = len(diagS)
                ind = torch.arange(1, Dmin + 1)
                dLdC = torch.zeros(D, D).double()


                # dLdC[A == 1] = dzdy
                dLdC[:D,:p]=dzdy
                dLdU =dLdC.clone()
                if sum(ind) == 1:
                    pass
                else:
                    e = diagS
                    dim = e.size(0)
                    s = e.view(dim, 1)
                    ones = torch.ones(1, dim, dtype=torch.float64)
                    s = s @ ones
                    k = 1 / (s - s.t())
                    k[torch.eye(dim) > 0] = 0
                    k[k == float("Inf")] = 0
                    indices = (diagS < 1e-10).nonzero().squeeze()
                    if len(indices) != 0:
                        k[indices, indices.unsqueeze(1)] = 0
                    ss= k.t().mul(U_t.t()@dLdU)
                    ss = (ss + ss.t()) / 2
                    Y[i3,i4,:,:]=U_t@ss@U_t.t()
                # Y[i3,i4,:,:]=calculate_grad_svd(U_t,S_t,10,S_t.shape[1],P[i3,i4,:,:])
        # print(n2)
        # print(w)
        # print(h)
        # dzdy = torch.bmm(P.view(-1,w,h),X.view(-1,w1,h1)).view(n1,n2,w,h1)

        return Y

    @staticmethod
    def calculate_grad_svd( U, S, p, D, dzdy):
        pass

class ProjMap(F):
    @staticmethod
    def forward(ctx, X):
        ctx.save_for_backward(X)
        # ch1, ch2, w, h = X.shape
        # P1 = X.view(-1, w, h)
        # P2 = torch.bmm(P1, P1.transpose(1, 2))
        # # n1, n2, w1, h1 = P2.shape
        # # print(P2.shape)
        # P2 = P2.view(X.shape[0], X.shape[1], w, w)
        return X@X.transpose(-1,-2)
    @staticmethod
    def backward(ctx,P):
        X = ctx.saved_variables[0]
        n1,n2,w,h =  P.shape
        n11,n22,w1,h1 = X.shape
        res = torch.zeros(n1,n2,w,h1).double()
        #P = d_t
        for ix in range(n1):
            d_t = P[ix,:,:,:]
            for iy in range(n2):
                res[ix,iy,:,:]=2.*(torch.matmul(d_t[iy,:,:],X[ix,iy,:,:]))
        # print(n2)
        # print(w)
        # print(h)

        return res