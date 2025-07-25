import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def apply_smoothing_differential_filter(img, kernel_size=5, sigma=2):
    """
    Compute smoothed image gradients using a simple difference filter on a blurred image.
    Returns dx, dy as float64 arrays.
    """
    img_f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (kernel_size, kernel_size), sigmaX=sigma).astype(np.float64)
    kx = np.array([[-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    dx = cv2.filter2D(blurred, cv2.CV_64F, kx)
    dy = cv2.filter2D(blurred, cv2.CV_64F, ky)
    return dx, dy

def estimate_by_gauss_newton2(img_input, img_transformed,
                               theta_init=0.0, scale_init=1.0,
                               threshold=1e-5, max_iter=100):
    # initial params (radians)
    theta = np.deg2rad(theta_init)
    scale = scale_init
    H, W = img_input.shape[:2]
    yv, xv = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x = xv - W/2; y = yv - H/2
    mask = (x**2 + y**2) <= (min(W, H)/2)**2

    I = img_transformed.astype(np.float64)
    I_org = img_input
    # histories
    theta_hist, scale_hist = [], []
    grad_theta_hist, grad_scale_hist = [], []
    cost_hist, delta_norm_hist = [], []

    for _ in range(max_iter):
        # transform and warp
        a = scale * np.cos(theta); b = scale * np.sin(theta)
        M = np.array([[a, -b, 0],[b, a, 0],[0,0,1]], dtype=np.float64)
        M_inv = np.linalg.inv(M)
        pts = np.stack([x.flatten(), y.flatten(), np.ones_like(x.flatten())])
        uv = M_inv @ pts
        xi = np.round(uv[0] + W/2).astype(int)
        yi = np.round(uv[1] + H/2).astype(int)
        I_prime = np.zeros_like(I)
        valid = (xi>=0)&(xi<W)&(yi>=0)&(yi<H)
        fp = I_prime.flatten(); fp[valid] = I_org[yi[valid], xi[valid]]
        I_prime = fp.reshape(H, W)

        # gradients and residual
        I_dx, I_dy = apply_smoothing_differential_filter(I_prime)
        diff = I_prime - I
        
        # partial derivatives
        dx_dtheta = -scale*(x*np.sin(theta) + y*np.cos(theta))
        dy_dtheta =  scale*(x*np.cos(theta) - y*np.sin(theta))
        dx_dscale = x*np.cos(theta) - y*np.sin(theta)
        dy_dscale = x*np.sin(theta) + y*np.cos(theta)

        # Jacobian components
        A = (I_dx*dx_dtheta + I_dy*dy_dtheta)[mask]
        B = (I_dx*dx_dscale + I_dy*dy_dscale)[mask]
        r = diff[mask]

        # gradient and Hessian approximation
        Jt = np.sum(r * A); Js = np.sum(r * B)
        Jtt = np.sum(A**2); Jss = np.sum(B**2); Jts = np.sum(A*B)
        
        # cost
        cost = 0.5 * np.sum(r**2)

        # store histories
        theta_hist.append(theta); scale_hist.append(scale)
        grad_theta_hist.append(Jt); grad_scale_hist.append(Js)
        cost_hist.append(cost)

        # solve update
        Hm = np.array([[Jtt, Jts],[Jts, Jss]])
        g = np.array([Jt, Js])
        try:
            delta = np.linalg.solve(Hm, g)
        except np.linalg.LinAlgError:
            print("Singular Hessian; stopping.")
            break

        theta -= delta[0]; scale -= delta[1]
        delta_norm_hist.append(np.linalg.norm(delta))
        if np.linalg.norm(delta) < threshold:
            break

    return (theta, scale,
            theta_hist, scale_hist,
            grad_theta_hist, grad_scale_hist,
            cost_hist, delta_norm_hist)

def main():
    if len(sys.argv)<3:
        print("Usage: python gauss2.py input transformed [theta0] [scale0]")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    t0 = float(sys.argv[3]) if len(sys.argv)>=4 else 0.0
    s0 = float(sys.argv[4]) if len(sys.argv)>=5 else 1.0
    img_i = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)
    img_t = cv2.imread(outp, cv2.IMREAD_GRAYSCALE)
    if img_i is None or img_t is None:
        print("Error: cannot open images."); sys.exit(1)

    (theta, scale,
     th_h, sc_h,
     gt_h, gs_h,
     cost_h, dn_h) = estimate_by_gauss_newton2(img_i, img_t, t0, s0)

    print(f"Estimated rotation (deg): {np.rad2deg(theta):.4f}")
    print(f"Estimated scale: {scale:.4f}")

    # save histories
    it = np.arange(len(th_h))
    plt.figure(); plt.plot(it, np.rad2deg(th_h)); plt.xlabel('Iteration'); plt.ylabel('Theta (deg)'); plt.grid(); plt.savefig('theta_history.png')
    plt.figure(); plt.plot(it, sc_h); plt.xlabel('Iteration'); plt.ylabel('Scale'); plt.grid(); plt.savefig('scale_history.png')
    plt.figure(); plt.plot(it, gt_h); plt.xlabel('Iteration'); plt.ylabel('Grad Theta'); plt.grid(); plt.savefig('grad_theta.png')
    plt.figure(); plt.plot(it, gs_h); plt.xlabel('Iteration'); plt.ylabel('Grad Scale'); plt.grid(); plt.savefig('grad_scale.png')
    plt.figure(); plt.plot(it, cost_h); plt.xlabel('Iteration'); plt.ylabel('Cost'); plt.grid(); plt.savefig('cost_history.png')
    plt.figure(); plt.plot(it, dn_h); plt.xlabel('Iteration'); plt.ylabel('Delta Norm'); plt.grid(); plt.savefig('delta_history.png')
    plt.figure(); plt.plot(np.rad2deg(th_h), sc_h, marker='o'); plt.xlabel('Theta (deg)'); plt.ylabel('Scale'); plt.grid(); plt.savefig('trajectory.png')

    print("Saved: theta_history.png, scale_history.png, grad_theta.png, grad_scale.png, cost_history.png, delta_history.png, trajectory.png")

if __name__ == "__main__": main()
