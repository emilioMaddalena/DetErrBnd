# DetErrBnd

This code is the supplementary material for the paper

```
@article{maddalena2021embedded,
  title={Deterministic error bounds for kernel-based learning techniques under bounded noise},
  author={E. T. Maddalena, P. Scharnhorst and C. N. Jones},
  journal={Automatica (accepted)},
  volume={-},
  pages={1--18},
  year={2021}
}
```

## Description :books:

- We derive deterministic error-bounds for two non-parametric kernel models: kernel ridge regression (KRR) and support vector regression (SVR). Our setting is that of bounded measurement noise. Our expressions only involve solving a single box-constrained quadratic program off-line, but otherwise are given in closed-form.

- By deterministic, we mean that the ground-truth cannot assume values outside the uncertainty envelope. 

- The derived bounds require an estimate of the unknown function complexity as measured by the chosen kernel, through its RKHS. If you would like to know how to approximate this quantity from data alone, please check "Robust Uncertainty Bounds in Reproducing Kernel Hilbert Spaces: A Convex Optimization Approach", Appendix A.

## Dependencies  :building_construction:

- YALMIP (https://yalmip.github.io/)
- Gurobi (https://www.gurobi.com/)

![alt text](https://github.com/emilioMaddalena/DetErrBnd/blob/master/pics/1a.png)

![alt text](https://github.com/emilioMaddalena/DetErrBnd/blob/master/pics/1b.png)
